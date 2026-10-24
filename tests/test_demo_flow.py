"""The whole demo script, end to end, 5 times in a row, with fakes for camera, mic and UI-Venus.

Run with: python -m unittest tests.test_demo_flow
"""
import unittest

from engine.core import topics
from engine.core.mode_manager import Mode, ModeManager
from engine.core.types import GazePoint, Gesture, TargetResult
from engine.voice.dictation import Dictation, DictationWorker
from tests.helpers import FakeBus, FakeWhisperModel

# (kind, value)
SCRIPT = [
    ("look", (800, 450)),
    ("say", "click compose"),
    ("ui_venus", TargetResult("Compose", True, 812, 440, 300.0)),
    ("gesture", "nod"),                                   # click Compose
    ("say", "start dictation"),
    ("say", "hi mom, home at seven? tonight"),
    ("gesture", "shake"),                                 # misheard: throw it away
    ("say", "hi mom, home at seven tonight"),
    ("gesture", "nod"),                                   # insert
    ("say", "send it"),                                   # asks first
    ("gesture", "nod"),                                   # send
    ("say", "stop dictation"),
    ("gesture", "tilt_right"),                            # scroll down
    ("say", "pause"),
    ("say", "scroll down"),                               # ignored while paused
    ("say", "resume"),
]

EXPECTED_OS = [
    (topics.OS_CLICK, {"x": 812, "y": 440}),
    (topics.OS_TYPE, {"text": "hi mom, home at seven tonight "}),
    (topics.OS_KEY, {"key": "enter"}),
    (topics.OS_SCROLL, {"direction": "down"}),
]


def run_once() -> FakeBus:
    bus = FakeBus()
    worker = DictationWorker(bus, Dictation(model=FakeWhisperModel()))
    mm = ModeManager(bus)
    for kind, value in SCRIPT:
        if kind == "look":
            bus.publish(topics.GAZE, GazePoint(0, *value, 0.9))
        elif kind == "say":
            worker.process(value)
        elif kind == "gesture":
            bus.publish(topics.GESTURE, Gesture(0, value))
        elif kind == "ui_venus":
            bus.publish(topics.TARGET_RESULT, value)
    assert mm.mode == Mode.POINTER and not mm.preview and mm.question is None
    return bus


class TestDemoFlow(unittest.TestCase):
    def test_five_runs_in_a_row(self):
        for i in range(5):
            with self.subTest(run=i + 1):
                bus = run_once()
                self.assertEqual([(t, p) for t, p in bus.log if t.startswith("os.")], EXPECTED_OS)


if __name__ == "__main__":
    unittest.main()
