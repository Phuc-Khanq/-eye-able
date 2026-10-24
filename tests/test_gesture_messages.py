"""Run with: python -m unittest tests.test_gesture_messages

The UI gets a "gesture" message for every gesture the mode manager acts on, in any mode, so it can
flash "raise seen" on a click or a yes. Gestures it ignores send nothing.
"""
import unittest
from types import SimpleNamespace

from engine.core import messages, topics
from engine.core.mode_manager import Mode, ModeManager
from engine.core.types import GazePoint, Gesture, TargetResult
from tests.helpers import FakeBus
from tests.test_mouth_gate import FakeMouth


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class Base(unittest.TestCase):
    def setUp(self, mouth=None):
        self.bus, self.clock = FakeBus(), Clock()
        self.mm = ModeManager(self.bus, clock=self.clock, mouth=mouth, log=lambda s: None,
                              settings=SimpleNamespace(scroll_idle_s=10.0))
        self.bus.publish(topics.GAZE, GazePoint(0, 500, 300, 0.9))

    def say(self, text):
        self.bus.publish(topics.TRANSCRIPT, {"text": text, "words": [{"text": w, "conf": 0.95} for w in text.split()],
                                             "ms": 0.0})

    def gesture(self, kind, t_ms=100_000):
        self.bus.publish(topics.GESTURE, Gesture(t_ms=t_ms, type=kind))

    def shown(self):
        return [m["gesture"] for m in self.bus.ui(messages.GESTURE)]


class TestActedOnIsShown(Base):
    def test_a_raise_snap_clicks_and_is_shown(self):
        self.gesture("brow_raise")
        self.assertEqual(self.shown(), ["brow_raise"])
        self.assertEqual(self.bus.of(topics.TARGET_REQUEST)[-1].snap, True)

    def test_a_blink_too(self):
        self.gesture("blink")
        self.assertEqual(self.shown(), ["blink"])

    def test_mouth_open_right_clicks_and_is_shown(self):
        self.gesture("mouth_open")
        self.assertEqual(self.shown(), ["mouth_open"])
        self.assertEqual(self.bus.of(topics.OS_CLICK)[-1]["button"], "right")

    def test_answers_are_shown(self):
        for kind, state in (("brow_raise", "confirmed"), ("nod", "confirmed"), ("shake", "cancelled")):
            with self.subTest(kind=kind):
                self.setUp()
                self.say("click send")
                req = self.bus.of(topics.TARGET_REQUEST)[-1]
                self.bus.publish(topics.TARGET_RESULT, TargetResult(req.label, True, 410, 220))
                self.gesture(kind)
                self.assertEqual(self.shown(), [kind])
                self.assertEqual(self.bus.ui(messages.CONFIRM)[-1]["state"], state)

    def test_tilts_that_scroll(self):
        self.gesture("tilt_right")
        self.assertEqual(self.shown(), ["tilt_right"])
        self.assertEqual(self.bus.of(topics.OS_SCROLL), [{"direction": "down"}])

    def test_a_raise_that_inserts_the_preview(self):
        self.say("start dictation")
        self.say("hi mom")
        self.gesture("brow_raise")
        self.assertEqual(self.shown(), ["brow_raise"])
        self.assertEqual([p["text"] for p in self.bus.of(topics.OS_TYPE)], ["hi mom "])

    def test_a_raise_that_leaves_scroll_mode(self):
        self.say("scroll")
        self.gesture("brow_raise")
        self.assertEqual(self.shown(), ["brow_raise"])
        self.assertEqual(self.mm.mode, Mode.POINTER)

    def test_menu_is_as_before_once_each_and_a_shake_closes_it(self):
        self.say("menu")
        for kind in ("tilt_left", "nod", "brow_raise"):
            self.gesture(kind)
        self.assertEqual(self.shown(), ["tilt_left", "nod", "brow_raise"])
        self.gesture("shake")
        self.assertEqual(self.shown(), ["tilt_left", "nod", "brow_raise", "shake"])
        self.assertEqual(self.mm.mode, Mode.POINTER)


class TestIgnoredIsNotShown(Base):
    def test_paused(self):
        self.say("pause")
        for kind in ("brow_raise", "nod", "mouth_open", "tilt_left"):
            self.gesture(kind)
        self.assertEqual(self.shown(), [])

    def test_a_raise_while_talking(self):
        self.setUp(mouth=FakeMouth(moving=[(98_000, 101_000)]))
        self.gesture("brow_raise")
        self.assertEqual(self.shown(), [])

    def test_mouth_open_while_the_mic_hears_speech(self):
        self.bus.publish(topics.TO_UI, messages.make(messages.LISTENING, level=0.4, speaking=True))
        self.gesture("mouth_open")
        self.assertEqual(self.shown(), [])
        self.assertEqual(self.bus.of(topics.OS_CLICK), [])

    def test_a_raise_while_dictating_with_nothing_waiting(self):
        self.say("start dictation")
        self.gesture("brow_raise")
        self.assertEqual(self.shown(), [])

    def test_gestures_with_no_meaning_in_the_mode(self):
        self.gesture("nod")                  # Pointer: a nod means nothing with nothing asked
        self.say("scroll")
        self.gesture("tilt_left")            # Scroll mode: the head scrolls, its gestures mean nothing
        self.assertEqual(self.shown(), [])


if __name__ == "__main__":
    unittest.main()
