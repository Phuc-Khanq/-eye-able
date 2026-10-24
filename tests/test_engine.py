"""The demo script through engine.main.Engine: the real threaded EventBus, the Whisper worker
thread, the mode manager and audio_in, with fakes only for Whisper, the mic and UI-Venus.

Run with: python -m unittest tests.test_engine
"""
import threading
import unittest
from types import SimpleNamespace

import numpy as np

from engine.core import messages, topics
from engine.core.events import EventBus
from engine.core.mode_manager import Mode
from engine.core.types import GazePoint, Gesture
from engine.main import Engine
from engine.voice.audio_in import ArraySource, EnergyVad
from engine.voice.dictation import Dictation
from tests.helpers import FakeWhisperModel
from tests.test_demo_flow import EXPECTED_OS, SCRIPT

SETTINGS = SimpleNamespace(pause_seconds=1.5, mic_gain=1.0)


class Recorder:
    def __init__(self, bus):
        self.log, self.lock = [], threading.Lock()
        self.transcripts = threading.Semaphore(0)
        for topic in (topics.OS_CLICK, topics.OS_TYPE, topics.OS_KEY, topics.OS_SCROLL, topics.TO_UI):
            bus.subscribe(topic, lambda p, t=topic: self._add(t, p))
        # Subscribed after the mode manager, so this runs once it has handled the transcript.
        bus.subscribe(topics.TRANSCRIPT, lambda p: self.transcripts.release())

    def _add(self, topic, payload):
        with self.lock:
            self.log.append((topic, payload))

    def os_actions(self):
        with self.lock:
            return [(t, p) for t, p in self.log if t.startswith("os.")]


def run_once():
    bus = EventBus()
    silence = ArraySource(np.zeros(16000, np.float32))
    engine = Engine(bus, SETTINGS, dictation=Dictation(model=FakeWhisperModel()),
                    audio_source=silence, vad=EnergyVad(), warm_up=False)
    rec = Recorder(bus)
    engine.start()
    try:
        for kind, value in SCRIPT:
            if kind == "look":
                bus.publish(topics.GAZE, GazePoint(0, *value, 0.9))
            elif kind == "say":
                bus.publish(topics.UTTERANCE, value)          # the fake Whisper "hears" the text
                assert rec.transcripts.acquire(timeout=5), f"no transcript for {value!r}"
            elif kind == "gesture":
                bus.publish(topics.GESTURE, Gesture(0, value))
            elif kind == "ui_venus":
                bus.publish(topics.TARGET_RESULT, value)
        mm = engine.mode_manager
        assert mm.mode == Mode.POINTER and not mm.preview and mm.question is None
        return rec
    finally:
        engine.stop()


class TestEngine(unittest.TestCase):
    def test_demo_five_times_in_a_row_on_real_threads(self):
        for i in range(5):
            with self.subTest(run=i + 1):
                rec = run_once()
                self.assertEqual(rec.os_actions(), EXPECTED_OS)
                first_ui = next(p for t, p in rec.log if t == topics.TO_UI)
                self.assertEqual(first_ui, messages.make(messages.MODE, mode="pointer"))

    def test_a_missing_mic_does_not_stop_the_engine(self):
        class NoMic:
            rate = 48000

            def open(self, on_block):
                raise RuntimeError("No mic format worked")

        bus = EventBus()
        engine = Engine(bus, SETTINGS, dictation=Dictation(model=FakeWhisperModel()),
                        audio_source=NoMic(), vad=EnergyVad(), warm_up=False)
        rec = Recorder(bus)
        engine.start()
        try:
            bus.publish(topics.UTTERANCE, "start dictation")   # voice still works from other sources
            self.assertTrue(rec.transcripts.acquire(timeout=5))
            self.assertEqual(engine.mode_manager.mode, Mode.DICTATION)
            statuses = [p for t, p in rec.log if t == topics.TO_UI and p["type"] == messages.STATUS]
            self.assertTrue(any("No microphone" in s["text"] for s in statuses))
        finally:
            engine.stop()


if __name__ == "__main__":
    unittest.main()
