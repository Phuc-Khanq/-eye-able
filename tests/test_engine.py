"""The demo script through engine.main.Engine: the real threaded EventBus, the Whisper worker
thread, the mode manager and audio_in, with fakes only for Whisper, the mic and UI-Venus.

Run with: python -m unittest tests.test_engine
"""
import threading
import time
import unittest
from types import SimpleNamespace

import numpy as np

from engine.core import messages, topics
from engine.core.events import EventBus
from engine.core.mode_manager import Mode
from engine.core.types import GazePoint, Gesture
from engine.main import Engine, play_wav
from engine.voice.dictation import Dictation
from engine.voice.mic import MicSettings
from tests.helpers import FakeWhisperModel
from tests.test_audio_in import FakeSD, FakeVad
from tests.test_demo_flow import EXPECTED_OS, SCRIPT

SETTINGS = SimpleNamespace(pause_seconds=1.0)
MIC = MicSettings(0, "fake mic", "MME", 16000, 1)


class Recorder:
    def __init__(self, bus):
        self.log, self.lock = [], threading.Lock()
        self.transcripts = threading.Semaphore(0)
        self.utterances = threading.Semaphore(0)
        for topic in (topics.OS_CLICK, topics.OS_TYPE, topics.OS_KEY, topics.OS_SCROLL, topics.TO_UI):
            bus.subscribe(topic, lambda p, t=topic: self._add(t, p))
        # Subscribed after the mode manager, so this runs once it has handled the transcript.
        bus.subscribe(topics.TRANSCRIPT, lambda p: self.transcripts.release())
        bus.subscribe(topics.UTTERANCE, lambda p: self.utterances.release())

    def _add(self, topic, payload):
        with self.lock:
            self.log.append((topic, payload))

    def os_actions(self):
        with self.lock:
            return [(t, p) for t, p in self.log if t.startswith("os.")]

    def statuses(self):
        with self.lock:
            return [p for t, p in self.log if t == topics.TO_UI and p["type"] == messages.STATUS]


def make_engine(bus, **kwargs):
    return Engine(bus, SETTINGS, dictation=Dictation(model=FakeWhisperModel()),
                  mic_settings=MIC, vad=FakeVad(min_silence_s=1.0), warm_up=False, **kwargs)


def run_once():
    bus = EventBus()
    engine = make_engine(bus, sd=FakeSD())
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
            def InputStream(self, **kwargs):
                raise RuntimeError("No mic format worked")

        bus = EventBus()
        engine = make_engine(bus, sd=NoMic())
        rec = Recorder(bus)
        engine.start()
        try:
            self.assertTrue(any("No microphone" in s["text"] for s in rec.statuses()))
            bus.publish(topics.UTTERANCE, "start dictation")   # the rest still works
            self.assertTrue(rec.transcripts.acquire(timeout=5))
            self.assertEqual(engine.mode_manager.mode, Mode.DICTATION)
        finally:
            engine.stop()

    def test_wav_instead_of_the_mic(self):
        bus = EventBus()
        engine = make_engine(bus)
        rec = Recorder(bus)
        engine.start(listen=False)
        t = np.arange(16000) / 16000
        speech = np.concatenate([np.zeros(8000), 0.3 * np.sin(2 * np.pi * 220 * t)]).astype(np.float32)
        try:
            play_wav(engine.audio_in, speech, 16000, threading.Event(), realtime=False)
            self.assertTrue(rec.utterances.acquire(timeout=5))
        finally:
            engine.stop()

    def test_dictation_clean_up_is_off_unless_asked(self):
        engine = make_engine(EventBus(), sd=FakeSD())
        self.assertIsNone(engine.rewrite)
        self.assertIsNone(engine.mode_manager.rewrite_wait_s)

    def test_dictation_clean_up_wired_end_to_end(self):
        """rewrite=True: the preview shows the clean-up, and "yes" types the clean text."""
        from tests.test_rewrite_worker import FakeRewriter   # upper-cases the text

        bus = EventBus()
        engine = make_engine(bus, sd=FakeSD(), rewrite=True, rewriter=FakeRewriter())
        rec = Recorder(bus)
        results = threading.Semaphore(0)
        bus.subscribe(topics.REWRITE_RESULT, lambda r: results.release())
        engine.start()
        try:
            self.assertEqual(engine.mode_manager.rewrite_wait_s, 1.5)
            for text in ("start dictation", "hello world"):
                bus.publish(topics.UTTERANCE, text)
                self.assertTrue(rec.transcripts.acquire(timeout=5))
            self.assertTrue(results.acquire(timeout=5))
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:   # the result reaches the mode manager on the worker's thread
                previews = [p for t, p in rec.log if t == topics.TO_UI and p["type"] == messages.DICTATION_PREVIEW]
                if previews and previews[-1].get("cleaned"):
                    break
                time.sleep(0.02)
            last = previews[-1]
            self.assertTrue(last.get("cleaned"))
            self.assertEqual(" ".join(w["text"] for w in last["words"]), "HELLO WORLD")
            self.assertEqual(last["original"], "hello world")
            bus.publish(topics.UTTERANCE, "yes")
            self.assertTrue(rec.transcripts.acquire(timeout=5))
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and not any(t == topics.OS_TYPE for t, _ in rec.os_actions()):
                time.sleep(0.02)
            typed = [p["text"] for t, p in rec.os_actions() if t == topics.OS_TYPE]
            self.assertEqual(typed, ["HELLO WORLD "])
        finally:
            engine.stop()


if __name__ == "__main__":
    unittest.main()
