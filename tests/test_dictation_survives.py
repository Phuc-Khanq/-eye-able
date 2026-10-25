"""Dictation keeps working after one bad utterance. In the live run on Oct 24 a word with a letter the
Windows console couldn't print ("\\u0103") raised UnicodeEncodeError inside the mouth-gate log line; the
Whisper thread died and nothing more was ever heard.

Run with: python -m unittest tests.test_dictation_survives
"""
import time
import unittest
from types import SimpleNamespace

from engine.core import messages, topics
from engine.voice.dictation import Dictation, DictationWorker
from tests.helpers import FakeBus, FakeWhisperModel


class AlwaysMoving:
    def moving_fraction(self, start_ms, end_ms):
        return 1.0


class TestDictationSurvives(unittest.TestCase):
    def test_a_log_line_that_cannot_print_does_not_stop_dictation(self):
        def cp1252_console(line):
            line.encode("cp1252")          # what print() does on a Windows console or pipe

        bus = FakeBus()
        w = DictationWorker(bus, Dictation(model=FakeWhisperModel()), mouth=AlwaysMoving(),
                            settings=SimpleNamespace(), log=cp1252_console)
        w.start(warm_up=False)
        try:
            w.on_speech_span({"start_ms": 0.0, "end_ms": 1000.0})
            w.on_utterance("xin chăo")              # this one's log line can't be printed
            w.on_speech_span({"start_ms": 2000.0, "end_ms": 3000.0})
            w.on_utterance("start dictation")
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline and not any(
                    t["text"].lower().startswith("start dictation") for t in bus.of(topics.TRANSCRIPT)):
                time.sleep(0.02)
        finally:
            w.stop()
        texts = [t["text"].lower() for t in bus.of(topics.TRANSCRIPT)]
        self.assertTrue(any(t.startswith("start dictation") for t in texts), texts)
        errors = [m["text"] for m in bus.ui(messages.STATUS) if m["level"] == "error"]
        self.assertTrue(any("skipped one utterance" in e for e in errors), errors)


if __name__ == "__main__":
    unittest.main()


class TestResumeWhilePaused(unittest.TestCase):
    """A quiet "resume" while paused gets past the mouth gate (Khang's live test, Oct 24)."""

    class Still:
        def moving_fraction(self, start_ms, end_ms):
            return 0.0                      # the camera sees a face whose mouth isn't moving

    def worker(self):
        bus = FakeBus()
        w = DictationWorker(bus, Dictation(model=FakeWhisperModel()), mouth=self.Still(),
                            settings=SimpleNamespace(), log=lambda s: None)
        return bus, w

    def say(self, w, text):
        w.on_speech_span({"start_ms": 0.0, "end_ms": 1000.0})
        w.process(text, {"start_ms": 0.0, "end_ms": 1000.0})

    def test_resume_counts_while_paused(self):
        bus, w = self.worker()
        bus.publish(topics.TO_UI, messages.make(messages.MODE, mode="paused"))
        self.say(w, "resume")
        self.assertEqual([t["text"].lower().strip(". ") for t in bus.of(topics.TRANSCRIPT)], ["resume"])

    def test_other_words_are_still_gated_while_paused(self):
        bus, w = self.worker()
        bus.publish(topics.TO_UI, messages.make(messages.MODE, mode="paused"))
        self.say(w, "open chrome")
        self.assertEqual(bus.of(topics.TRANSCRIPT), [])

    def test_resume_is_gated_when_not_paused(self):
        bus, w = self.worker()
        bus.publish(topics.TO_UI, messages.make(messages.MODE, mode="pointer"))
        self.say(w, "resume")
        self.assertEqual(bus.of(topics.TRANSCRIPT), [])
