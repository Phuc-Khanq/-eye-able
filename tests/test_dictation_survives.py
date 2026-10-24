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
