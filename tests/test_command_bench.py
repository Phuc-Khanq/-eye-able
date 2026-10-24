"""Run with: python -m unittest tests.test_command_bench  (the scoring only; no audio, no Whisper)"""
import unittest

from engine.voice.command_bench import PHRASES, run, score
from engine.voice.dictation import Dictation, words_to_text
from tests.helpers import FakeWhisperModel


class TestScore(unittest.TestCase):
    def test_right_text_and_command(self):
        self.assertEqual(score("send it", "pointer", ["Send it."]),
                         {"text": True, "command": True, "empty": False, "became": "send_it"})

    def test_right_command_wrong_text(self):
        s = score("yes", "pointer", ["Yeah."])
        self.assertEqual((s["text"], s["command"]), (False, True))

    def test_wrong_command(self):
        s = score("no", "pointer", ["Know."])
        self.assertEqual((s["text"], s["command"], s["became"]), (False, False, "none"))
        s = score("scroll", "pointer", ["Scroll up."])
        self.assertEqual((s["command"], s["became"]), (False, "scroll_up"))

    def test_empty(self):
        self.assertEqual(score("stop", "scroll", []),
                         {"text": False, "command": False, "empty": True, "became": "empty"})

    def test_stop_and_done_leave_scroll_mode(self):
        self.assertTrue(score("stop", "scroll", ["Stop."])["command"])
        self.assertTrue(score("done", "scroll", ["Done!"])["command"])
        self.assertFalse(score("stop", "scroll", ["Top."])["command"])

    def test_targets(self):
        self.assertTrue(score("click send", "pointer", ["Click Send."])["command"])
        self.assertFalse(score("click send", "pointer", ["Click sent."])["command"])

    def test_two_utterances_for_one_clip_is_wrong(self):
        self.assertFalse(score("send it", "pointer", ["Send.", "It."])["command"])


class TestRun(unittest.TestCase):
    def test_counts_with_a_fake_whisper(self):
        # every phrase's clip "says" its own text, except "no", which comes back as "know"
        dictation = Dictation(model=FakeWhisperModel())
        rows = {pid: [{"id": pid, "mode": "scroll" if text in ("stop", "done") else "pointer",
                       "voice": "fake", "variant": "normal", "text": text}] for pid, text in PHRASES}

        def transcribe(row):
            text = "know" if row["text"] == "no" else row["text"]
            return [words_to_text(dictation.transcribe(text))]

        logged = []
        result = run(dictation, None, rows, transcribe=transcribe, log=logged.append)
        self.assertEqual(result["total"], {"clips": 20, "text_right": 19, "command_right": 19,
                                           "command_wrong": 1, "empty": 0})
        self.assertEqual(result["per_phrase"]["command_092"]["command_wrong"], 1)
        self.assertEqual(len(logged), 1)
        self.assertIn("'no'", logged[0])

    def test_twenty_phrases_all_in_the_phrase_list(self):
        import csv
        from pathlib import Path

        with (Path(__file__).resolve().parent / "speech_phrases.csv").open(encoding="utf-8", newline="") as f:
            texts = {r["id"]: r["text"] for r in csv.DictReader(f)}
        self.assertEqual(len(PHRASES), 20)
        for pid, text in PHRASES:
            self.assertEqual(texts.get(pid), text, pid)
        self.assertEqual({t for _, t in PHRASES[:7]}, {"yes", "no", "click", "scroll", "stop", "send it", "open"})


if __name__ == "__main__":
    unittest.main()
