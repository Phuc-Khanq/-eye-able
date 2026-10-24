"""Run with: python -m unittest tests.test_voice_check  (scoring helpers only; no audio, no Whisper)"""
import csv
import unittest
from pathlib import Path

from engine.voice import commands
from engine.voice.voice_check import command_name, normalize_words, percentile, word_errors

PHRASES = Path(__file__).resolve().parent / "speech_phrases.csv"


class TestNormalize(unittest.TestCase):
    def test_case_punctuation_and_numbers(self):
        self.assertEqual(normalize_words("Hi mom, home at 7 tonight."), "hi mom home at seven tonight".split())
        self.assertEqual(normalize_words("Grandpa turns 80."), ["grandpa", "turns", "eighty"])
        self.assertEqual(normalize_words("21 Oak Street"), ["twenty", "one", "oak", "street"])

    def test_times_and_phone_numbers(self):
        self.assertEqual(normalize_words("Meet Nhi at 3:30?"), normalize_words("meet nhi at three thirty"))
        self.assertEqual(normalize_words("at 8:00"), normalize_words("at eight"))
        self.assertEqual(normalize_words("10 a.m."), normalize_words("ten am"))
        self.assertEqual(normalize_words("ends in 4729"), normalize_words("ends in 4 7 2 9"))

    def test_apostrophes(self):
        self.assertEqual(normalize_words("Doctor’s 8 o'clock"), ["doctor's", "eight", "o'clock"])


class TestWordErrors(unittest.TestCase):
    def test_counts(self):
        self.assertEqual(word_errors("the cat sat", "The cat sat."), (0, 3))
        self.assertEqual(word_errors("the cat sat", "a cat"), (2, 3))
        self.assertEqual(word_errors("seven", "7"), (0, 1))
        self.assertEqual(word_errors("hello", ""), (1, 1))

    def test_percentile(self):
        self.assertEqual(percentile([3, 1, 2], 0.5), 2)
        self.assertEqual(percentile([1, 2, 3, 4, 100], 0.95), 100)


class TestPhraseList(unittest.TestCase):
    def test_every_phrase_parses_to_its_expected_result_as_text(self):
        """If this fails, the phrase list or commands.py changed; the speech check would mislead."""
        with PHRASES.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertGreater(len(rows), 100)
        for r in rows:
            got = command_name(commands.parse(r["text"], allow_targets=r["mode"] == "pointer"))
            expected = "none" if r["expected"] == "dictation" else r["expected"]
            self.assertEqual(got, expected, r["id"])


if __name__ == "__main__":
    unittest.main()
