"""Run with: python -m unittest tests.test_training  (no torch or datasets needed)"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from engine.voice import dictation
from training.torgo_utils import (Gate, is_dysarthric_speaker, is_usable_transcript, normalize,
                                  speaker_of, training_target, wer, word_errors)


class TestTorgoUtils(unittest.TestCase):
    def test_speaker_from_path(self):
        self.assertEqual(speaker_of("FC01_1_arrayMic_0066.wav"), "FC01")
        self.assertEqual(speaker_of(r"C:\data\M05\Session2\wav_headMic\M05_2_headMic_0012.wav"), "M05")
        self.assertEqual(speaker_of("F03-session1-0001.wav"), "F03")
        self.assertIsNone(speaker_of("random_0001.wav"))

    def test_dysarthric_codes(self):
        self.assertTrue(is_dysarthric_speaker("F04"))
        self.assertFalse(is_dysarthric_speaker("MC04"))

    def test_normalize(self):
        self.assertEqual(normalize("The 7 dogs, ran!"), "the seven dogs ran")
        self.assertEqual(normalize("21"), "twenty one")
        self.assertEqual(normalize("Don\u2019t"), "don't")

    def test_usable(self):
        for bad in ["[say Ah-P-Eee repeatedly]", "xxx", "", "input/images/pic01.jpg"]:
            self.assertFalse(is_usable_transcript(bad), bad)
        self.assertTrue(is_usable_transcript("yes"))

    def test_training_target_looks_like_whisper(self):
        self.assertEqual(training_target("THE QUICK BROWN FOX"), "The quick brown fox.")
        self.assertEqual(training_target("yes"), "Yes.")
        self.assertEqual(training_target("Is it raining?"), "Is it raining?")
        self.assertEqual(training_target("Call Grandpa"), "Call Grandpa.")

    def test_wer(self):
        self.assertEqual(word_errors("the cat sat", "the cat sat"), (0, 3))
        self.assertEqual(word_errors("the cat sat", "a cat"), (2, 3))
        self.assertEqual(word_errors("seven", "7."), (0, 1))
        self.assertAlmostEqual(wer([("a b", "a"), ("c d", "c d")]), 0.25)
        self.assertEqual(word_errors("hello", ""), (1, 1))

    def test_gate(self):
        g = Gate()
        self.assertTrue(g.decide(0.60, 0.40, 0.08, 0.085)[0])
        self.assertFalse(g.decide(0.60, 0.59, 0.08, 0.08)[0])     # too little gain
        self.assertFalse(g.decide(0.60, 0.40, 0.08, 0.12)[0])     # hurts typical speech


class TestResolveModel(unittest.TestCase):
    def test_names_pass_through(self):
        self.assertEqual(dictation.resolve_model("large-v3-turbo"), "large-v3-turbo")
        self.assertEqual(dictation.resolve_model("openai/whisper-large-v3-turbo"),
                         "openai/whisper-large-v3-turbo")

    def test_local_folder_from_repo_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "models" / "my-ct2").mkdir(parents=True)
            with mock.patch.object(dictation, "ROOT", Path(tmp)):
                self.assertEqual(dictation.resolve_model("models/my-ct2"),
                                 str(Path(tmp) / "models" / "my-ct2"))
                with self.assertRaises(FileNotFoundError):
                    dictation.resolve_model("models/missing-ct2")


if __name__ == "__main__":
    unittest.main()
