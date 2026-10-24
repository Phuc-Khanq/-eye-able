"""Run with: python -m unittest tests.test_voice_data"""
import tempfile
import unittest
from pathlib import Path

from engine.core import messages, topics
from engine.voice import voice_data
from engine.voice.finetune import record_pack
from tests.helpers import FakeBus


class TestVoiceData(unittest.TestCase):
    def test_delete_all_removes_personal_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            personal = Path(tmp) / "personal"
            personal.mkdir()
            (personal / "a.wav").write_bytes(b"x")
            (Path(tmp) / "manifests").mkdir()        # other data stays
            self.assertEqual(voice_data.delete_all(Path(tmp)), [personal])
            self.assertFalse(personal.exists())
            self.assertTrue((Path(tmp) / "manifests").exists())
            self.assertEqual(voice_data.delete_all(Path(tmp)), [])

    def test_default_folder_is_record_packs_folder(self):
        self.assertEqual(voice_data.PERSONAL_DIR,
                         record_pack.personal_dir(record_pack.common.DEFAULT_DATA_DIR))

    def test_delete_request_on_the_bus(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "personal").mkdir()
            bus = FakeBus()
            voice_data.VoiceData(bus, data_dir=Path(tmp))
            bus.publish(topics.DELETE_VOICE_DATA, {})
            self.assertFalse((Path(tmp) / "personal").exists())
            [msg] = bus.ui(messages.STATUS)
            self.assertEqual(msg["level"], "info")
            bus.publish(topics.DELETE_VOICE_DATA, {})
            self.assertIn("no voice recordings", bus.ui(messages.STATUS)[-1]["text"])


class TestRecordPackFiles(unittest.TestCase):
    def test_transcripts_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            table = Path(tmp) / "transcripts.csv"
            record_pack.append_row(table, "1.wav", "open my email")
            record_pack.append_row(table, "2.wav", "Hi, mom.")
            self.assertEqual(record_pack.done_texts(table), {"open my email", "Hi, mom."})
            record_pack.remove_row(table, "1.wav")
            self.assertEqual(record_pack.done_texts(table), {"Hi, mom."})

    def test_wav_is_16k_mono(self):
        import wave

        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.wav"
            record_pack.write_wav(p, np.zeros(16000, np.float32))
            with wave.open(str(p)) as w:
                self.assertEqual((w.getframerate(), w.getnchannels(), w.getnframes()), (16000, 1, 16000))

    def test_150_unique_phrases(self):
        self.assertEqual(len(record_pack.PHRASES), 150)
        self.assertEqual(len(set(record_pack.PHRASES)), 150)


if __name__ == "__main__":
    unittest.main()
