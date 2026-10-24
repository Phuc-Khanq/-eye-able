"""Run with: python -m unittest tests.test_voice_data  (works in a temp folder only)"""
import tempfile
import unittest
from pathlib import Path

from engine.core import messages, topics
from engine.core.mode_manager import ModeManager
from engine.voice import voice_data
from engine.voice.dictation import Dictation, DictationWorker
from engine.voice.voice_data import VoiceData, delete_all
from tests.helpers import FakeBus, FakeWhisperModel


def make_voice_data(folder: Path) -> None:
    (folder / "commands" / "open_gmail").mkdir(parents=True)
    (folder / "commands" / "open_gmail" / "1.npy").write_bytes(b"x")
    (folder / "commands" / "open_gmail" / "2.npy").write_bytes(b"x")
    (folder / "vocabulary.txt").write_text("Khoa Nhi Gmail", encoding="utf-8")


class TestDeleteAll(unittest.TestCase):
    def test_removes_everything_inside_and_keeps_the_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "voice"
            make_voice_data(folder)
            outside = Path(tmp) / "keep.txt"
            outside.write_text("not voice data", encoding="utf-8")
            self.assertEqual(delete_all(folder), 3)
            self.assertTrue(folder.is_dir())
            self.assertEqual(list(folder.iterdir()), [])
            self.assertTrue(outside.exists())

    def test_nothing_saved_yet(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(delete_all(Path(tmp) / "missing"), 0)
            self.assertEqual(delete_all(Path(tmp)), 0)

    def test_refuses_the_repo_or_a_drive(self):
        for bad in (voice_data.ROOT, voice_data.ROOT.parent, Path(voice_data.ROOT.anchor)):
            with self.assertRaises(ValueError, msg=str(bad)):
                delete_all(bad)

    def test_default_folder_is_git_ignored_user_data(self):
        self.assertEqual(voice_data.VOICE_DIR, voice_data.ROOT / "user_data" / "voice")
        gitignore = (voice_data.ROOT / ".gitignore").read_text(encoding="utf-8").split()
        self.assertIn("user_data/", gitignore)


class TestSayDeleteMyVoiceData(unittest.TestCase):
    def test_spoken_command_then_yes_deletes_and_forgets_vocabulary(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "voice"
            make_voice_data(folder)
            bus = FakeBus()
            dictation = Dictation(model=FakeWhisperModel(), hotwords="Khoa Nhi Gmail")
            worker = DictationWorker(bus, dictation)
            ModeManager(bus)
            VoiceData(bus, dictation, folder)

            worker.process("delete my voice data")
            self.assertTrue((folder / "vocabulary.txt").exists())        # it asks first
            question = bus.ui(messages.CONFIRM)[-1]
            bus.publish(topics.from_ui(messages.CONFIRM_ANSWER),
                        {"type": "confirm_answer", "id": question["id"], "yes": True})

            self.assertEqual(list(folder.iterdir()), [])
            self.assertIsNone(dictation.hotwords)
            self.assertEqual(bus.ui(messages.STATUS)[-1],
                             {"type": "status", "level": "info", "text": "Your voice data is deleted."})

    def test_a_failure_is_reported(self):
        bus = FakeBus()
        VoiceData(bus, voice_dir=voice_data.ROOT)               # refused on purpose
        bus.publish(topics.DELETE_VOICE_DATA, {})
        self.assertEqual(bus.ui(messages.STATUS)[-1]["level"], "error")


if __name__ == "__main__":
    unittest.main()
