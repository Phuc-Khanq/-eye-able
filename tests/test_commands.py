"""Run with: python -m unittest discover -s tests -t ."""
import unittest

from engine.voice.commands import parse


class TestCommands(unittest.TestCase):
    def test_fixed(self):
        cases = {
            "Send it.": "send_it", "um, send it please": "send_it", "sent it": "send_it",
            "Scroll down!": "scroll_down", "scrol down": "scroll_down",
            "Scratch that.": "undo", "can you open the menu": "menu", "paws": "pause",
            "wake up": "resume", "Never mind": "cancel", "yeah": "yes", "no": "no",
            "Start dictation.": "start_dictation", "delete my voice data": "delete_voice_data",
        }
        for text, name in cases.items():
            with self.subTest(text=text):
                self.assertEqual(parse(text).name, name)

    def test_targets(self):
        cases = {
            "Click Send.": "Send", "Klick compose": "Compose",
            "click on the send button": "Send Button", "Open the first result": "First Result",
            "select Inbox": "Inbox",
        }
        for text, label in cases.items():
            with self.subTest(text=text):
                cmd = parse(text)
                self.assertEqual((cmd.name, cmd.label), ("target", label))

    def test_not_commands(self):
        for text in ["I'll send it tomorrow morning", "send it to my professor and cc Khoa",
                     "go", "hello there", "", "click", "click on the very big blue send button now"]:
            with self.subTest(text=text):
                self.assertIsNone(parse(text))

    def test_targets_off_in_dictation(self):
        self.assertIsNone(parse("Open the door when you get home", allow_targets=False))
        self.assertEqual(parse("send it", allow_targets=False).name, "send_it")

    def test_fixed_beats_target(self):
        self.assertEqual(parse("open menu").name, "menu")


if __name__ == "__main__":
    unittest.main()
