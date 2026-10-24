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
                     "go", "hello there", "", "click on the very big blue send button now"]:
            with self.subTest(text=text):
                self.assertIsNone(parse(text))

    def test_answers_to_click_question(self):
        # After "click Send" the engine asks "Click Send?"; people answer like this.
        # "click it" used to start a new search for a button named "It".
        # "Okay." / "OK" / "Sure." are no longer yes (part D review): said in passing they confirmed
        # "send" and "delete my voice data". See tests/test_command_fixes.py.
        for text in ["Yes.", "Click.", "Click it.", "Do it.", "Go ahead."]:
            with self.subTest(text=text):
                self.assertEqual(parse(text).name, "yes")
        self.assertEqual(parse("okay click send").label, "Send")   # a filler before a target still works

    def test_type_and_dictate_are_one_shot_dictation(self):
        self.assertEqual((parse("Type see you at noon.").name, parse("Type see you at noon.").label),
                         ("type_text", "see you at noon"))
        self.assertEqual(parse("dictate hello").name, "type_text")
        self.assertIsNone(parse("type"))
        self.assertEqual(parse("start typing").name, "start_dictation")   # fixed commands still win

    def test_only_click_counts_while_dictating(self):
        from engine.voice.commands import DICTATION_TARGET_VERBS
        self.assertEqual(parse("click send", target_verbs=DICTATION_TARGET_VERBS).label, "Send")
        self.assertIsNone(parse("open the door when you get home", target_verbs=DICTATION_TARGET_VERBS))

    def test_split_clauses(self):
        from engine.voice.commands import split_clauses
        def cut(text):
            return [(c.text, c.triggered) for c in split_clauses([{"text": w, "conf": 1.0} for w in text.split()])]
        self.assertEqual(cut("Click search box. Yeah."), [("Click search box.", False), ("Yeah.", False)])
        self.assertEqual(cut("yes yes click minus click minus"),
                         [("yes yes", False), ("click minus", False), ("click minus", False)])
        self.assertEqual(cut("Hey Venus, scroll down."), [("Hey", False), ("scroll down.", True)])
        self.assertEqual(cut("see you at noon computer stop dictation"),
                         [("see you at noon", False), ("stop dictation", True)])
        self.assertEqual(parse("Yes. Yes.").name, "yes")   # repeats collapse

    def test_targets_off_in_dictation(self):
        self.assertIsNone(parse("Open the door when you get home", allow_targets=False))
        self.assertEqual(parse("send it", allow_targets=False).name, "send_it")

    def test_fixed_beats_target(self):
        self.assertEqual(parse("open menu").name, "menu")


if __name__ == "__main__":
    unittest.main()
