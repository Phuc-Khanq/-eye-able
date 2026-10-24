"""The command list can't fall behind the code: every command has a row, every shortcut has keys, and the
generated list says everything.

Run with: python -m unittest tests.test_command_list
"""
import unittest

from engine.voice import command_list, commands


class TestRegistry(unittest.TestCase):
    def test_every_command_has_a_row_and_every_row_a_command(self):
        names = set(commands.FIXED_COMMANDS.values())
        self.assertEqual(sorted(names - set(commands.COMMAND_INFO)), [], "add a row to COMMAND_INFO")
        self.assertEqual(sorted(set(commands.COMMAND_INFO) - names), [], "a row for a command nobody can say")

    def test_every_shortcut_has_keys_and_a_label(self):
        for name in commands.FIXED_COMMANDS.values():
            if name.startswith("key_"):
                self.assertIn(name, commands.KEY_SHORTCUTS, name)
                self.assertIn(name, commands.COMMAND_LABELS, name)

    def test_every_row_has_a_category_and_a_sentence(self):
        for name, (cat, text) in commands.COMMAND_INFO.items():
            self.assertTrue(cat.strip() and text.strip(), name)

    def test_no_phrase_is_two_commands(self):
        # a dict can't hold a duplicate key, but a typo like "new tab" twice would silently win the last one
        import re
        src = open(commands.__file__, encoding="utf-8").read()
        start = src.index("FIXED_COMMANDS: dict")
        block = src[start:src.index("\n}\n", start)]
        keys = re.findall(r'^    "([^"]+)":', block, flags=re.M)
        self.assertEqual([k for k in set(keys) if keys.count(k) > 1], [])


class TestRender(unittest.TestCase):
    def setUp(self):
        self.text = command_list.render()

    def test_every_phrase_is_in_the_list(self):
        for phrase in commands.FIXED_COMMANDS:
            self.assertIn(f'"{phrase}"', self.text, phrase)

    def test_has_the_sections(self):
        for head in ("## Dictation", "## Browser and tabs", "## Volume and screen", "## Apps and sites",
                     "## Gestures"):
            self.assertIn(head, self.text)

    def test_the_apps_and_sites_come_from_the_settings(self):
        self.assertIn("chrome", self.text)
        self.assertIn("youtube", self.text)

    def test_says_the_blink_clicks_and_the_brow_does_not(self):
        self.assertIn("Long blink", self.text)
        self.assertIn("Does not click", self.text)

    def test_tables_are_well_formed(self):
        for line in self.text.splitlines():
            if line.startswith("| ") and "---" not in line:
                self.assertEqual(line.count("|") - line.count("\\|"), 3, line)


if __name__ == "__main__":
    unittest.main()
