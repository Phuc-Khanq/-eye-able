"""The command flash: when a voice command runs, the card's pill shows its name ("Volume up") for a moment, in
every mode, without showing what the user said.

Run with: python -m unittest tests.test_command_flash
"""
import unittest
from types import SimpleNamespace

from engine.core import messages, topics
from engine.core.mode_manager import launch_label
from engine.eyes.dictation_card import FLASH_S, CardModel
from engine.eyes.os_output import OsOutput
from engine.eyes.system_controls import BrightnessUnsupported
from engine.voice import commands
from tests.helpers import FakeBus
from tests.test_mode_manager import Base


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def flash(label):
    return messages.make(messages.COMMAND, label=label)


class TestCardFlash(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.m = CardModel(self.clock)

    def test_a_command_shows_in_the_pill_then_goes(self):
        self.m.on_ui(flash("Volume up"))
        v = self.m.view()
        self.assertEqual((v.kind, v.pill, v.panel, v.listening), ("command", "Volume up", False, False))
        self.clock.t += FLASH_S + 0.1
        self.assertEqual(self.m.view().kind, "hidden")

    def test_a_newer_command_replaces_the_flash(self):
        self.m.on_ui(flash("Brightness up"))
        self.clock.t += 0.5
        self.m.on_ui(flash("Brightness 60%"))
        self.assertEqual(self.m.view().pill, "Brightness 60%")

    def test_questions_and_previews_win_over_a_flash(self):
        self.m.on_ui(flash("Copy"))
        self.m.on_ui(messages.make(messages.CONFIRM, id="q1", prompt="Close this window?", state="open",
                                   expires_in_s=12.0))
        self.assertEqual(self.m.view().kind, "confirm")

    def test_flashes_while_dictating_then_back_to_the_dictation_pill(self):
        self.m.on_ui(messages.make(messages.MODE, mode="dictation"))
        self.m.on_ui(flash("Volume down"))
        self.assertEqual(self.m.view().pill, "Volume down")
        self.clock.t += FLASH_S + 0.1
        self.assertEqual((self.m.view().kind, self.m.view().pill), ("mode", "Dictation"))

    def test_leaving_a_mode_says_so(self):
        for mode, text in (("dictation", "Dictation off"), ("scroll", "Scroll off"), ("paused", "Resumed")):
            with self.subTest(mode=mode):
                self.m.on_ui(messages.make(messages.MODE, mode=mode))
                self.m.on_ui(messages.make(messages.MODE, mode="pointer"))
                self.assertEqual(self.m.view().pill, text)
                self.clock.t += FLASH_S + 0.1

    def test_can_be_turned_off(self):
        m = CardModel(self.clock, command_flash=False)
        m.on_ui(flash("Volume up"))
        self.assertEqual(m.view().kind, "hidden")

    def test_an_empty_label_never_flashes(self):
        self.m.on_ui(flash("  "))
        self.assertEqual(self.m.view().kind, "hidden")


class TestModeManagerFlashes(Base):
    def labels(self):
        return [m["label"] for m in self.bus.ui(messages.COMMAND)]

    def test_shortcuts_and_system_commands_flash_their_name(self):
        self.say("new tab")
        self.say("volume up")
        self.say("scroll down")
        self.assertEqual(self.labels(), ["New tab", "Volume up", "Scroll down"])

    def test_opening_an_app_or_a_search_flashes(self):
        self.say("open Chrome")
        self.say("search for cats")
        self.assertEqual(self.labels(), ["Open Chrome", "Search: cats"])

    def test_chatter_never_flashes_but_a_spoken_click_does(self):
        self.say("I think the weather is nice")
        self.assertEqual(self.labels(), [])
        self.look()
        self.say("click")                       # no ring at the cursor any more: the pill says it was heard
        self.assertEqual(self.labels(), ["Click"])

    def test_undo_on_a_preview_does_not_flash(self):
        self.say("start dictation")
        self.say("wrong words")
        self.say("scratch that")
        self.assertEqual(self.labels(), [])

    def test_every_label_is_short(self):
        for name, label in commands.COMMAND_LABELS.items():
            self.assertLessEqual(len(label), 20, name)


class TestLaunchLabel(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(launch_label({"app": "file explorer"}), "Open File Explorer")
        self.assertEqual(launch_label({"url": "https://www.youtube.com"}), "Open youtube.com")
        self.assertEqual(launch_label({"url": "https://www.google.com/search?q=red+shoes"}), "Search: red shoes")


class FakeOs:
    def combo(self, k):
        pass


class TestBrightnessResultFlash(unittest.TestCase):
    def run_system(self, system):
        bus = FakeBus()
        out = OsOutput(bus, os_api=FakeOs(), settings=SimpleNamespace(), system=system)
        bus.publish(topics.OS_SYSTEM, {"action": "brightness_up"})
        out.flush()
        out.stop()
        return [m["label"] for m in bus.ui(messages.COMMAND)]

    def test_the_new_level_flashes(self):
        self.assertEqual(self.run_system(lambda a, o: "Brightness 60%"), ["Brightness 60%"])

    def test_an_unsupported_screen_flashes_a_short_warning(self):
        def nope(a, o):
            raise BrightnessUnsupported("external monitor")
        self.assertEqual(self.run_system(nope), ["Can't change brightness"])


if __name__ == "__main__":
    unittest.main()
