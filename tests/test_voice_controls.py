"""Always-on voice commands: system controls (volume, brightness), keyboard shortcuts, and the ways out of
Dictation mode ("dictate off", or "close" / "off" / "turn off" said alone while dictating).

Run with: python -m unittest tests.test_voice_controls
"""
import subprocess
import unittest
from types import SimpleNamespace

from engine.core import messages, topics
from engine.core.mode_manager import Mode
from engine.eyes import system_controls as sc
from engine.eyes.os_output import OsOutput
from engine.voice import commands
from tests.helpers import FakeBus
from tests.test_mode_manager import Base


class TestParsing(unittest.TestCase):
    def test_every_shortcut_and_system_command_has_a_phrase_and_keys(self):
        names = set(commands.FIXED_COMMANDS.values())
        for name in commands.KEY_SHORTCUTS:
            self.assertIn(name, names)
        for name in commands.SYSTEM_COMMANDS:
            self.assertIn(name, names)

    def test_phrases(self):
        cases = {"volume up": "volume_up", "Volume down.": "volume_down", "louder please": "volume_up",
                 "mute": "mute", "brightness up": "brightness_up", "brightness down": "brightness_down",
                 "copy": "key_copy", "paste": "key_paste", "new tab": "key_new_tab", "next tab": "key_next_tab",
                 "close tab": "key_close_tab", "refresh": "key_refresh", "switch window": "key_switch_window",
                 "zoom in": "key_zoom_in", "go to the top": "key_top", "page down": "key_page_down",
                 "close window": "close_window", "dictate off": "stop_dictation",
                 "dictation off": "stop_dictation", "stop dictating": "stop_dictation",
                 "dictate on": "start_dictation"}
        for text, name in cases.items():
            with self.subTest(text=text):
                cmd = commands.parse(text)
                self.assertIsNotNone(cmd)
                self.assertEqual(cmd.name, name)

    def test_small_mishearings_still_work(self):
        self.assertEqual(commands.parse("Volume, dawn").name, "volume_down")
        self.assertEqual(commands.parse("brightness dawn").name, "brightness_down")

    def test_sentences_are_not_commands(self):
        for text in ("I copy the file tomorrow", "turn the volume up a bit later", "can we close the tab later"):
            with self.subTest(text=text):
                cmd = commands.parse(text, allow_targets=False)
                self.assertTrue(cmd is None or cmd.name == "type_text", cmd)

    def test_bare_exit_words_are_not_fixed_commands(self):
        for w in ("close", "off", "turn off", "exit"):
            self.assertIsNone(commands.parse(w, allow_targets=False))


class TestPointerMode(Base):
    def test_shortcuts_press_their_keys(self):
        self.say("new tab")
        self.say("copy")
        self.say("switch window")
        self.assertEqual(self.os(), [(topics.OS_KEY, {"key": "ctrl+t"}), (topics.OS_KEY, {"key": "ctrl+c"}),
                                     (topics.OS_KEY, {"key": "alt+tab"})])

    def test_system_commands_go_to_os_system(self):
        self.say("volume up")
        self.say("brightness down")
        self.assertEqual(self.os(), [(topics.OS_SYSTEM, {"action": "volume_up"}),
                                     (topics.OS_SYSTEM, {"action": "brightness_down"})])

    def test_close_window_asks_first(self):
        self.say("close window")
        self.assertEqual(self.os(), [])
        self.assertEqual(self.bus.ui(messages.CONFIRM)[-1]["prompt"], "Close this window?")
        self.say("yes")
        self.assertEqual(self.os(), [(topics.OS_KEY, {"key": "alt+f4"})])

    def test_close_window_no_does_nothing(self):
        self.say("close window")
        self.say("no")
        self.assertEqual(self.os(), [])

    def test_chatter_does_nothing(self):
        self.say("I think the volume is fine")
        self.say("close")                      # not dictating: just a word
        self.assertEqual(self.os(), [])
        self.assertEqual(self.mm.mode, Mode.POINTER)

    def test_paused_ignores_them(self):
        self.say("pause")
        self.say("volume up")
        self.assertEqual(self.os(), [])


class TestDictationOnOff(Base):
    def test_dictate_then_dictate_off(self):
        self.say("dictate")
        self.assertEqual(self.mm.mode, Mode.DICTATION)
        self.say("dictate off")
        self.assertEqual(self.mm.mode, Mode.POINTER)

    def test_dictate_on(self):
        self.say("dictate on")
        self.assertEqual(self.mm.mode, Mode.DICTATION)

    def test_bare_exit_words_end_dictation(self):
        for word in ("Close.", "Off.", "turn off", "Exit.", "Stop.", "Done.", "I'm done."):
            with self.subTest(word=word):
                self.say("start dictation")
                self.assertEqual(self.mm.mode, Mode.DICTATION)
                self.say(word)
                self.assertEqual(self.mm.mode, Mode.POINTER)

    def test_exit_with_words_waiting_asks_to_insert_them_first(self):
        self.say("start dictation")
        self.say("see you at noon")
        self.say("close")
        self.assertIn("before stopping", self.bus.ui(messages.CONFIRM)[-1]["prompt"])
        self.say("yes")
        self.assertEqual(self.mm.mode, Mode.POINTER)
        self.assertEqual([p["text"] for t, p in self.os() if t == topics.OS_TYPE], ["see you at noon "])

    def test_exit_words_inside_a_sentence_stay_text(self):
        self.say("start dictation")
        self.say("please close the door when you are done")
        self.assertEqual(self.mm.mode, Mode.DICTATION)
        self.assertIn("close the door", self.mm._preview_text())

    def test_whole_utterance_commands_work_while_dictating(self):
        self.say("start dictation")
        self.say("volume down")
        self.assertEqual(self.os(), [(topics.OS_SYSTEM, {"action": "volume_down"})])
        self.assertEqual(self.mm.mode, Mode.DICTATION)


class FakeOs:
    def __init__(self):
        self.keys = []

    def combo(self, k):
        self.keys.append(k)


def ps(stdout="", code=0):
    def run(cmd, **kw):
        run.cmd = cmd
        return SimpleNamespace(stdout=stdout, returncode=code)
    return run


class TestSystemControls(unittest.TestCase):
    def test_volume_presses_the_media_key_five_times(self):
        os_ = FakeOs()
        self.assertEqual(sc.volume("volume_up", os_.combo), "Volume up")
        self.assertEqual(os_.keys, ["media_volume_up"] * 5)

    def test_mute_toggles_once(self):
        os_ = FakeOs()
        sc.volume("mute", os_.combo)
        self.assertEqual(os_.keys, ["media_volume_mute"])

    def test_brightness_returns_the_new_level(self):
        run = ps("70\n")
        self.assertEqual(sc.brightness(10, run, platform="win32"), 70)
        self.assertEqual(run.cmd[:3], ["powershell", "-NoProfile", "-NonInteractive"])
        self.assertIn("+(10)", run.cmd[-1])
        self.assertIn("Max(10,", run.cmd[-1])           # never darker than BRIGHTNESS_MIN

    def test_an_external_monitor_says_unsupported(self):
        with self.assertRaises(sc.BrightnessUnsupported):
            sc.brightness(-10, ps("", code=1), platform="win32")

    def test_not_windows_says_unsupported(self):
        with self.assertRaises(sc.BrightnessUnsupported):
            sc.brightness(10, ps("50"), platform="linux")

    def test_powershell_timeout_says_unsupported(self):
        def slow(cmd, **kw):
            raise subprocess.TimeoutExpired(cmd, 10)
        with self.assertRaises(sc.BrightnessUnsupported):
            sc.brightness(10, slow, platform="win32")

    def test_run_action_text(self):
        self.assertEqual(sc.run_action("brightness_up", FakeOs(), ps("40"), "win32"), "Brightness 40%")
        with self.assertRaises(ValueError):
            sc.run_action("explode", FakeOs())


class TestOsOutputSystem(unittest.TestCase):
    def setUp(self):
        self.bus = FakeBus()

    def statuses(self):
        return [(p["level"], p["text"]) for p in self.bus.ui(messages.STATUS)]

    def test_os_system_runs_and_says_what_it_did(self):
        seen = []
        out = OsOutput(self.bus, os_api=FakeOs(), settings=SimpleNamespace(),
                       system=lambda a, os_: seen.append(a) or "Volume up")
        self.bus.publish(topics.OS_SYSTEM, {"action": "volume_up"})
        out.flush()
        out.stop()
        self.assertEqual(seen, ["volume_up"])
        self.assertEqual(self.statuses(), [("info", "Volume up")])

    def test_unsupported_brightness_is_a_warning(self):
        def nope(a, os_):
            raise sc.BrightnessUnsupported("an external monitor?")
        out = OsOutput(self.bus, os_api=FakeOs(), settings=SimpleNamespace(), system=nope)
        self.bus.publish(topics.OS_SYSTEM, {"action": "brightness_up"})
        out.flush()
        out.stop()
        self.assertEqual(self.statuses()[0][0], "warn")


if __name__ == "__main__":
    unittest.main()
