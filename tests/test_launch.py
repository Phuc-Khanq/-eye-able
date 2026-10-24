"""Opening apps and sites by voice: engine.voice.launch.match on what Whisper really writes, and
engine.eyes.os_output's os.launch handler with a fake launcher (nothing really starts).

Run with: python -m unittest tests.test_launch
"""
import unittest
from types import SimpleNamespace

from engine.config import Settings
from engine.core import messages, topics
from engine.eyes.os_output import OsOutput
from engine.voice import commands
from engine.voice.launch import PHRASES, match
from tests.helpers import FakeBus

S = Settings()          # the defaults (Khang's laptop), not this laptop's config.local.json


class TestApps(unittest.TestCase):
    def test_the_six_apps_and_three_verbs(self):
        cases = {
            "open chrome": "chrome", "open Chrome": "chrome", "launch Edge": "edge",
            "start notepad": "notepad", "open file explorer": "file explorer",
            "open Settings": "settings", "open mail": "mail", "start chrome": "chrome",
            "launch notepad": "notepad", "open the settings": "settings",
        }
        for text, app in cases.items():
            with self.subTest(text=text):
                self.assertEqual(match(text, S), {"app": app})

    def test_aliases(self):
        cases = {
            "open google chrome": "chrome", "Launch Microsoft Edge!": "edge", "open explorer": "file explorer",
            "open files": "file explorer", "open my files": "file explorer", "open email": "mail",
            "open Outlook": "mail",
        }
        for text, app in cases.items():
            with self.subTest(text=text):
                self.assertEqual(match(text, S), {"app": app})

    def test_whisper_variants(self):
        cases = {"Open Chrome.": "chrome", "um, open chrome please": "chrome", "Uh, start Notepad.": "notepad",
                 "Can you open settings?": "settings", "OPEN CHROME!": "chrome", "open chrome. ": "chrome"}
        for text, app in cases.items():
            with self.subTest(text=text):
                self.assertEqual(match(text, S), {"app": app})

    def test_not_on_the_list_stays_look_and_say(self):
        for text in ["open Spotify", "open the door", "open compose", "open the first result",
                     "start the timer", "launch rocket", "open chrome tabs", "open notepad plus plus"]:
            with self.subTest(text=text):
                self.assertIsNone(match(text, S))
                if text.startswith("open"):
                    self.assertEqual(commands.parse(text).name, "target")   # look-and-say still gets it

    def test_fixed_commands_are_never_stolen(self):
        for text in ["open menu", "Open menu.", "can you open the menu", "start dictation", "start typing",
                     "start listening", "start clicking", "start scrolling", "go back", "search"]:
            with self.subTest(text=text):
                self.assertIsNone(match(text, S))

    def test_whole_utterance_only(self):
        for text in ["I want to open Chrome tomorrow", "open the door when you get home",
                     "please open chrome and notepad", "open chrome now", "can we start notepad later",
                     "I'll go to youtube.com later", "let's search for cats", "open", "start", "", "   "]:
            with self.subTest(text=text):
                self.assertIsNone(match(text, S))

    def test_only_listed_names_even_with_other_settings(self):
        mine = SimpleNamespace(launch_apps={"paint": "mspaint.exe"}, launch_aliases={"ms paint": "paint"},
                               launch_search_url=S.launch_search_url)
        self.assertEqual(match("open ms paint", mine), {"app": "paint"})
        self.assertIsNone(match("open chrome", mine))


class TestSites(unittest.TestCase):
    def test_go_to(self):
        cases = {
            "go to youtube.com": "https://youtube.com", "go to YouTube dot com": "https://youtube.com",
            "Go to YouTube dot com.": "https://youtube.com", "go to youtube": "https://youtube.com",
            "go to wikipedia.org": "https://wikipedia.org", "go to news.bbc.co.uk": "https://news.bbc.co.uk",
            "go to YouTube.com.": "https://youtube.com", "um, go to gmail.com please": "https://gmail.com",
            "go to youtube.com/feed/trending": "https://youtube.com/feed/trending",
            "go to my-site.org": "https://my-site.org",
        }
        for text, url in cases.items():
            with self.subTest(text=text):
                self.assertEqual(match(text, S), {"url": url})

    def test_not_a_host(self):
        for text in ["go to the store", "go to", "go to -bad-.com", "go to bad-.com", "go to c++.com",
                     "go to https://evil.com", "go to http://evil.com", "go to file:///C:/",
                     "go to you tube dot com", "go to youtube..com"]:
            with self.subTest(text=text):
                self.assertIsNone(match(text, S))

    def test_only_https(self):
        for text in ["go to youtube.com", "go to youtube", "search for cats"]:
            self.assertTrue(match(text, S)["url"].startswith("https://"))


class TestSearch(unittest.TestCase):
    def test_search(self):
        cases = {
            "search for cats": "cats", "Search for cats.": "cats", "search cats": "cats",
            "search for how to bake bread": "how+to+bake+bread", "Search for C++?": "c%2B%2B",
            "search for fish & chips": "fish+chips", "um, search for weather please": "weather",
        }
        for text, q in cases.items():
            with self.subTest(text=text):
                self.assertEqual(match(text, S), {"url": f"https://www.google.com/search?q={q}"})

    def test_empty_search_and_non_https_search_url(self):
        self.assertIsNone(match("search for", S))
        plain = SimpleNamespace(launch_apps={}, launch_aliases={}, launch_search_url="http://x.com/?q={q}")
        self.assertIsNone(match("search for cats", plain))


class TestPhrasesAndSettings(unittest.TestCase):
    def test_every_example_phrase_matches(self):
        for text in PHRASES:
            with self.subTest(text=text):
                self.assertIsNotNone(match(text, S))

    def test_aliases_point_at_listed_apps(self):
        self.assertEqual(set(S.launch_apps), {"chrome", "edge", "notepad", "file explorer", "settings", "mail"})
        for alias, app in S.launch_aliases.items():
            with self.subTest(alias=alias):
                self.assertIn(app, S.launch_apps)
        self.assertNotIn(S.launch_apps["chrome"], (None, ""))

    def test_each_settings_object_gets_its_own_dicts(self):
        a, b = Settings(), Settings()
        a.launch_apps["paint"] = "mspaint.exe"
        self.assertNotIn("paint", b.launch_apps)


class FakeLauncher:
    def __init__(self, fail: bool = False) -> None:
        self.started: list[tuple[str, str]] = []
        self.fail = fail

    def __call__(self, kind: str, target: str) -> None:
        if self.fail:
            raise OSError("The system cannot find the file specified")
        self.started.append((kind, target))


class TestOsOutputLaunch(unittest.TestCase):
    def setUp(self) -> None:
        self.bus = FakeBus()
        self.launcher = FakeLauncher()
        self.out = OsOutput(self.bus, os_api=SimpleNamespace(), settings=S, launcher=self.launcher)

    def tearDown(self) -> None:
        self.out.stop()

    def send(self, payload: dict) -> list[dict]:
        self.bus.publish(topics.OS_LAUNCH, payload)
        self.out.flush()
        return self.bus.ui(messages.STATUS)

    def test_app_on_the_list_starts_its_target(self):
        status = self.send({"app": "chrome"})
        self.assertEqual(self.launcher.started, [("app", "chrome.exe")])
        self.assertEqual([(s["level"], s["text"]) for s in status], [("info", "Opening Chrome")])

    def test_settings_and_mail_use_their_uris(self):
        self.send({"app": "settings"})
        self.send({"app": "mail"})
        self.assertEqual(self.launcher.started, [("app", "ms-settings:"), ("app", "mailto:")])

    def test_app_not_on_the_list_starts_nothing(self):
        for bad in ["cmd", "chrome.exe", "powershell", "", None, 5, ["chrome"]]:
            with self.subTest(app=bad):
                status = self.send({"app": bad})
                self.assertEqual(self.launcher.started, [])
                self.assertEqual(status[-1]["level"], "warn")

    def test_only_https_urls(self):
        for bad in ["http://x.com", "file:///C:/", "ms-settings:", "javascript:alert(1)", "HTTPS://x.com",
                    "https://x.com/a b", "", None]:
            with self.subTest(url=bad):
                status = self.send({"url": bad})
                self.assertEqual(self.launcher.started, [])
                self.assertEqual(status[-1]["level"], "warn")
        status = self.send({"url": "https://youtube.com"})
        self.assertEqual(self.launcher.started, [("url", "https://youtube.com")])
        self.assertEqual(status[-1], messages.make(messages.STATUS, level="info", text="Opening youtube.com"))

    def test_a_launcher_that_fails_is_a_status_error(self):
        self.out.launcher = FakeLauncher(fail=True)
        status = self.send({"app": "notepad"})
        self.assertEqual(status[-1]["level"], "error")
        self.assertIn("Couldn't open Notepad", status[-1]["text"])
        self.out.launcher = self.launcher                            # the worker still runs after a failure
        self.send({"app": "edge"})
        self.assertEqual(self.launcher.started, [("app", "msedge.exe")])

    def test_match_then_launch(self):
        """What the mode manager will do: match() the clause, publish the payload."""
        self.send(match("go to YouTube dot com", S))
        self.send(match("open google chrome", S))
        self.assertEqual(self.launcher.started, [("url", "https://youtube.com"), ("app", "chrome.exe")])


if __name__ == "__main__":
    unittest.main()
