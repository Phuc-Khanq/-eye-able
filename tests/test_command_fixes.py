"""Run with: python -m unittest tests.test_command_fixes

Fixes from the part D review of the clause / trigger-word / one-shot commands. Each class is one
fix: the exact phrase that went wrong, and near-misses both ways so the feature still works.
"""
import unittest

from engine.core import messages, topics
from engine.core.mode_manager import ModeManager
from engine.core.types import GazePoint, TargetResult
from engine.voice.commands import parse
from engine.voice.dictation import Dictation, DictationWorker
from tests.helpers import FakeBus, FakeWhisperModel


class Base(unittest.TestCase):
    def setUp(self):
        self.bus = FakeBus()
        self.worker = DictationWorker(self.bus, Dictation(model=FakeWhisperModel()), settings=None)
        self.mm = ModeManager(self.bus)
        self.bus.publish(topics.GAZE, GazePoint(0, 500, 300, 0.9))

    def say(self, text):
        self.worker.process(text)

    def os(self):
        return [(t, p) for t, p in self.bus.log if t.startswith("os.")]

    def searches(self):
        return [r.label for r in self.bus.of(topics.TARGET_REQUEST)]

    def questions(self):
        return [(q["prompt"], q["state"]) for q in self.bus.ui(messages.CONFIRM)]

    def found(self):
        req = self.bus.of(topics.TARGET_REQUEST)[-1]
        self.bus.publish(topics.TARGET_RESULT, TargetResult(req.label, True, 410, 220))


class TestOkayAndSureAreNotYes(Base):
    """Fix 1: "okay" / "ok" / "sure" confirmed "Insert ... and send?" and "delete my voice data"."""

    def test_they_parse_as_nothing(self):
        for text in ["Okay.", "okay", "OK", "ok", "Sure.", "sure", "sure?", "okay okay"]:
            with self.subTest(text=text):
                self.assertIsNone(parse(text))

    def test_okay_does_not_send(self):
        self.say("start dictation")
        self.say("hi mom")
        self.say("send it")
        self.say("okay")
        self.assertEqual(self.os(), [])
        self.assertEqual(self.questions(), [('Insert "hi mom" and send?', "open")])

    def test_sure_does_not_delete_voice_data(self):
        self.say("delete my voice data")
        self.say("sure")
        self.assertEqual(self.bus.of(topics.DELETE_VOICE_DATA), [])
        self.assertEqual(self.questions()[-1][1], "open")

    def test_the_other_yes_words_still_work(self):
        for text in ["Yes.", "yeah", "Click.", "Click it.", "Do it.", "Go ahead."]:
            with self.subTest(text=text):
                self.assertEqual(parse(text).name, "yes")
        self.say("click send")
        self.found()
        self.say("click it")
        self.assertEqual(self.os(), [(topics.OS_CLICK, {"x": 410, "y": 220})])

    def test_okay_is_still_a_filler_before_a_command(self):
        self.assertEqual(parse("okay click send").label, "Send")
        self.assertEqual(parse("Okay, send it.").name, "send_it")


class TestClickWhileDictating(Base):
    """Fix 2: "I'll click the link later" searched UI-Venus for "Link Later" and lost the words."""

    def dictate(self, text):
        self.say("start dictation")
        self.say(text)

    def test_click_inside_a_sentence_stays_text(self):
        for text in ["I'll click the link later", "Remember to click save before you close it.",
                     "you just click it and wait"]:
            with self.subTest(text=text):
                self.setUp()
                self.dictate(text)
                self.assertEqual(self.searches(), [])
                self.assertEqual(self.mm._preview_text(), text)

    def test_a_long_label_stays_text_even_at_the_start(self):
        self.dictate("Click save before you close it.")
        self.assertEqual(self.searches(), [])
        self.assertEqual(self.mm._preview_text(), "Click save before you close it.")

    def test_click_as_its_own_sentence_still_clicks(self):
        cases = {"click send": ("Send", ""), "see you at noon. click send": ("Send", "see you at noon."),
                 "Okay, click send": ("Send", "Okay,"), "um click send": ("Send", ""),
                 "click on the send button": ("Send Button", ""), "Done for today! Click save.": ("Save", "Done for today!")}
        for text, (label, preview) in cases.items():
            with self.subTest(text=text):
                self.setUp()
                self.dictate(text)
                self.assertEqual(self.searches(), [label])
                self.assertEqual(self.mm._preview_text(), preview)

    def test_split_clauses_marks_sentence_starts(self):
        from engine.voice.commands import split_clauses

        def cut(text):
            return [(c.text, c.starts_sentence) for c in split_clauses([{"text": w, "conf": 1.0} for w in text.split()])]
        self.assertEqual(cut("I'll click the link later"), [("I'll", True), ("click the link later", False)])
        self.assertEqual(cut("see you at noon. click send"), [("see you at noon.", True), ("click send", True)])
        self.assertEqual(cut("um click send"), [("um click send", True)])   # fillers don't split


if __name__ == "__main__":
    unittest.main()
