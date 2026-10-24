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


class TestTriggerWordsOnlyBeforeACommand(Base):
    """Fix 3: "my computer is slow today" typed "my is slow today"."""

    def dictate(self, text):
        self.say("start dictation")
        self.say(text)

    def test_the_word_stays_when_no_command_follows(self):
        for text in ["my computer is slow today", "tell Venus I said hi", "I love my computer",
                     "Computer science is fun.", "my computer, the old one, broke"]:
            with self.subTest(text=text):
                self.setUp()
                self.dictate(text)
                self.assertEqual(self.mm._preview_text(), text)
                self.assertEqual(self.os(), [])

    def test_a_command_after_the_word_still_runs(self):
        self.dictate("hello team")
        self.say("Venus, scroll down")
        self.assertEqual(self.os(), [(topics.OS_SCROLL, {"direction": "down"})])
        self.assertEqual(self.mm._preview_text(), "hello team")

    def test_mid_sentence_trigger_still_works(self):
        # Khang's example, no punctuation: the trigger counts because a command follows
        self.dictate("see you at noon computer stop dictation")
        self.assertEqual(self.mm._preview_text(), "see you at noon")
        self.assertEqual(self.questions(), [('Insert "see you at noon" before stopping?', "open")])

    def test_split_clauses_puts_the_word_back(self):
        from engine.voice.commands import split_clauses

        def cut(text):
            return [(c.text, c.triggered) for c in split_clauses([{"text": w, "conf": 1.0} for w in text.split()])]
        self.assertEqual(cut("my computer is slow today"), [("my computer is slow today", False)])
        self.assertEqual(cut("I love my computer"), [("I love my computer", False)])
        self.assertEqual(cut("Hey Venus, scroll down."), [("Hey", False), ("scroll down.", True)])
        self.assertEqual(cut("see you at noon computer stop dictation"),
                         [("see you at noon", False), ("stop dictation", True)])


class TestPointerChatter(Base):
    """Fix 4: in Pointer mode "I said click it" was cut before "click" and answered yes."""

    def test_a_command_cut_from_mid_sentence_does_not_count(self):
        self.say("click send")
        self.found()                                         # "Click Send?" is open
        self.say("I said click it")
        self.assertEqual(self.os(), [])
        self.assertEqual(self.questions()[-1], ("Click Send?", "open"))
        self.say("you should click compose")
        self.assertEqual(self.searches(), ["Send"])          # no new search

    def test_commands_as_their_own_sentences_still_count(self):
        self.say("Click search box. Yeah.")                 # Khang's live test
        self.found()
        self.assertEqual(self.os(), [(topics.OS_CLICK, {"x": 410, "y": 220})])

    def test_a_trigger_word_still_makes_it_count(self):
        self.say("I said Venus, scroll down")
        self.assertEqual(self.os(), [(topics.OS_SCROLL, {"direction": "down"})])

    def test_fillers_before_a_command_are_fine(self):
        self.say("um, click send")
        self.say("please click compose")
        self.assertEqual(self.searches(), ["Send", "Compose"])


class TestTypeOf(Base):
    """Fix 5: "Type of food matters." made a one-shot preview "of food matters."."""

    def test_type_of_is_ordinary_speech(self):
        for text in ["Type of food matters.", "type of thing I like", "dictate of course"]:
            with self.subTest(text=text):
                self.assertIsNone(parse(text))
        self.say("Type of food matters.")
        self.assertEqual(self.mm.preview, [])

    def test_one_shot_typing_still_works(self):
        self.assertEqual(parse("Type see you at noon.").label, "see you at noon")
        self.say("type offer accepted")                       # "of..." as part of a word is fine
        self.assertEqual(self.mm._preview_text(), "offer accepted")


class TestOneWordReplies(Base):
    """Fix 6: dictating a one-word reply ("Yes.", "No.", "Okay.") gave "Nothing is waiting" instead."""

    def test_with_nothing_waiting_the_word_is_dictated(self):
        for text in ["Yes.", "No.", "Yeah!", "Nope.", "Okay.", "Sure."]:
            with self.subTest(text=text):
                self.setUp()
                self.say("start dictation")
                self.say(text)
                self.assertEqual(self.mm._preview_text(), text)
                self.assertEqual(self.bus.ui(messages.STATUS), [])
                self.say("yes")                               # now it IS waiting: yes types it
                self.assertEqual(self.os(), [(topics.OS_TYPE, {"text": text + " "})])

    def test_with_something_waiting_yes_and_no_still_answer(self):
        self.say("start dictation")
        self.say("hi mom")
        self.say("yes")
        self.assertEqual(self.os(), [(topics.OS_TYPE, {"text": "hi mom "})])
        self.say("see you")
        self.say("no")
        self.assertEqual(self.mm.preview, [])                 # discarded, not dictated
        self.say("click send")                                # a search is waiting
        self.say("yes")
        self.assertEqual(self.mm.preview, [])
        self.found()
        self.assertEqual(self.os()[-1], (topics.OS_CLICK, {"x": 410, "y": 220}))

    def test_pointer_mode_still_says_nothing_is_waiting(self):
        self.say("yes")
        self.assertIn("Nothing is waiting", self.bus.ui(messages.STATUS)[-1]["text"])


class TestKnowIsNoWhileAsked(Base):
    """Khang's live log: answering "Click Search Box?" with "no", Whisper wrote "Know."."""

    def test_know_answers_no(self):
        for text in ["Know.", "know", "Know!"]:
            with self.subTest(text=text):
                self.setUp()
                self.say("click search box")
                self.found()
                self.say(text)
                self.assertEqual(self.questions()[-1], ("Click Search Box?", "cancelled"))
                self.assertEqual(self.os(), [])

    def test_know_cancels_send_while_dictating_and_keeps_the_preview(self):
        self.say("start dictation")
        self.say("hi mom")
        self.say("send it")
        self.say("Know.")
        self.assertEqual(self.questions()[-1], ('Insert "hi mom" and send?', "cancelled"))
        self.assertEqual(self.os(), [])
        self.assertEqual(self.mm._preview_text(), "hi mom")

    def test_know_with_nothing_asked_is_just_a_word(self):
        self.say("start dictation")
        self.say("know")
        self.assertEqual(self.mm._preview_text(), "know")
        self.assertIsNone(parse("know"))

    def test_i_know_stays_text(self):
        self.say("start dictation")
        self.say("hi mom")
        self.say("send it")
        self.say("I know")
        self.assertEqual(self.questions()[-1][1], "open")
        self.setUp()
        self.say("start dictation")
        self.say("I know")
        self.assertEqual(self.mm._preview_text(), "I know")


if __name__ == "__main__":
    unittest.main()
