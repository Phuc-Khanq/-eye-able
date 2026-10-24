"""Run with: python -m unittest tests.test_rewrite_mode_manager

The dictation clean-up in the mode manager. No model and no worker thread: the test answers each
dictation.rewrite_request itself, so every timing case is exact.
"""
import unittest

from engine.core import messages, topics
from engine.core.mode_manager import Mode, ModeManager
from engine.core.types import Gesture
from engine.voice.dictation import Dictation, DictationWorker
from tests.helpers import FakeBus, FakeWhisperModel
from tests.test_demo_flow import SCRIPT


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class Base(unittest.TestCase):
    WAIT = 1.5

    def setUp(self):
        self.bus, self.clock = FakeBus(), Clock()
        self.worker = DictationWorker(self.bus, Dictation(model=FakeWhisperModel()), settings=None)
        self.mm = ModeManager(self.bus, clock=self.clock, rewrite_wait_s=self.WAIT)

    def say(self, text):
        self.worker.process(text)

    def gesture(self, kind):
        self.bus.publish(topics.GESTURE, Gesture(t_ms=0, type=kind))

    def requests(self):
        return self.bus.of(topics.REWRITE_REQUEST)

    def answer(self, clean, used=True, request=None):
        req = request or self.requests()[-1]
        self.bus.publish(topics.REWRITE_RESULT, {"id": req["id"], "clean": clean if used else req["text"],
                                                 "used": used, "ms": 300.0})

    def preview(self):
        return self.bus.ui(messages.DICTATION_PREVIEW)[-1]

    def shown(self):
        return " ".join(w["text"] for w in self.preview()["words"])

    def typed(self):
        return [p["text"] for p in self.bus.of(topics.OS_TYPE)]

    def dictating(self, text):
        self.say("start dictation")
        self.say(text)


class TestCleanPreview(Base):
    def test_clean_text_is_shown_with_the_original_and_yes_types_it(self):
        self.dictating("um so I I think we should uh go")
        [req] = self.requests()
        self.assertEqual(req["text"], "um so I I think we should uh go")
        self.assertEqual(self.preview()["cleaned"], False)                  # raw until the result
        self.answer("I think we should go.")
        p = self.preview()
        self.assertEqual((self.shown(), p["original"], p["cleaned"]),
                         ("I think we should go.", "um so I I think we should uh go", True))
        self.say("yes")
        self.assertEqual(self.typed(), ["I think we should go. "])
        self.assertEqual(self.preview()["words"], [])                       # bubble hidden

    def test_a_rejected_rewrite_keeps_the_users_words(self):
        self.dictating("what is the weather")
        self.answer(None, used=False)
        self.assertEqual((self.shown(), self.preview()["cleaned"]), ("what is the weather", False))
        self.assertNotIn("original", self.preview())
        self.say("yes")
        self.assertEqual(self.typed(), ["what is the weather "])

    def test_every_change_asks_again_and_only_the_latest_answer_counts(self):
        self.dictating("hi mom")
        first = self.requests()[-1]
        self.say("home at seven")                                           # the preview grows
        second = self.requests()[-1]
        self.assertEqual(second["text"], "hi mom home at seven")
        self.assertGreater(second["id"], first["id"])
        self.answer("Hi mom.", request=first)                               # stale: for older text
        self.assertEqual((self.shown(), self.preview()["cleaned"]), ("hi mom home at seven", False))
        self.answer("Hi mom, home at seven.", request=second)
        self.assertEqual(self.shown(), "Hi mom, home at seven.")

    def test_clean_words_keep_whispers_doubt(self):
        self.dictating("meet at seven? tonight")                            # "seven" with 0.3 confidence
        self.answer("Meet at seven tonight, okay.")
        conf = {w["text"]: w["conf"] for w in self.preview()["words"]}
        self.assertEqual(conf["seven"], 0.3)
        self.assertEqual(conf["Meet"], 0.95)                                # case-insensitive match
        self.assertEqual(conf["okay."], 1.0)                                # written by the clean-up


class TestWaitingInsert(Base):
    def test_yes_while_cleaning_waits_for_the_result(self):
        self.dictating("um so I I think we should uh go")
        self.say("yes")
        self.assertEqual(self.typed(), [])                                  # nothing yet, nothing blocked
        self.answer("I think we should go.")
        self.assertEqual(self.typed(), ["I think we should go. "])

    def test_yes_while_cleaning_types_the_raw_text_after_the_wait(self):
        self.dictating("um so I I think we should uh go")
        self.say("yes")
        self.clock.t = self.WAIT - 0.01
        self.mm.tick()
        self.assertEqual(self.typed(), [])
        self.clock.t = self.WAIT + 0.01
        self.mm.tick()
        self.assertEqual(self.typed(), ["um so I I think we should uh go "])
        self.answer("I think we should go.")                                # too late: ignored
        self.assertEqual(len(self.typed()), 1)

    def test_a_discard_while_waiting_types_nothing(self):
        self.dictating("um so I I think we should uh go")
        self.say("yes")
        self.say("no")                                                      # changed their mind
        self.answer("I think we should go.")
        self.clock.t = 10
        self.mm.tick()
        self.assertEqual(self.typed(), [])

    def test_more_words_while_waiting_cancel_the_insert(self):
        self.dictating("hi mom")
        self.say("yes")
        self.say("home at seven")
        self.clock.t = 10
        self.mm.tick()
        self.assertEqual(self.typed(), [])
        self.assertEqual(self.requests()[-1]["text"], "hi mom home at seven")

    def test_leaving_dictation_while_waiting_types_nothing(self):
        self.dictating("hi mom")
        self.say("yes")
        self.bus.publish(topics.from_ui(messages.SET_MODE), {"type": "set_mode", "mode": "pointer"})
        self.answer("Hi mom.")
        self.assertEqual(self.typed(), [])

    def test_nod_brow_raise_and_the_ui_button_all_wait_the_same_way(self):
        for trigger in ("nod", "brow_raise", "ui"):
            with self.subTest(trigger=trigger):
                self.setUp()
                self.dictating("hi mom")
                if trigger == "ui":
                    self.bus.publish(topics.from_ui(messages.DICTATION), {"type": "dictation", "action": "insert"})
                else:
                    self.gesture(trigger)
                self.assertEqual(self.typed(), [])
                self.answer("Hi mom.")
                self.assertEqual(self.typed(), ["Hi mom. "])

    def test_send_it_presses_enter_only_after_the_text_is_typed(self):
        self.dictating("hi mom")
        self.say("send it")
        [q] = self.bus.ui(messages.CONFIRM)
        self.assertEqual(q["prompt"], 'Insert "hi mom" and send?')         # shows what is showing
        self.gesture("nod")                                                 # yes, while still cleaning
        self.assertEqual([t for t, _ in self.bus.log if t.startswith("os.")], [])
        self.answer("Hi mom.")
        self.assertEqual([(t, p) for t, p in self.bus.log if t.startswith("os.")],
                         [(topics.OS_TYPE, {"text": "Hi mom. "}), (topics.OS_KEY, {"key": "enter"})])

    def test_send_it_asks_with_the_clean_text_when_it_is_showing(self):
        self.dictating("hi mom")
        self.answer("Hi mom.")
        self.say("send it")
        self.assertEqual(self.bus.ui(messages.CONFIRM)[-1]["prompt"], 'Insert "Hi mom." and send?')

    def test_stop_dictation_leaves_the_mode_only_after_typing(self):
        self.dictating("hi mom")
        self.say("stop dictation")
        self.gesture("nod")                                                 # "Insert ... before stopping?"
        self.assertEqual(self.mm.mode, Mode.DICTATION)
        self.answer("Hi mom.")
        self.assertEqual(self.typed(), ["Hi mom. "])
        self.assertEqual(self.mm.mode, Mode.POINTER)


class TestSwitchingVersions(Base):
    def test_use_original_then_clean_it_up(self):
        self.dictating("um so I I think we should uh go")
        self.answer("I think we should go.")
        self.say("use original")
        self.assertEqual((self.shown(), self.preview()["cleaned"]), ("um so I I think we should uh go", False))
        self.say("clean it up")
        self.assertEqual((self.shown(), self.preview()["cleaned"]), ("I think we should go.", True))
        self.say("use original")
        self.say("yes")
        self.assertEqual(self.typed(), ["um so I I think we should uh go "])

    def test_the_choice_resets_for_the_next_preview(self):
        self.dictating("hi mom")
        self.answer("Hi mom.")
        self.say("use original")
        self.say("yes")
        self.say("see you soon")
        self.answer("See you soon.")
        self.assertEqual((self.shown(), self.preview()["cleaned"]), ("See you soon.", True))

    def test_use_original_while_cleaning_types_at_once(self):
        self.dictating("hi mom")
        self.say("use original")
        self.say("yes")                                                     # no need to wait
        self.assertEqual(self.typed(), ["hi mom "])

    def test_clean_it_up_with_nothing_to_show_says_why(self):
        self.dictating("hi mom")
        self.say("clean it up")
        self.assertEqual(self.bus.ui(messages.STATUS)[-1]["text"], "Still cleaning up...")
        self.answer(None, used=False)
        self.say("clean it up")
        self.assertEqual(self.bus.ui(messages.STATUS)[-1]["text"], "There's no cleaned-up version of this text")

    def test_the_commands_never_become_dictated_text(self):
        self.dictating("hi mom")
        self.say("use original")
        self.say("clean it up")
        self.assertEqual(self.requests()[-1]["text"], "hi mom")


class TestEditWord(Base):
    def test_editing_the_clean_text_adopts_it_and_never_cleans_again(self):
        self.dictating("meet nee at the cafe")
        self.answer("Meet nee at the cafe.")
        n = len(self.requests())
        self.bus.publish(topics.from_ui(messages.DICTATION),
                         {"type": "dictation", "action": "edit_word", "index": 1, "text": "Nhi"})
        self.assertEqual((self.shown(), self.preview()["cleaned"]), ("Meet Nhi at the cafe.", False))
        self.assertEqual(len(self.requests()), n)                           # a fixed word is final
        self.say("yes")
        self.assertEqual(self.typed(), ["Meet Nhi at the cafe. "])


class TestOneShotAndOff(Base):
    def test_one_shot_type_in_pointer_mode_is_cleaned_too(self):
        self.say("type um see you soon")
        self.answer("See you soon.")
        self.gesture("brow_raise")
        self.assertEqual(self.typed(), ["See you soon. "])

    def test_off_is_exactly_the_old_behavior(self):
        bus = FakeBus()
        worker = DictationWorker(bus, Dictation(model=FakeWhisperModel()), settings=None)
        ModeManager(bus)                                                    # no rewrite_wait_s: off
        for kind, value in SCRIPT:
            if kind == "say":
                worker.process(value)
            elif kind == "gesture":
                bus.publish(topics.GESTURE, Gesture(0, value))
        self.assertEqual(bus.of(topics.REWRITE_REQUEST), [])
        for p in bus.ui(messages.DICTATION_PREVIEW):
            self.assertEqual(set(p), {"type", "words"})                     # no new fields


if __name__ == "__main__":
    unittest.main()
