"""Run with: python -m unittest tests.test_refine

The whole-passage refine: the guard, the (?) marks, the mode manager's timing, the worker, the card's clipping.
No model: each test answers the refine request itself.
"""
import unittest
from types import SimpleNamespace

from engine.core import messages, topics
from engine.core.mode_manager import ModeManager
from engine.eyes.dictation_card import MAX_PREVIEW_LINES, clip_lines
from engine.voice.dictation import Dictation, DictationWorker
from engine.voice.rewrite import (RewriteWorker, check_refine, mark_unsure, unsure_words)
from engine.voice.rewrite import REFINE_SYSTEM_PROMPT
from tests.helpers import FakeBus, FakeWhisperModel
from tests.test_rewriter import HAS_TORCH, make as make_rewriter


class TestMarks(unittest.TestCase):
    def test_only_unsure_words_are_marked(self):
        words = [{"text": "go", "conf": 0.9}, {"text": "beech", "conf": 0.3}, {"text": "—", "conf": 0.1}]
        marked = mark_unsure(words)
        self.assertEqual(marked, "go (?)beech(?) —")
        self.assertEqual(unsure_words(marked), {"beech"})


class TestCheckRefine(unittest.TestCase):
    def test_a_good_refine_passes(self):
        self.assertIsNone(check_refine("so um we should go to the beech on saturday and and then get lunch",
                                       "We should go to the beach on Saturday, and then get lunch.", {"beech"}))

    def test_a_self_correction_may_drop_the_old_number(self):
        self.assertIsNone(check_refine("the meeting is at three no wait at four", "The meeting is at four."))

    def test_rejects(self):
        cases = {
            "negation": ("I am not coming to the party tonight at all", "I am coming to the party tonight."),
            "number": ("see you at four on thursday", "See you at five on Thursday."),
            "person": ("please pick me up at the station tonight", "Please pick you up at the station tonight."),
            "name": ("tell Khoa that the build is ready now", "Tell them that the build is ready now."),
            "marks": ("the plan is ready now for the team", "The (?)plan(?) is ready now for the team."),
            "empty": ("the plan is ready now", "   "),
            "invented": ("send it to the team", "Send it to the team and include all the three reports from last week."),
        }
        for what, (orig, new) in cases.items():
            with self.subTest(what=what):
                self.assertIsNotNone(check_refine(orig, new))

    def test_an_unsure_name_may_change(self):
        self.assertIsNone(check_refine("tell Koa that the build is ready now", "Tell Khoa that the build is ready now.",
                                       {"koa"}))


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


class TestRefineInTheModeManager(unittest.TestCase):
    TEXT = "um so I want to tell the the team that the meeting is moved to thursday"

    def setUp(self):
        self.bus, self.clock = FakeBus(), Clock()
        self.worker = DictationWorker(self.bus, Dictation(model=FakeWhisperModel()), settings=None)
        self.mm = ModeManager(self.bus, clock=self.clock, rewrite_wait_s=1.5, refine_after_s=3.0)

    def say(self, text):
        self.worker.process(text)

    def refines(self):
        return [r for r in self.bus.of(topics.REWRITE_REQUEST) if r.get("mode") == "refine"]

    def answer(self, clean, mode=None, used=True, req=None):
        req = req or (self.refines()[-1] if mode == "refine" else self.bus.of(topics.REWRITE_REQUEST)[-1])
        out = {"id": req["id"], "clean": clean, "used": used, "ms": 900.0}
        if mode:
            out["mode"] = mode
        self.bus.publish(topics.REWRITE_RESULT, out)

    def shown(self):
        p = self.bus.ui(messages.DICTATION_PREVIEW)[-1]
        return " ".join(w["text"] for w in p["words"])

    def typed(self):
        return [p["text"] for p in self.bus.of(topics.OS_TYPE)]

    def test_refine_starts_after_the_pause_and_replaces_the_quick_clean_up(self):
        self.say("start dictation")
        self.say(self.TEXT)
        self.answer("So I want to tell the team that the meeting is moved to Thursday.")
        self.assertEqual(self.refines(), [])
        self.clock.t = 2.0
        self.mm.tick()
        self.assertEqual(self.refines(), [])                       # not yet
        self.clock.t = 3.5
        self.mm.tick()
        [req] = self.refines()
        self.assertEqual(req["text"], self.TEXT)
        self.answer("I want to tell the team the meeting moved to Thursday.", mode="refine")
        self.assertEqual(self.shown(), "I want to tell the team the meeting moved to Thursday.")
        self.say("yes")
        self.assertEqual(self.typed(), ["I want to tell the team the meeting moved to Thursday. "])

    def test_short_passages_are_not_refined(self):
        self.say("start dictation")
        self.say("see you at noon")
        self.clock.t = 10
        self.mm.tick()
        self.assertEqual(self.refines(), [])

    def test_more_words_restart_the_wait_and_drop_a_stale_result(self):
        self.say("start dictation")
        self.say(self.TEXT)
        self.clock.t = 3.5
        self.mm.tick()
        first = self.refines()[-1]
        self.say("and the room changed too")                        # the passage grew: that refine is stale
        self.answer("STALE", mode="refine", req=first)
        self.assertNotIn("STALE", self.shown())
        self.assertEqual(len(self.refines()), 1)
        self.clock.t = 7.0
        self.mm.tick()
        self.assertEqual(len(self.refines()), 2)

    def test_yes_before_the_pause_refines_now_and_types_the_result(self):
        self.say("start dictation")
        self.say(self.TEXT)
        self.answer("So I want to tell the team that the meeting is moved to Thursday.")
        self.say("yes")
        self.assertEqual(self.typed(), [])                          # waiting for the refine
        self.assertEqual(len(self.refines()), 1)
        self.answer("I want to tell the team the meeting moved to Thursday.", mode="refine")
        self.assertEqual(self.typed(), ["I want to tell the team the meeting moved to Thursday. "])

    def test_yes_gives_up_waiting_after_refine_wait_s(self):
        self.say("start dictation")
        self.say(self.TEXT)
        self.answer("So I want to tell the team that the meeting is moved to Thursday.")
        self.say("yes")
        self.clock.t = 7.0
        self.mm.tick()
        self.assertEqual(self.typed(), ["So I want to tell the team that the meeting is moved to Thursday. "])

    def test_use_original_brings_back_the_users_words(self):
        self.say("start dictation")
        self.say(self.TEXT)
        self.clock.t = 3.5
        self.mm.tick()
        self.answer("I want to tell the team the meeting moved to Thursday.", mode="refine")
        self.say("use original")
        self.assertEqual(self.shown(), self.TEXT)

    def test_a_rejected_refine_keeps_the_quick_clean_up(self):
        self.say("start dictation")
        self.say(self.TEXT)
        self.answer("So I want to tell the team that the meeting is moved to Thursday.")
        self.clock.t = 3.5
        self.mm.tick()
        self.answer(self.TEXT, mode="refine", used=False)
        self.assertEqual(self.shown(), "So I want to tell the team that the meeting is moved to Thursday.")

    def test_off_by_default(self):
        mm = ModeManager(FakeBus(), clock=Clock(), rewrite_wait_s=1.5)
        self.assertIsNone(mm.refine_after_s)


class FakeRewriter:
    def rewrite(self, text):
        return text.upper(), True, 1.0

    def refine(self, marked):
        return "REFINED " + marked, True, 2.0


class TestWorker(unittest.TestCase):
    def test_a_refine_request_gets_a_refine_result(self):
        bus = FakeBus()
        w = RewriteWorker(bus, FakeRewriter(), log=lambda s: None)
        w.process({"id": 4, "mode": "refine", "text": "a b", "marked": "a (?)b(?)"})
        w.process({"id": 5, "text": "a b"})
        first, second = bus.of(topics.REWRITE_RESULT)
        self.assertEqual((first["id"], first["mode"], first["clean"]), (4, "refine", "REFINED a (?)b(?)"))
        self.assertNotIn("mode", second)

    def test_a_rewriter_without_refine_is_not_used(self):
        bus = FakeBus()
        w = RewriteWorker(bus, SimpleNamespace(rewrite=lambda t: (t, False, 0.0)), log=lambda s: None)
        w.process({"id": 1, "mode": "refine", "text": "a b"})
        self.assertFalse(bus.of(topics.REWRITE_RESULT)[0]["used"])


class TestCardClip(unittest.TestCase):
    def test_a_long_passage_shows_its_last_lines(self):
        lines = [[("w", False, 0)] * 3 for _ in range(MAX_PREVIEW_LINES + 4)]
        shown, hidden = clip_lines(lines)
        self.assertEqual((len(shown), hidden), (MAX_PREVIEW_LINES, 12))
        self.assertEqual(clip_lines(lines[:3]), (lines[:3], 0))


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestRefineFull(unittest.TestCase):
    MARKED = "we should go to the (?)beech(?) on saturday if the weather is nice and and then get lunch"

    def test_a_good_refine_is_used_and_gets_the_marked_text_and_a_long_budget(self):
        r, model, tok = make_rewriter("We should go to the beach on Saturday if the weather is nice, and then get lunch.")
        out = r.refine_full(self.MARKED)
        self.assertTrue(out.used, out.reason)
        self.assertEqual(model.calls[0]["max_time"], 15.0)
        self.assertGreater(model.calls[0]["max_new_tokens"], 32)
        messages = tok.templates[0][0]
        self.assertEqual(messages[0]["content"], REFINE_SYSTEM_PROMPT)
        self.assertEqual(messages[-1]["content"], self.MARKED)          # the marks reach the model
        self.assertNotIn("(?)", out.text)                                 # but `text` is the plain words

    def test_a_flipped_negation_is_not_used(self):
        r, _, _ = make_rewriter("I am coming to the party tonight at all.")
        out = r.refine_full("I am not coming to the party tonight at all")
        self.assertFalse(out.used)
        self.assertTrue(out.reason.startswith("negation"))

    def test_a_cut_off_refine_is_not_used(self):
        r, _, _ = make_rewriter("We should go to the beach", cut=True)
        self.assertFalse(r.refine_full(self.MARKED).used)


if __name__ == "__main__":
    unittest.main()
