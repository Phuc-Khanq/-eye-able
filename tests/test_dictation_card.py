"""The on-screen dictation card (what it shows; no Tk) and the dictation fixes from the Oct 24 live run:
"Dictate how are you feeling right now. Yes." typed "Yes." into the text instead of answering, and
"Dictate." alone was "not a command".

Run with: python -m unittest tests.test_dictation_card
"""
import unittest

from engine.core import messages, topics
from engine.core.mode_manager import Mode
from engine.eyes.dictation_card import (DONE_S, HEARD_S, PROMPT_MAX_CHARS, CardModel, DictationCard,
                                        layout_words)
from tests.helpers import FakeBus
from tests.test_mode_manager import Base


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def preview(*words, conf=0.95, **extra):
    return messages.make(messages.DICTATION_PREVIEW, words=[{"text": w, "conf": conf} for w in words], **extra)


def confirm(qid, state, prompt="Send?"):
    return messages.make(messages.CONFIRM, id=qid, prompt=prompt, state=state,
                         expires_in_s=12.0 if state == "open" else 0)


class TestCardModel(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.m = CardModel(self.clock)

    def test_hidden_when_nothing_happens(self):
        self.assertEqual(self.m.view().kind, "hidden")

    def dictating(self):
        self.m.on_ui(messages.make(messages.MODE, mode="dictation"))

    def test_listening_shows_the_waveform_pill_only(self):
        self.dictating()
        self.m.on_ui(messages.make(messages.LISTENING, level=0.5, speaking=True))
        v = self.m.view()
        self.assertEqual((v.kind, v.pill, v.listening, v.panel), ("listening", "Listening", True, False))
        self.clock.t += 1.0
        self.assertEqual(self.m.view().kind, "mode")   # back to the Dictation pill

    def test_outside_dictation_what_the_mic_hears_never_shows(self):
        self.m.on_ui(messages.make(messages.LISTENING, level=0.5, speaking=True))
        self.m.on_heard("volume up")
        v = self.m.view()
        self.assertEqual((v.kind, v.listening), ("hidden", False))
        for mode in ("scroll", "paused", "menu"):
            with self.subTest(mode=mode):
                self.m.on_ui(messages.make(messages.MODE, mode=mode))
                v = self.m.view()
                self.assertNotIn(v.kind, ("heard", "listening"))
                self.assertFalse(v.listening)

    def test_saying_dictate_brings_the_card_and_what_it_hears(self):
        self.m.on_heard("hello there")
        self.assertEqual(self.m.view().kind, "hidden")
        self.dictating()
        self.m.on_heard("see you at noon")
        self.assertEqual((self.m.view().kind, self.m.view().chip), ("heard", "see you at noon"))
        self.m.on_ui(messages.make(messages.MODE, mode="pointer"))     # "dictate off"
        self.assertEqual(self.m.view().kind, "hidden")

    def test_the_old_behaviour_is_a_setting(self):
        m = CardModel(self.clock, heard_outside_dictation=True)
        m.on_heard("new tab")
        self.assertEqual(m.view().kind, "heard")

    def test_questions_and_a_one_shot_preview_still_show_outside_dictation(self):
        self.m.on_ui(preview("see", "you"))                  # "dictate see you" from Pointer mode
        self.assertEqual(self.m.view().kind, "preview")
        self.m.on_ui(confirm("q9", "open", "Close this window?"))
        self.assertEqual(self.m.view().kind, "confirm")

    def test_the_waveform_scrolls_with_the_mic_level(self):
        self.m.on_ui(messages.make(messages.LISTENING, level=0.4, speaking=True))
        self.m.tick_wave()
        self.assertEqual(self.m.levels[-1], 0.4)
        self.clock.t += 1.0
        self.m.tick_wave()
        self.assertEqual(self.m.levels[-1], 0.0)

    def test_heard_words_show_in_the_chip_for_a_moment(self):
        self.dictating()
        self.m.on_heard("new tab")
        v = self.m.view()
        self.assertEqual((v.kind, v.label, v.chip, v.panel), ("heard", "Heard", "new tab", True))
        self.clock.t += HEARD_S + 0.1
        self.assertEqual(self.m.view().kind, "mode")

    def test_preview_stays_with_actions_and_key_hint_and_marks_shaky_words(self):
        self.m.on_heard("Dictate meet at nine")
        self.m.on_ui(messages.make(messages.DICTATION_PREVIEW, words=[
            {"text": "meet", "conf": 0.9}, {"text": "at", "conf": 0.9}, {"text": "nine", "conf": 0.3}]))
        self.clock.t += 20
        v = self.m.view()
        self.assertEqual(v.kind, "preview")
        self.assertEqual(v.words, [("meet", False), ("at", False), ("nine", True)])
        self.assertEqual(v.chip, "Dictate meet at nine")
        self.assertEqual(v.actions, ["type", "drop"])
        self.assertTrue(v.keys)

    def test_preview_beats_listening_but_the_pill_still_shows_the_waveform(self):
        self.dictating()
        self.m.on_ui(preview("hello"))
        self.m.on_ui(messages.make(messages.LISTENING, level=0.5, speaking=True))
        v = self.m.view()
        self.assertEqual(v.kind, "preview")
        self.assertTrue(v.listening)

    def test_cleaned_preview_puts_your_own_words_in_the_chip(self):
        self.m.on_ui(preview("I'll", "be", "there.", cleaned=True, original="um I'll be there"))
        v = self.m.view()
        self.assertEqual(v.chip, "um I'll be there")
        self.assertIn("original", v.actions)

    def test_a_long_chip_is_shortened(self):
        self.m.on_heard("word " * 40)
        self.assertLessEqual(len(self.m.view().chip), 70)

    def test_typed_then_closed_says_typed(self):
        self.m.on_ui(preview("hello"))
        self.m.on_typed()
        self.m.on_ui(messages.make(messages.DICTATION_PREVIEW, words=[]))
        self.assertEqual(self.m.view().body, "Typed")

    def test_closed_without_typing_says_dropped(self):
        self.m.on_ui(preview("hello"))
        self.m.on_ui(messages.make(messages.DICTATION_PREVIEW, words=[]))
        self.assertEqual(self.m.view().body, "Dropped")

    def test_dictation_mode_shows_the_pill_with_how_to_stop(self):
        self.m.on_ui(messages.make(messages.MODE, mode="dictation"))
        v = self.m.view()
        self.assertEqual((v.kind, v.pill, v.panel), ("mode", "Dictation", False))
        self.assertIn("dictate off", v.pill_hint)


class TestConfirmQuestion(unittest.TestCase):
    """The engine's yes/no questions ("Insert ... and send?", "Click Send?") on the card."""

    def setUp(self):
        self.clock = Clock()
        self.m = CardModel(self.clock)

    def test_an_open_question_shows_its_prompt_and_the_hint(self):
        self.m.on_ui(confirm("q1", "open", 'Insert "see you at noon" and send?'))
        v = self.m.view()
        self.assertEqual((v.kind, v.label, v.body, v.panel), ("confirm", "Question",
                                                              'Insert "see you at noon" and send?', True))
        self.assertTrue(v.keys)
        self.assertEqual((v.hint_yes, v.hint_no), ("", "to cancel"))
        self.assertEqual((v.time_left, v.time_total), (12, 12))
        self.clock.t += 4.5
        self.assertEqual(self.m.view().time_left, 8)

    def test_the_preview_keeps_its_hint_wording(self):
        self.m.on_ui(preview("hello"))
        v = self.m.view()
        self.assertEqual((v.hint_yes, v.hint_no, v.time_total), ("to type", "to drop it", 0))

    def test_confirmed_flashes_done_then_hides(self):
        self.m.on_ui(confirm("q1", "open"))
        self.m.on_ui(confirm("q1", "confirmed"))
        v = self.m.view()
        self.assertEqual((v.kind, v.body), ("done", "Done"))
        self.clock.t += DONE_S + 0.1
        self.assertEqual(self.m.view().kind, "hidden")

    def test_cancelled_and_expired_flash_cancelled(self):
        for state in ("cancelled", "expired"):
            self.m.on_ui(confirm("q1", "open"))
            self.m.on_ui(confirm("q1", state))
            self.assertEqual(self.m.view().body, "Cancelled", state)
            self.clock.t += DONE_S + 0.1

    def test_a_newer_question_replaces_the_old_one(self):
        self.m.on_ui(confirm("q1", "open", "Send?"))
        self.m.on_ui(confirm("q1", "cancelled", "Send?"))     # the engine closes the old one first
        self.m.on_ui(confirm("q2", "open", "Click Send?"))
        v = self.m.view()
        self.assertEqual((v.kind, v.body), ("confirm", "Click Send?"))

    def test_a_newer_question_wins_even_without_the_old_ones_end(self):
        self.m.on_ui(confirm("q1", "open", "Send?"))
        self.m.on_ui(confirm("q2", "open", "Click Send?"))
        self.assertEqual(self.m.view().body, "Click Send?")
        self.m.on_ui(confirm("q1", "cancelled", "Send?"))     # late end of the old one: q2 stays
        self.assertEqual(self.m.view().body, "Click Send?")

    def test_the_question_beats_the_preview_and_the_preview_comes_back(self):
        self.m.on_ui(preview("see", "you", "at", "noon"))
        self.m.on_ui(confirm("q1", "open", 'Insert "see you at noon" and send?'))
        self.assertEqual(self.m.view().kind, "confirm")
        self.m.on_ui(confirm("q1", "cancelled"))
        v = self.m.view()
        self.assertEqual(v.kind, "preview")
        self.assertEqual(v.words, [("see", False), ("you", False), ("at", False), ("noon", False)])

    def test_an_end_for_a_question_that_is_not_open_changes_nothing(self):
        self.m.on_ui(confirm("q7", "confirmed"))
        self.assertEqual(self.m.view().kind, "hidden")
        self.m.on_ui(confirm("q1", "open", "Send?"))
        self.m.on_ui(confirm("q2", "confirmed"))
        v = self.m.view()
        self.assertEqual((v.kind, v.body), ("confirm", "Send?"))

    def test_send_it_says_typed_once(self):
        # "send it" -> yes: the engine closes the question, types the preview, then closes the preview
        self.m.on_ui(preview("see", "you"))
        self.m.on_ui(confirm("q1", "open", 'Insert "see you" and send?'))
        self.m.on_ui(confirm("q1", "confirmed"))
        self.m.on_typed()
        self.m.on_ui(messages.make(messages.DICTATION_PREVIEW, words=[]))
        self.assertEqual(self.m.view().body, "Typed")

    def test_done_does_not_replace_a_typed_flash(self):
        self.m.on_ui(confirm("q1", "open"))
        self.m.on_typed()                                      # typed first, then the question's end
        self.m.on_ui(confirm("q1", "confirmed"))
        self.assertEqual(self.m.view().body, "Typed")

    def test_a_long_dictation_inside_the_prompt_is_shortened(self):
        long = "word " * 80
        self.m.on_ui(confirm("q1", "open", f'Insert "{long}" and send?'))
        body = self.m.view().body
        self.assertLessEqual(len(body), PROMPT_MAX_CHARS)
        self.assertTrue(body.endswith('" and send?'))
        self.assertIn("…", body)

    def test_a_question_whose_end_never_came_goes_away(self):
        self.m.on_ui(confirm("q1", "open"))
        self.clock.t += 12.0 + 3.0
        self.assertEqual(self.m.view().kind, "hidden")


class TestLayout(unittest.TestCase):
    class Mono:
        def measure(self, text):
            return 10 * len(text)

    def test_a_chip_that_is_too_long_is_cut_with_an_ellipsis_never_wrapped(self):
        from engine.eyes.dictation_card import fit
        self.assertEqual(fit("click", self.Mono(), 100), "click")
        cut = fit("um I'll be there at nine tomorrow", self.Mono(), 100)
        self.assertTrue(cut.endswith("\u2026"))
        self.assertLessEqual(self.Mono().measure(cut), 100)

    def test_the_heard_chip_fits_inside_its_panel(self):
        from engine.eyes import dictation_card as D

        class Canvas:
            def __init__(self):
                self.rects = []
            def delete(self, _): pass
            def find_all(self): return list(range(len(self.rects)))
            def tag_lower(self, *a): pass
            def create_polygon(self, pts, smooth=False, fill=None, outline=None):
                self.rects.append((min(pts[0::2]), max(pts[0::2]), fill)); return len(self.rects)
            def create_text(self, *a, **k): self.rects.append(None); return len(self.rects)
            def create_oval(self, *a, **k): self.rects.append(None); return len(self.rects)
            def create_rectangle(self, *a, **k): self.rects.append(None); return len(self.rects)
            def create_line(self, *a, **k): self.rects.append(None); return len(self.rects)
            def create_arc(self, *a, **k): self.rects.append(None); return len(self.rects)

        class F:
            def measure(self, t): return 8 * len(t)
            def metrics(self, _): return 20
        fonts = {k: F() for k in ("pill", "pill_hint", "label", "chip", "body", "shaky", "action", "hint", "key")}
        for chip in ("click", "new tab", "open chrome and go to the library website please now"):
            c = Canvas()
            D.draw(c, D.View("heard", pill="Diseyeble", label="Heard", chip=chip), (0.0,) * D.WAVE_BARS, fonts)
            boxes = [r for r in c.rects if r]
            panel = next(r for r in boxes if r[2] == D.PANEL)
            chip_box = next(r for r in boxes if r[2] == D.CHIP)
            self.assertGreaterEqual(chip_box[0], panel[0])
            self.assertLessEqual(chip_box[1], panel[1])

    def test_words_wrap_to_the_width(self):
        class Mono:
            def measure(self, text):
                return 10 * len(text)
        lines = layout_words([("aaaa", False), ("bbbb", True), ("cccc", False)], Mono(), Mono(), 95)
        self.assertEqual([[t for t, _, _ in line] for line in lines], [["aaaa", "bbbb"], ["cccc"]])
        self.assertEqual(lines[0][1][2], 50)   # after "aaaa" and a space


class TestCardOnTheBus(unittest.TestCase):
    def test_follows_the_engine_messages(self):
        bus, clock = FakeBus(), Clock()
        card = DictationCard(bus, clock=clock)
        bus.publish(topics.TO_UI, messages.make(messages.MODE, mode="dictation"))
        bus.publish(topics.TRANSCRIPT, {"text": "new tab", "words": [], "ms": 1.0})
        self.assertEqual(card.view()[0].chip, "new tab")
        bus.publish(topics.TO_UI, preview("see", "you"))
        bus.publish(topics.OS_TYPE, {"text": "see you "})
        bus.publish(topics.TO_UI, messages.make(messages.DICTATION_PREVIEW, words=[]))
        self.assertEqual(card.view()[0].body, "Typed")

    def test_shows_the_engines_question(self):
        bus, clock = FakeBus(), Clock()
        card = DictationCard(bus, clock=clock)
        bus.publish(topics.TO_UI, confirm("q1", "open", "Click Send?"))
        self.assertEqual(card.view()[0].kind, "confirm")
        bus.publish(topics.TO_UI, confirm("q1", "confirmed", "Click Send?"))
        self.assertEqual(card.view()[0].body, "Done")


class TestQuestionsFromTheModeManager(Base):
    """The real mode manager's questions, fed to the card: it shows each one and ends it."""

    def card(self):
        m = CardModel(Clock())
        for t, msg in self.bus.log:
            if t == topics.TO_UI:
                m.on_ui(msg)
            elif t == topics.OS_TYPE:
                m.on_typed()
        return m

    def test_send_it_shows_the_question(self):
        self.say("Dictate see you at noon")
        self.say("send it")
        v = self.card().view()
        self.assertEqual(v.kind, "confirm")
        self.assertIn("see you at noon", v.body)

    def test_send_it_then_yes_ends_with_typed(self):
        self.say("Dictate see you at noon")
        self.say("send it")
        self.say("yes")
        v = self.card().view()
        self.assertEqual((v.kind, v.body), ("done", "Typed"))

    def test_send_it_then_no_brings_the_preview_back(self):
        self.say("Dictate see you at noon")
        self.say("send it")
        self.say("no")
        self.assertEqual(self.card().view().kind, "preview")


class TestDictateFixes(Base):
    def test_dictate_then_yes_in_one_breath_types_without_the_yes(self):
        self.say("Dictate how are you feeling right now. Yes.")
        types = [p["text"] for t, p in self.os() if t == topics.OS_TYPE]
        self.assertEqual(len(types), 1)
        self.assertNotIn("Yes", types[0])
        self.assertIn("feeling right now", types[0])

    def test_dictate_then_no_drops_it(self):
        self.say("Dictate how are you feeling right now. No.")
        self.assertEqual([t for t, _ in self.os() if t == topics.OS_TYPE], [])
        self.assertFalse(self.mm.preview)

    def test_a_yes_inside_the_sentence_stays_text(self):
        self.say("Dictate I said yes to the offer")
        self.assertIn("yes", self.mm._preview_text())

    def test_dictate_alone_starts_dictation(self):
        self.say("Dictate.")
        self.assertEqual(self.mm.mode, Mode.DICTATION)


if __name__ == "__main__":
    unittest.main()
