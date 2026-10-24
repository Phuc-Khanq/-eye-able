"""The on-screen dictation card (what it shows; no Tk) and the dictation fixes from the Oct 24 live run:
"Dictate how are you feeling right now. Yes." typed "Yes." into the text instead of answering, and
"Dictate." alone was "not a command".

Run with: python -m unittest tests.test_dictation_card
"""
import unittest

from engine.core import messages, topics
from engine.core.mode_manager import Mode
from engine.eyes.dictation_card import HEARD_S, CardModel, DictationCard, layout_words
from tests.helpers import FakeBus
from tests.test_mode_manager import Base


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def preview(*words, conf=0.95, **extra):
    return messages.make(messages.DICTATION_PREVIEW, words=[{"text": w, "conf": conf} for w in words], **extra)


class TestCardModel(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.m = CardModel(self.clock)

    def test_hidden_when_nothing_happens(self):
        self.assertEqual(self.m.view().kind, "hidden")

    def test_listening_shows_the_waveform_pill_only(self):
        self.m.on_ui(messages.make(messages.LISTENING, level=0.5, speaking=True))
        v = self.m.view()
        self.assertEqual((v.kind, v.pill, v.listening, v.panel), ("listening", "Listening", True, False))
        self.clock.t += 1.0
        self.assertEqual(self.m.view().kind, "hidden")

    def test_the_waveform_scrolls_with_the_mic_level(self):
        self.m.on_ui(messages.make(messages.LISTENING, level=0.4, speaking=True))
        self.m.tick_wave()
        self.assertEqual(self.m.levels[-1], 0.4)
        self.clock.t += 1.0
        self.m.tick_wave()
        self.assertEqual(self.m.levels[-1], 0.0)

    def test_heard_words_show_in_the_chip_for_a_moment(self):
        self.m.on_heard("new tab")
        v = self.m.view()
        self.assertEqual((v.kind, v.label, v.chip, v.panel), ("heard", "Heard", "new tab", True))
        self.clock.t += HEARD_S + 0.1
        self.assertEqual(self.m.view().kind, "hidden")

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
        self.assertIn("stop dictation", v.pill_hint)


class TestLayout(unittest.TestCase):
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
        bus.publish(topics.TRANSCRIPT, {"text": "new tab", "words": [], "ms": 1.0})
        self.assertEqual(card.view()[0].chip, "new tab")
        bus.publish(topics.TO_UI, preview("see", "you"))
        bus.publish(topics.OS_TYPE, {"text": "see you "})
        bus.publish(topics.TO_UI, messages.make(messages.DICTATION_PREVIEW, words=[]))
        self.assertEqual(card.view()[0].body, "Typed")


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
