"""Oct 24 night: more ways to answer (and "Know." as a no while a preview waits), "pause" as Whisper writes it, the
click flash and sound that replace the ring, and the Whisper hotwords setting.

Run with: python -m unittest tests.test_answers_and_click_fx
"""
import io
import tempfile
import unittest
import wave
from types import SimpleNamespace

from engine.core import messages, topics
from engine.core.mode_manager import Mode
from engine.eyes.click_feedback import (COLORS, FLASH_S, R_END, R_START, ClickFeedback, ClickSound, FlashModel,
                                        tick_wav)
from engine.voice import commands
from tests.helpers import FakeBus
from tests.test_mode_manager import Base


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class TestAnswerWords(Base):
    def dictate(self, text="see you at noon"):
        self.say("start dictation")
        self.say(text)
        self.assertTrue(self.mm.preview)

    def test_know_drops_a_waiting_preview_instead_of_joining_it(self):
        self.dictate()
        self.say("Know.")                       # Khang's live test: "no" kept coming out as "know"
        self.assertEqual(self.mm.preview, [])
        self.assertEqual(self.os(), [])

    def test_every_way_to_say_no_drops_the_preview(self):
        for no in ["no", "nope", "nah", "don't write it", "Don't type it.", "do not write it", "drop it", "scrap that",
                   "no thanks", "Don't.", "know"]:
            with self.subTest(no=no):
                self.setUp()
                self.dictate()
                self.say(no)
                self.assertEqual(self.mm.preview, [], no)
                self.assertEqual(self.os(), [], no)

    def test_every_way_to_say_yes_types_the_preview(self):
        for yes in ["yes", "yeah", "yep", "yup", "write it", "type it", "Write that.", "insert it", "keep it",
                    "yes please", "Ya."]:
            with self.subTest(yes=yes):
                self.setUp()
                self.dictate()
                self.say(yes)
                self.assertEqual(self.os(), [(topics.OS_TYPE, {"text": "see you at noon "})], yes)

    def test_with_nothing_waiting_they_are_just_words(self):
        self.say("start dictation")
        self.say("Know.")
        self.assertEqual(self.mm._preview_text(), "Know.")       # dictated, not an answer
        self.mm._discard_preview()
        self.say("don't")
        self.assertEqual(self.mm._preview_text(), "don't")

    def test_know_answers_a_question_in_pointer_mode_too(self):
        self.look()
        self.say("close window")
        self.assertIsNotNone(self.mm.question)
        self.say("Know.")
        self.assertIsNone(self.mm.question)
        self.assertEqual(self.os(), [])

    def test_type_it_no_longer_types_the_word_it(self):
        self.say("type it")
        self.assertEqual(self.mm.preview, [])
        self.say("type see you soon")                           # the one-shot still works
        self.assertEqual(self.mm._preview_text(), "see you soon")
        self.say("type it")                                     # and "type it" now answers it
        self.assertEqual(self.os(), [(topics.OS_TYPE, {"text": "see you soon "})])

    def test_answers_while_paused_do_nothing(self):
        self.dictate()
        self.say("pause")
        self.say("Know.")
        self.assertTrue(self.mm.preview)


class TestPauseAsWhisperWritesIt(Base):
    def test_pot_and_friends_pause(self):
        for said in ["Pot.", "pots", "paws", "paw", "Paused.", "pause it", "take a break", "Go to sleep.",
                     "pause listening"]:
            with self.subTest(said=said):
                self.setUp()
                self.say(said)
                self.assertEqual(self.mm.mode, Mode.PAUSED, said)

    def test_pot_pauses_from_dictation_instead_of_being_typed(self):
        self.say("start dictation")
        self.say("Pot.")
        self.assertEqual(self.mm.mode, Mode.PAUSED)
        self.assertEqual(self.mm.preview, [])

    def test_resume_ways(self):
        for said in ["resume", "resumed", "unpause", "continue", "wake up"]:
            with self.subTest(said=said):
                self.setUp()
                self.say("pause")
                self.say(said)
                self.assertEqual(self.mm.mode, Mode.POINTER, said)

    def test_near_words_do_not_pause(self):
        for said in ["passed", "post it", "note that", "pass"]:
            with self.subTest(said=said):
                cmd = commands.parse(said)
                self.assertFalse(cmd and cmd.name in ("pause", "no", "yes"), (said, cmd))


class TestFlash(unittest.TestCase):
    def test_a_flash_pops_and_fades(self):
        clock = Clock()
        f = FlashModel(clock)
        self.assertIsNone(f.frame())
        f.on_click(400, 300)
        x, y, r0, a0, color = f.frame()
        self.assertEqual((x, y, color), (400.0, 300.0, COLORS["left"]))
        self.assertAlmostEqual(r0, R_START)
        clock.t += FLASH_S / 2
        _, _, r1, a1, _ = f.frame()
        self.assertGreater(r1, r0)
        self.assertLess(a1, a0)
        self.assertLessEqual(r1, R_END)
        clock.t += FLASH_S
        self.assertIsNone(f.frame())

    def test_a_right_click_has_its_own_colour(self):
        f = FlashModel(Clock())
        f.on_click(10, 20, "right")
        self.assertEqual(f.frame()[4], COLORS["right"])

    def test_every_click_on_the_bus_flashes_and_ticks(self):
        bus, clock, played = FakeBus(), Clock(), []
        fx = ClickFeedback(bus, clock=clock, player=played.append)
        bus.publish(topics.OS_CLICK, {"x": 500, "y": 300})
        bus.publish(topics.OS_CLICK, {"x": 510, "y": 310, "button": "right"})
        self.assertEqual(played, ["left", "right"])
        self.assertEqual(fx.flash.frame()[:2], (510.0, 310.0))

    def test_switched_off(self):
        bus, played = FakeBus(), []
        fx = ClickFeedback(bus, flash=False, sound=False, player=played.append)
        bus.publish(topics.OS_CLICK, {"x": 500, "y": 300})
        self.assertEqual(played, [])
        self.assertIsNone(fx.flash)

    def test_a_blink_click_flashes_where_it_clicked(self):
        b = Base()
        b.setUp()
        fx = ClickFeedback(b.bus, player=lambda button: None)
        b.look(640, 360)
        b.gesture("blink")
        self.assertEqual(fx.flash.frame()[:2], (640.0, 360.0))


class TestSound(unittest.TestCase):
    def test_the_tick_is_a_short_quiet_wav(self):
        data = tick_wav(1900.0, 0.01, 0.5)
        with wave.open(io.BytesIO(data)) as w:
            self.assertEqual((w.getnchannels(), w.getsampwidth()), (1, 2))
            seconds = w.getnframes() / w.getframerate()
            frames = w.readframes(w.getnframes())
        self.assertLess(seconds, 0.1)
        import numpy as np
        peak = np.abs(np.frombuffer(frames, dtype="<i2")).max() / 32767
        self.assertLess(peak, 0.4)                                   # volume 0.5 -> at most 0.35 of full scale

    def test_prepare_writes_two_files_and_volume_zero_is_silent(self):
        with tempfile.TemporaryDirectory() as d:
            s = ClickSound(0.5, folder=d)
            s.prepare()
            self.assertEqual(sorted(s._paths), ["left", "right"])
            self.assertTrue(all(p.exists() for p in s._paths.values()))
            played = []
            quiet = ClickSound(0.0, player=played.append, folder=d)
            quiet.prepare()
            quiet.play("left")
            self.assertEqual(played, [])


class TestSettings(unittest.TestCase):
    def test_defaults(self):
        from engine.config import Settings
        s = Settings()
        self.assertEqual((s.pointer_mode, s.snap_gesture_clicks, s.click_flash, s.click_sound), ("head-eyes", False,
                                                                                                 True, True))
        self.assertEqual(s.whisper_hotwords, "")

    def test_hotwords_reach_whisper(self):
        from engine.voice.dictation import Dictation
        self.assertIsNone(Dictation(SimpleNamespace(whisper_hotwords="")).hotwords)
        self.assertEqual(Dictation(SimpleNamespace(whisper_hotwords=" pause yes no ")).hotwords, "pause yes no")
        self.assertEqual(Dictation(SimpleNamespace(whisper_hotwords="x"), hotwords="Khoa Nhi").hotwords, "Khoa Nhi")
        self.assertIsNone(Dictation().hotwords)

    def test_every_answer_on_the_card_is_a_real_answer(self):
        from engine.eyes.dictation_card import DROP_IT, NO_TO_QUESTION, TYPE_IT, YES_TO_QUESTION
        for phrase in TYPE_IT + YES_TO_QUESTION:
            self.assertEqual(commands.parse(phrase).name, "yes", phrase)
        for phrase in DROP_IT + NO_TO_QUESTION:
            self.assertIn(commands.parse(phrase).name, ("no", "cancel"), phrase)


if __name__ == "__main__":
    unittest.main()
