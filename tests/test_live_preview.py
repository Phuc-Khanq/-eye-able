"""Run with: python -m unittest tests.test_live_preview

The live preview: while the user is still speaking, the card shows what Whisper hears so far (grey), and the
finished words replace it. No mic, no model: FakeWhisperModel says the "audio" text back.
"""
import unittest
from types import SimpleNamespace

import numpy as np

from engine.core import messages, topics
from engine.eyes.dictation_card import LIVE_STALE_S, CardModel
from engine.voice import audio_in
from engine.voice.audio_in import AudioIn
from engine.voice.dictation import Dictation, DictationWorker
from engine.voice.mic import MicSettings
from tests.helpers import FakeBus, FakeWhisperModel
from tests.test_audio_in import FakeVad, silence, tone

RATE = audio_in.RATE
LIVE = SimpleNamespace(live_preview=True, live_every_s=1.0, live_min_s=0.8, pause_seconds=1.0)
LOG = []
SPAN = {"start_ms": 1000.0, "end_ms": 3000.0}


def worker(mouth=None, mode="dictation", settings=None):
    bus = FakeBus()
    s = settings or SimpleNamespace(live_preview=True, mouth_gate=True, mouth_gate_mode="utterance",
                                    mouth_min_moving=0.3, mouth_pad_ms=150.0, mouth_status_every_s=10.0)
    w = DictationWorker(bus, Dictation(model=FakeWhisperModel()), mouth=mouth, settings=s, log=LOG.append)
    w._mode = mode
    return w, bus


def partial(text, **extra):
    return {"audio": text, **SPAN, **extra}


def live_texts(bus):
    return [m["text"] for m in bus.ui(messages.DICTATION_LIVE)]


class FakeMouth:
    def __init__(self, f):
        self.f = f

    def moving_fraction(self, a, b):
        return self.f


class TestAudioInPartials(unittest.TestCase):
    def make(self, settings=LIVE):
        bus = FakeBus()
        a = AudioIn(bus, settings, mic_settings=MicSettings(-1, "synthetic", "file", RATE, 1),
                    vad=FakeVad(min_silence_s=0.5))
        return a, bus

    def feed(self, a, clip):
        for i in range(0, len(clip), 1536):
            a.process(clip[i:i + 1536])

    def test_partials_go_out_about_once_a_second_while_speaking(self):
        a, bus = self.make()
        self.feed(a, np.concatenate([tone(3.5), silence(1.0)]))
        parts = bus.of(topics.PARTIAL_AUDIO)
        self.assertTrue(3 <= len(parts) <= 4, len(parts))
        sizes = [len(p["audio"]) for p in parts]
        self.assertEqual(sizes, sorted(sizes))                       # each holds the audio so far, so it grows
        self.assertGreaterEqual(sizes[0], 0.8 * RATE)
        self.assertEqual(len(bus.of(topics.UTTERANCE)), 1)           # the final utterance still comes
        self.assertTrue(all(p["end_ms"] > p["start_ms"] for p in parts))

    def test_a_short_clip_makes_no_partial(self):
        a, bus = self.make()
        self.feed(a, np.concatenate([tone(0.6), silence(1.0)]))
        self.assertEqual(bus.of(topics.PARTIAL_AUDIO), [])

    def test_off_by_default_and_by_setting(self):
        for settings in (SimpleNamespace(pause_seconds=1.0), SimpleNamespace(live_preview=False, pause_seconds=1.0)):
            a, bus = self.make(settings)
            self.feed(a, np.concatenate([tone(3.0), silence(1.0)]))
            self.assertEqual(bus.of(topics.PARTIAL_AUDIO), [])

    def test_a_pause_is_not_transcribed_again(self):
        a, bus = self.make(SimpleNamespace(live_preview=True, live_every_s=0.5, live_min_s=0.8, pause_seconds=1.0))
        self.feed(a, np.concatenate([tone(1.5), silence(3.0)]))     # the FakeVad waits 0.5 s of quiet, then ends
        self.assertLessEqual(len(bus.of(topics.PARTIAL_AUDIO)), 3)
        n = len(bus.of(topics.PARTIAL_AUDIO))
        a2, bus2 = self.make(SimpleNamespace(live_preview=True, live_every_s=0.5, live_min_s=0.8, pause_seconds=1.0))
        self.feed(a2, np.concatenate([tone(1.5), silence(0.3)]))
        self.assertEqual(len(bus2.of(topics.PARTIAL_AUDIO)), n)    # the quiet part added none

    def test_silence_makes_no_partial(self):
        a, bus = self.make()
        self.feed(a, silence(4.0))
        self.assertEqual(bus.of(topics.PARTIAL_AUDIO), [])

    def test_the_next_utterance_starts_its_own_count(self):
        a, bus = self.make()
        self.feed(a, np.concatenate([tone(2.0), silence(1.0), tone(2.0), silence(1.0)]))
        parts = bus.of(topics.PARTIAL_AUDIO)
        self.assertEqual(len(bus.of(topics.UTTERANCE)), 2)
        self.assertGreaterEqual(len(parts), 2)
        self.assertLess(len(parts[-1]["audio"]), 3.5 * RATE)          # not the two utterances joined


class TestWorkerLive(unittest.TestCase):
    def setUp(self):
        LOG.clear()

    def test_the_console_says_what_happened(self):
        w, bus = worker()
        w.on_partial(partial("hi mom"))
        w._live_step()
        self.assertTrue(any("[live]" in line and "hi mom" in line for line in LOG), LOG)
        w2, _ = worker(mode="pointer")
        w2.on_partial(partial("hi mom"))
        self.assertTrue(any("not showing" in line and "pointer" in line for line in LOG), LOG)
        w.on_partial(partial("late"))
        w.on_utterance("late")
        w.on_partial(partial("hi again"))
        w._queue.put_nowait(("x", None))
        w._live_step()
        self.assertTrue(any("too late" in line for line in LOG), LOG)

    def test_the_whisper_time_of_a_finished_utterance_is_logged(self):
        w, bus = worker()
        w.process("hi mom home", None)
        self.assertTrue(any(line.startswith("[whisper]") for line in LOG), LOG)

    def test_a_partial_shows_as_live_text(self):
        w, bus = worker()
        w.on_partial(partial("hi mom home at"))
        w._live_step()
        self.assertEqual(live_texts(bus), ["hi mom home at"])

    def test_only_the_newest_partial_is_transcribed(self):
        w, bus = worker()
        w.on_partial(partial("hi mom"))
        w.on_partial(partial("hi mom home at seven"))
        w._live_step()
        w._live_step()
        self.assertEqual(live_texts(bus), ["hi mom home at seven"])

    def test_the_same_text_is_not_sent_twice(self):
        w, bus = worker()
        for _ in range(3):
            w.on_partial(partial("hi mom"))
            w._live_step()
        self.assertEqual(live_texts(bus), ["hi mom"])

    def test_not_outside_dictation_mode(self):
        for mode in ("pointer", "scroll", "paused", "menu", ""):
            with self.subTest(mode=mode):
                w, bus = worker(mode=mode)
                w.on_partial(partial("volume up"))
                w._live_step()
                self.assertEqual(live_texts(bus), [])

    def test_off_with_the_setting(self):
        w, bus = worker(settings=SimpleNamespace(live_preview=False, mouth_gate=False))
        w.on_partial(partial("hi mom"))
        w._live_step()
        self.assertEqual(live_texts(bus), [])

    def test_the_finished_utterance_clears_the_live_text_after_the_transcript(self):
        w, bus = worker()
        w.on_partial(partial("hi mom home"))
        w._live_step()
        n = len(bus.log)
        w.process("hi mom home at seven", None)
        new = bus.log[n:]
        self.assertEqual(new[0][0], topics.TRANSCRIPT)                       # the words first ...
        self.assertEqual((new[-1][0], new[-1][1]["text"]), (topics.TO_UI, ""))   # ... then the live text clears: no gap

    def test_a_dropped_utterance_still_clears_it(self):
        w, bus = worker(mouth=FakeMouth(0.0))
        w._live_text = "someone"
        w.process("someone else", SPAN)                 # the mouth gate drops it
        self.assertEqual(live_texts(bus)[-1], "")

    def test_a_partial_overtaken_by_the_final_is_never_shown(self):
        w, bus = worker()
        w.on_partial(partial("hi mom"))
        w.on_utterance("hi mom home at seven")          # the finished audio arrived first
        w._live_step()
        self.assertEqual(live_texts(bus), [])

    def test_a_partial_taken_before_the_final_but_transcribed_after_is_dropped(self):
        w, bus = worker()
        w.on_partial(partial("hi mom"))
        real = w.dictation.transcribe

        def slow(audio):
            w.on_utterance("hi mom home at seven")      # the final lands while Whisper runs the partial
            return real(audio)

        w.dictation.transcribe = slow
        w._live_step()
        self.assertEqual(live_texts(bus), [])

    def test_leaving_dictation_clears_it_and_drops_waiting_audio(self):
        w, bus = worker()
        w.on_partial(partial("hi mom"))
        w._live_step()
        w.on_partial(partial("hi mom home"))
        w.on_ui(messages.make(messages.MODE, mode="pointer"))
        w._live_step()
        self.assertEqual(live_texts(bus), ["hi mom", ""])

    def test_the_mouth_gate_hides_other_people(self):
        w, bus = worker(mouth=FakeMouth(0.0))
        w.on_partial(partial("someone else talking"))
        w._live_step()
        self.assertEqual(live_texts(bus), [])
        w2, bus2 = worker(mouth=FakeMouth(0.9))
        w2.on_partial(partial("my own voice"))
        w2._live_step()
        self.assertEqual(live_texts(bus2), ["my own voice"])

    def test_a_broken_whisper_call_only_skips_the_live_text(self):
        w, bus = worker()

        def boom(audio):
            raise RuntimeError("CUDA out of memory")

        w.dictation.transcribe = boom
        w.on_partial(partial("hi mom"))
        w._live_step()                                  # must not raise
        self.assertEqual(bus.ui(messages.STATUS), [])

    def test_a_waiting_final_is_not_held_up_by_showing_a_partial(self):
        w, bus = worker()
        w.on_partial(partial("hi mom"))
        w._queue.put_nowait(("hi mom home", None))      # a finished utterance is already waiting
        w._live_step()
        self.assertEqual(live_texts(bus), [])


class TestCardLive(unittest.TestCase):
    def setUp(self):
        self.t = [100.0]
        self.m = CardModel(lambda: self.t[0])
        self.m.on_ui(messages.make(messages.MODE, mode="dictation"))

    def live(self, text):
        self.m.on_ui(messages.make(messages.DICTATION_LIVE, text=text))

    def test_live_words_show_greyed_under_hearing(self):
        self.live("hi mom home at")
        v = self.m.view()
        self.assertEqual((v.kind, v.label), ("live", "Hearing"))
        self.assertEqual(v.words, [(w, "live") for w in "hi mom home at".split()])
        self.assertTrue(v.panel)

    def test_clearing_hides_it(self):
        self.live("hi mom")
        self.live("")
        self.assertEqual(self.m.view().kind, "mode")

    def test_with_a_preview_waiting_the_live_words_follow_it(self):
        self.m.on_ui(messages.make(messages.DICTATION_PREVIEW, words=[{"text": "Hello.", "conf": 0.9}]))
        self.live("and then")
        v = self.m.view()
        self.assertEqual(v.kind, "preview")
        self.assertEqual(v.words, [("Hello.", False), ("and", "live"), ("then", "live")])
        self.assertEqual(v.actions[:2], ["type", "drop"])

    def test_a_question_beats_it(self):
        self.live("send it")
        self.m.on_ui(messages.make(messages.CONFIRM, id="q1", prompt="Send?", state="open", expires_in_s=12.0))
        self.assertEqual(self.m.view().kind, "confirm")

    def test_stale_live_text_goes_away(self):
        self.live("hi mom")
        self.t[0] += LIVE_STALE_S + 0.1
        self.assertEqual(self.m.view().kind, "mode")

    def test_leaving_dictation_hides_it(self):
        self.live("hi mom")
        self.m.on_ui(messages.make(messages.MODE, mode="pointer"))
        self.assertNotEqual(self.m.view().kind, "live")
        self.m.on_ui(messages.make(messages.MODE, mode="dictation"))
        self.assertEqual(self.m.view().kind, "command")   # just the "Dictation off" flash, not back from the dead

    def test_it_beats_the_typed_flash(self):
        self.m.on_typed()
        self.live("next sentence")
        self.assertEqual(self.m.view().kind, "live")

    def test_not_shown_outside_dictation(self):
        self.m = CardModel(lambda: self.t[0])            # never in Dictation
        self.live("volume up")
        self.assertEqual(self.m.view().kind, "hidden")


if __name__ == "__main__":
    unittest.main()
