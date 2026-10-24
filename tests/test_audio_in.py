"""Run with: python -m unittest tests.test_audio_in  (no mic, torch or silero needed)

"Speech" here is a tone burst and the VAD is EnergyVad, so the cutting logic is tested
exactly, without depending on what Silero thinks of a sine wave.
"""
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from engine.core import messages, topics
from engine.voice import audio_in
from engine.voice.audio_in import (PRE_ROLL_S, TAIL_S, TARGET_RMS, ArraySource, AudioIn, EnergyVad,
                                   auto_gain, rms)
from tests.helpers import FakeBus

RATE = 16000


def tone(seconds, amp=0.3, rate=RATE):
    t = np.arange(int(seconds * rate)) / rate
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def quiet(seconds, rate=RATE):
    return (np.random.default_rng(0).standard_normal(int(seconds * rate)) * 0.001).astype(np.float32)


def run(*parts, rate=RATE, pause=1.5, mic_gain=1.0, flush=True):
    bus = FakeBus()
    signal = np.concatenate(parts)
    src = ArraySource(signal, rate)
    ai = AudioIn(bus, SimpleNamespace(pause_seconds=pause, mic_gain=mic_gain), source=src, vad=EnergyVad())
    for block in src.blocks():
        ai.feed(block)
    if flush:
        ai.flush()
    return bus, ai


class TestCutting(unittest.TestCase):
    def test_one_utterance_with_pre_roll_and_tail(self):
        bus, _ = run(quiet(0.5), tone(1.0), quiet(2.0))
        (utt,) = bus.of(topics.UTTERANCE)
        self.assertEqual(utt.dtype, np.float32)
        self.assertAlmostEqual(len(utt) / RATE, PRE_ROLL_S + 1.0 + TAIL_S, delta=0.1)

    def test_a_short_gap_stays_one_utterance(self):
        bus, _ = run(tone(0.6), quiet(0.8), tone(0.6), quiet(2.0), pause=1.0)
        self.assertEqual(len(bus.of(topics.UTTERANCE)), 1)

    def test_a_long_gap_splits(self):
        bus, _ = run(tone(0.6), quiet(1.5), tone(0.6), quiet(2.0), pause=1.0)
        self.assertEqual(len(bus.of(topics.UTTERANCE)), 2)

    def test_longer_pause_setting_keeps_slow_speech_together(self):
        slow = (tone(0.6), quiet(2.5), tone(0.6), quiet(4.5))
        self.assertEqual(len(run(*slow, pause=1.5)[0].of(topics.UTTERANCE)), 2)
        self.assertEqual(len(run(*slow, pause=3.0)[0].of(topics.UTTERANCE)), 1)

    def test_clicks_are_ignored(self):
        bus, _ = run(quiet(0.5), tone(0.1), quiet(2.0))
        self.assertEqual(bus.of(topics.UTTERANCE), [])

    def test_no_utterance_until_the_pause_ends(self):
        bus, _ = run(tone(1.0), quiet(0.5), flush=False)
        self.assertEqual(bus.of(topics.UTTERANCE), [])

    def test_flush_finishes_an_open_utterance(self):
        bus, _ = run(tone(1.0), quiet(0.5), flush=True)
        self.assertEqual(len(bus.of(topics.UTTERANCE)), 1)

    def test_very_long_speech_is_cut_for_whisper(self):
        bus, _ = run(tone(35.0), quiet(2.0))
        first = bus.of(topics.UTTERANCE)[0]
        self.assertLessEqual(len(first) / RATE, audio_in.MAX_UTTERANCE_S + 0.1)

    def test_48k_stereo_mic_becomes_16k(self):
        stereo = np.stack([np.concatenate([quiet(0.5, 48000), tone(1.0, rate=48000), quiet(2.0, 48000)])] * 2,
                          axis=1)
        bus, _ = run(stereo, rate=48000)
        (utt,) = bus.of(topics.UTTERANCE)
        self.assertAlmostEqual(len(utt) / RATE, PRE_ROLL_S + 1.0 + TAIL_S, delta=0.1)


class TestGainAndSettings(unittest.TestCase):
    def test_quiet_voice_is_heard_with_mic_gain_and_brought_up(self):
        whisper_quiet = (quiet(0.5), tone(1.0, amp=0.008), quiet(2.0))
        self.assertEqual(run(*whisper_quiet)[0].of(topics.UTTERANCE), [])      # too quiet to detect
        (utt,) = run(*whisper_quiet, mic_gain=3.0)[0].of(topics.UTTERANCE)
        self.assertGreater(rms(utt), TARGET_RMS * 0.5)

    def test_auto_gain_never_clips_and_leaves_loud_audio_alone(self):
        loud = tone(1.0, amp=0.5)
        np.testing.assert_array_equal(auto_gain(loud), loud)
        spiky = tone(1.0, amp=0.01)
        spiky[100] = 0.5
        self.assertLessEqual(np.max(np.abs(auto_gain(spiky))), 0.95 + 1e-6)
        self.assertEqual(rms(auto_gain(np.zeros(100, np.float32))), 0.0)

    def test_pause_from_the_settings_panel_is_clamped(self):
        bus, ai = run(quiet(0.1))
        for sent, expected in [(3, 3.0), (10, 4.0), (0.2, 1.0), ("x", 1.5)]:
            bus.publish(topics.from_ui(messages.SETTINGS), {"type": "settings", "pause_seconds": sent})
            self.assertEqual(ai.pause_seconds, expected)
        bus.publish(topics.from_ui(messages.SETTINGS), {"type": "settings", "dwell_ms": 900})
        self.assertEqual(ai.pause_seconds, 1.5)                                 # untouched

    def test_listening_messages_for_the_level_bar(self):
        bus, _ = run(quiet(0.5), tone(1.0), quiet(2.0))
        listening = bus.ui(messages.LISTENING)
        self.assertTrue(all(0.0 <= m["level"] <= 1.0 for m in listening))
        speaking = [m["speaking"] for m in listening]
        self.assertEqual(speaking[0], False)
        self.assertIn(True, speaking)
        self.assertEqual(speaking[-1], False)
        self.assertLess(len(listening), 3.5 / audio_in.LEVEL_EVERY_S + 10)        # about 10 per second


class TestStartAndFallbacks(unittest.TestCase):
    def test_threaded_start_with_an_array_instead_of_the_mic(self):
        bus, got = FakeBus(), threading.Event()
        bus.subscribe(topics.UTTERANCE, lambda a: got.set())
        src = ArraySource(np.concatenate([quiet(0.3), tone(0.8), quiet(1.8)]))
        ai = AudioIn(bus, SimpleNamespace(pause_seconds=1.5, mic_gain=1.0), source=src, vad=EnergyVad())
        self.assertTrue(ai.start())
        try:
            self.assertTrue(got.wait(5), "no utterance from the audio thread")
        finally:
            ai.stop()

    def test_no_mic_is_a_status_message_not_a_crash(self):
        class BrokenMic:
            rate = 48000

            def open(self, on_block):
                raise RuntimeError("No mic format worked")

        bus = FakeBus()
        ai = AudioIn(bus, SimpleNamespace(pause_seconds=1.5), source=BrokenMic(), vad=EnergyVad())
        self.assertFalse(ai.start())
        (status,) = bus.ui(messages.STATUS)
        self.assertEqual(status["level"], "error")
        self.assertIn("No microphone", status["text"])

    def test_without_torch_it_falls_back_to_loudness(self):
        warnings = []
        with mock.patch.object(audio_in, "SileroVad", side_effect=ImportError("no torch")):
            vad = audio_in.load_vad(warnings.append)
        self.assertIsInstance(vad, EnergyVad)
        self.assertIn("Silero VAD unavailable", warnings[0])


if __name__ == "__main__":
    unittest.main()
