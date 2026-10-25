"""Run with: python -m unittest tests.test_mouth_endpoint

A noisy room (Oct 24 night, Newell Hall): other people talked, so the VAD never heard a pause, clips ran to the 28 s cap
and the mouth gate then dropped "hi Dizzy" because the mouth moved in only 15% of the clip. Now a clip also ends when
the user's mouth has been still for pause_seconds. No mic, no camera: a loud tone stands in for the room.
"""
import unittest
from types import SimpleNamespace

import numpy as np

from engine.core import topics
from engine.voice import audio_in
from engine.voice.audio_in import AudioIn
from engine.voice.mic import MicSettings
from tests.helpers import FakeBus
from tests.test_audio_in import FakeVad, silence, tone

RATE = audio_in.RATE
T0 = 1_000_000.0          # ms, the clock of the fake camera and mic
SET = SimpleNamespace(mouth_endpoint=True, pause_seconds=1.0)


class FakeMouth:
    """moving_fraction = the share of the window inside the moving intervals (ms on the T0 clock); None when told."""

    def __init__(self, moving=(), none=False, boom=False):
        self.moving, self.none, self.boom = list(moving), none, boom
        self.calls = []

    def moving_fraction(self, a, b):
        self.calls.append((a, b))
        if self.boom:
            raise RuntimeError("camera thread died")
        if self.none:
            return None
        covered = sum(max(0.0, min(b, hi) - max(a, lo)) for lo, hi in self.moving)
        return covered / (b - a)


def make(mouth, settings=SET, vad_pause=60.0):
    bus = FakeBus()
    a = AudioIn(bus, settings, mic_settings=MicSettings(-1, "synthetic", "file", RATE, 1),
                vad=FakeVad(min_silence_s=vad_pause))      # a VAD that never hears a pause: a noisy room
    a.mouth = mouth
    return a, bus


def feed(a, clip, start_ms=T0):
    t = start_ms
    for i in range(0, len(clip), 1536):
        part = clip[i:i + 1536]
        t += len(part) * 1000.0 / RATE
        a.process(part, t)
    return t


def seconds(clip):
    return len(clip) / RATE


class TestMouthEndpoint(unittest.TestCase):
    def test_a_still_mouth_ends_the_clip_in_a_noisy_room(self):
        mouth = FakeMouth(moving=[(T0, T0 + 2000)])                 # talked for 2 s, then the room kept going
        a, bus = make(mouth)
        feed(a, tone(8.0))                                          # the room is loud for 8 s, no pause for the VAD
        [first, *_] = bus.of(topics.UTTERANCE)
        self.assertLess(seconds(first), 4.5)                        # about 2 s of speech + the 1 s pause, not 8 or 28
        self.assertGreater(seconds(first), 2.0)

    def test_without_it_the_clip_runs_on(self):
        for settings in (SimpleNamespace(pause_seconds=1.0), SimpleNamespace(mouth_endpoint=False, pause_seconds=1.0)):
            a, bus = make(FakeMouth(moving=[]), settings)
            feed(a, tone(8.0))
            self.assertEqual(bus.of(topics.UTTERANCE), [])

    def test_no_tracker_no_face_or_a_broken_tracker_leaves_it_to_the_vad(self):
        for mouth in (None, FakeMouth(none=True), FakeMouth(boom=True)):
            a, bus = make(mouth)
            feed(a, tone(8.0))
            self.assertEqual(bus.of(topics.UTTERANCE), [], mouth)

    def test_talking_mouth_does_not_end_it(self):
        a, bus = make(FakeMouth(moving=[(T0, T0 + 20_000)]))
        feed(a, tone(8.0))
        self.assertEqual(bus.of(topics.UTTERANCE), [])

    def test_a_short_gap_in_the_mouth_does_not_cut_a_sentence(self):
        a, bus = make(FakeMouth(moving=[(T0, T0 + 2000), (T0 + 2600, T0 + 6000)]))   # 0.6 s of stillness < 1.0 s pause
        feed(a, tone(6.0))
        self.assertEqual(bus.of(topics.UTTERANCE), [])

    def test_the_next_words_start_a_new_clip(self):
        mouth = FakeMouth(moving=[(T0, T0 + 2000), (T0 + 6000, T0 + 8000)])
        a, bus = make(mouth)
        feed(a, tone(10.0))
        self.assertGreaterEqual(len(bus.of(topics.UTTERANCE)), 2)

    def test_the_speech_span_matches_the_shortened_clip(self):
        a, bus = make(FakeMouth(moving=[(T0, T0 + 2000)]))
        feed(a, tone(8.0))
        [span, *_] = bus.of(topics.SPEECH_SPAN)
        [clip, *_] = bus.of(topics.UTTERANCE)
        self.assertAlmostEqual((span["end_ms"] - span["start_ms"]) / 1000.0, seconds(clip), delta=0.2)

    def test_a_quiet_room_still_ends_on_the_vad_as_before(self):
        a, bus = make(FakeMouth(moving=[(T0, T0 + 1200)]), vad_pause=1.0)
        feed(a, np.concatenate([tone(1.2), silence(1.5)]))
        self.assertEqual(len(bus.of(topics.UTTERANCE)), 1)

    def test_asks_the_tracker_for_the_last_pause_only(self):
        mouth = FakeMouth(moving=[(T0, T0 + 20_000)])
        a, bus = make(mouth, SimpleNamespace(mouth_endpoint=True, pause_seconds=1.5))
        feed(a, tone(6.0))
        self.assertTrue(mouth.calls)
        self.assertTrue(all(abs((b - x) - 1500.0) < 1.0 for x, b in mouth.calls))
        self.assertTrue(all(b - T0 >= 2000 for _, b in mouth.calls))     # never before pause + 0.5 s of speech


if __name__ == "__main__":
    unittest.main()
