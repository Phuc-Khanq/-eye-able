"""MouthTracker on synthetic jawOpen traces.  Run: python -m unittest tests.test_mouth"""
import math
import random
import unittest

from engine.core import topics
from engine.core.types import MouthSample
from engine.eyes.mouth import MouthTracker
from tests.helpers import FakeBus


def feed(m, t0, ms, f, fps=30):
    """Frames from t0 for ms milliseconds, jawOpen = f(t in seconds)."""
    step = 1000 / fps
    n = int(ms / step)
    for i in range(n):
        t = t0 + i * step
        m.add(MouthSample(t_ms=int(t), open=f((t - t0) / 1000)))
    return t0 + n * step


def talking(t):
    return 0.15 + 0.12 * math.sin(2 * math.pi * 4 * t) + random.uniform(-0.01, 0.01)  # jaw up and down, 4 Hz


def silent(t):
    return 0.03 + random.uniform(-0.006, 0.006)


class TestMouth(unittest.TestCase):
    def setUp(self):
        random.seed(3)
        self.m = MouthTracker()

    def test_talking_is_moving_silence_is_not(self):
        t = feed(self.m, 10_000, 5_000, silent)
        t2 = feed(self.m, t, 3_000, talking)
        self.assertLess(self.m.moving_fraction(11_000, t - 300), 0.05)
        self.assertGreater(self.m.moving_fraction(t + 200, t2 - 200), 0.8)

    def test_a_long_open_vowel_counts(self):
        t = feed(self.m, 10_000, 5_000, silent)
        t2 = feed(self.m, t, 1_000, lambda s: 0.45 + random.uniform(-0.005, 0.005))  # "aaah", jaw held open
        self.assertGreater(self.m.moving_fraction(t + 200, t2), 0.8)

    def test_no_face_means_cant_tell(self):
        feed(self.m, 10_000, 2_000, silent)
        self.assertIsNone(self.m.moving_fraction(20_000, 21_000))   # nothing recorded then

    def test_mixed_span_splits_by_time(self):
        t = feed(self.m, 10_000, 4_000, silent)
        t2 = feed(self.m, t, 1_000, talking)          # the user speaks for 1 s
        t3 = feed(self.m, t2, 1_000, silent)          # then a teammate speaks; the user's mouth is still
        self.assertGreater(self.m.moving_fraction(t + 150, t2 - 150), 0.8)
        self.assertLess(self.m.moving_fraction(t2 + 200, t3), 0.2)

    def test_slow_camera_still_answers(self):
        t = feed(self.m, 10_000, 5_000, silent, fps=8)
        t2 = feed(self.m, t, 2_000, talking, fps=8)
        self.assertIsNotNone(self.m.moving_fraction(t, t2))
        self.assertLess(self.m.moving_fraction(11_000, t - 300), 0.2)

    def test_reads_samples_from_the_bus(self):
        bus = FakeBus()
        m = MouthTracker(bus)
        for i in range(30):
            bus.publish(topics.MOUTH, MouthSample(t_ms=1000 + 33 * i, open=talking(i / 30)))
        self.assertIsNotNone(m.moving_fraction(1000, 1990))


if __name__ == "__main__":
    unittest.main()
