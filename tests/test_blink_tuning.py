"""The blink click's tuning knobs: a longer allowed close, a minimum close time, and a gain for a weak eye.

Run with: python -m unittest tests.test_blink_tuning
"""
import unittest
from types import SimpleNamespace

from engine.config import settings
from engine.eyes.blink_gesture import BlinkDetector
from tests.test_blink_click import feed, open_eyes, shut


class TestMaxClose(unittest.TestCase):
    def test_a_close_between_1_2_and_1_6_seconds_clicks_when_allowed(self):
        levels = open_eyes(30) + shut(1400) + open_eyes(30)
        self.assertEqual(feed(BlinkDetector(), levels), [])                         # the old limit: resting eyes
        self.assertEqual(len(feed(BlinkDetector(max_closed_ms=1600), levels)), 1)

    def test_longer_than_the_new_limit_still_never_clicks(self):
        levels = open_eyes(30) + shut(1900) + open_eyes(30)
        self.assertEqual(feed(BlinkDetector(max_closed_ms=1600), levels), [])

    def test_the_cursor_freeze_follows_the_limit(self):
        d = BlinkDetector(max_closed_ms=1600)
        held = []
        for i, s in enumerate(open_eyes(30) + shut(1500)):
            d.update(i * 33, {"eyeBlinkLeft": s, "eyeBlinkRight": s})
            held.append(d.held)
        self.assertTrue(held[-1])                                                    # still counts as a click in the making
        d2 = BlinkDetector()
        for i, s in enumerate(open_eyes(30) + shut(1500)):
            d2.update(i * 33, {"eyeBlinkLeft": s, "eyeBlinkRight": s})
        self.assertFalse(d2.held)                                                    # the old limit let go


class TestMinClose(unittest.TestCase):
    def test_a_slow_tired_blink_clicks_by_default_but_not_with_a_higher_minimum(self):
        levels = open_eyes(30) + shut(450) + open_eyes(30)
        self.assertEqual(len(feed(BlinkDetector(), levels)), 1)
        self.assertEqual(feed(BlinkDetector(min_closed_ms=500), levels), [])

    def test_a_deliberate_close_still_clicks_with_the_higher_minimum(self):
        levels = open_eyes(30) + shut(700) + open_eyes(30)
        self.assertEqual(len(feed(BlinkDetector(min_closed_ms=500), levels)), 1)


class TestEyeGain(unittest.TestCase):
    """A weak right eye: it only reaches about 0.5 when shut."""

    def weak_right(self, ms=700):
        n_open, n_shut = 30, len(shut(ms))
        left = open_eyes(n_open) + [0.85] * n_shut + open_eyes(30)
        right = open_eyes(n_open) + [0.50] * n_shut + open_eyes(30)
        return left, right

    def test_a_weak_eye_misses_without_a_gain(self):
        left, right = self.weak_right()
        self.assertEqual(feed(BlinkDetector(), left, right), [])

    def test_a_gain_on_the_weak_eye_makes_the_click(self):
        left, right = self.weak_right()
        self.assertEqual(len(feed(BlinkDetector(right_gain=1.5), left, right)), 1)

    def test_a_wink_with_a_gain_still_never_clicks(self):
        n = len(shut(700))
        left = open_eyes(30) + [0.9] * n + open_eyes(30)
        right = open_eyes(60 + n)                     # the right eye stays open (0.08 * 1.5 is still low)
        self.assertEqual(feed(BlinkDetector(right_gain=1.5), left, right), [])
        self.assertEqual(feed(BlinkDetector(left_gain=1.5), right, left), [])

    def test_a_squint_with_a_gain_still_never_clicks(self):
        self.assertEqual(feed(BlinkDetector(left_gain=1.5, right_gain=1.5),
                              open_eyes(30) + [0.3] * 20 + open_eyes(30)), [])

    def test_a_gain_does_not_make_open_eyes_look_shut(self):
        d = BlinkDetector(left_gain=1.6, right_gain=1.6)
        self.assertEqual(feed(d, open_eyes(200)), [])
        self.assertLess(d.score, 0.2)

    def test_a_nonsense_gain_is_kept_positive(self):
        self.assertGreater(BlinkDetector(left_gain=0, right_gain=-2).right_gain, 0)


class TestSettingsReachTheDetector(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual((settings.blink_min_ms, settings.blink_max_ms), (400, 1600))
        self.assertEqual((settings.blink_left_gain, settings.blink_right_gain), (1.0, 1.0))

    def test_the_eyes_service_builds_the_detector_from_the_settings(self):
        from engine.core.events import EventBus
        from engine.eyes.head_pose import HeadPoseEstimator
        from engine.eyes.service import EyesService
        from tests.test_eyes_service import SETTINGS, FakeFeatures, FakeOs

        cfg = SimpleNamespace(**{**vars(SETTINGS), "blink_click": True, "blink_min_ms": 500,
                                 "blink_max_ms": 1500, "blink_right_gain": 1.4})
        eyes = EyesService(EventBus(), cfg, os_api=FakeOs(), features=FakeFeatures(),
                           head_pose=HeadPoseEstimator(), model_path="no-such-file.json")
        d = eyes.blink
        self.assertEqual((d.min_closed_ms, d.max_closed_ms, d.right_gain, d.left_gain), (500, 1500, 1.4, 1.0))


class TestReplaySuggestions(unittest.TestCase):
    def test_equal_eyes_need_no_gain(self):
        from tests.blink_replay import suggest_gains
        self.assertEqual(suggest_gains(0.80, 0.78, 0.55), (1.0, 1.0))

    def test_a_weak_right_eye_gets_a_right_gain(self):
        from tests.blink_replay import suggest_gains
        self.assertEqual(suggest_gains(0.90, 0.50, 0.55), (1.0, 1.8))
        left, right = suggest_gains(0.80, 0.60, 0.55)
        self.assertEqual(left, 1.0)
        self.assertAlmostEqual(right, 1.33, places=2)

    def test_a_weak_left_eye_gets_a_left_gain(self):
        from tests.blink_replay import suggest_gains
        self.assertEqual(suggest_gains(0.60, 0.90, 0.55)[1], 1.0)
        self.assertGreater(suggest_gains(0.60, 0.90, 0.55)[0], 1.0)

    def test_missing_peaks_mean_no_gain(self):
        from tests.blink_replay import suggest_gains
        self.assertEqual(suggest_gains(None, 0.8, 0.55), (1.0, 1.0))

    def test_eye_peaks_from_a_recording(self):
        from tests.blink_replay import eye_peaks
        frames = []
        t = 0
        for phase, n, l, r in (("look", 40, 0.08, 0.10), ("click", 8, 0.85, 0.50), ("click", 20, 0.08, 0.10),
                               ("click", 8, 0.88, 0.52), ("click", 20, 0.08, 0.10)):
            for _ in range(n):
                frames.append({"t": t, "phase": phase, "row": "middle", "col": "1", "face": True, "L": l, "R": r,
                               "pitch": 0.0})
                t += 33
        lp, rp = eye_peaks(frames)
        self.assertAlmostEqual(lp, 0.865, places=2)
        self.assertAlmostEqual(rp, 0.51, places=2)


if __name__ == "__main__":
    unittest.main()
