"""The long-blink click, with no camera or model: BlinkDetector on made-up blink curves, the mode manager's
meaning of a "blink" gesture, and the eyes service end to end (cursor freeze, one click).

Run with: python -m unittest tests.test_blink_click
"""
import math
import unittest

from engine.core import messages, topics
from engine.core.mode_manager import Mode
from engine.core.types import TargetResult
from engine.eyes.blink_gesture import BlinkDetector
from tests.test_dwell_click import ButtonOs
from tests.test_eyes_service import Rig, face
from tests.test_mode_manager import Base

FRAME_MS = 33


def feed(d, left, right=None, t0=0):
    """Blink scores per frame, 33 ms apart; returns the frame indexes where it fired."""
    right = left if right is None else right
    out = []
    for i, (a, b) in enumerate(zip(left, right)):
        if d.update(t0 + i * FRAME_MS, {"eyeBlinkLeft": a, "eyeBlinkRight": b}) is not None:
            out.append(i)
    return out


def open_eyes(frames):
    return [0.08] * frames


def shut(ms, level=0.85):
    return [level] * round(ms / FRAME_MS)


class TestBlinkDetector(unittest.TestCase):
    def test_a_long_blink_fires_once_when_the_eyes_open(self):
        levels = open_eyes(30) + shut(600) + open_eyes(30)
        self.assertEqual(feed(BlinkDetector(), levels), [30 + len(shut(600))])

    def test_a_normal_blink_never_fires(self):
        for ms in (100, 200, 300):
            with self.subTest(ms=ms):
                self.assertEqual(feed(BlinkDetector(), open_eyes(30) + shut(ms) + open_eyes(30)), [])

    def test_two_quick_blinks_never_fire(self):
        levels = open_eyes(30) + shut(200) + open_eyes(4) + shut(200) + open_eyes(30)
        self.assertEqual(feed(BlinkDetector(), levels), [])

    def test_resting_the_eyes_never_fires(self):
        self.assertEqual(feed(BlinkDetector(), open_eyes(30) + shut(2500) + open_eyes(30)), [])

    def test_a_wink_never_fires(self):
        left = open_eyes(30) + shut(700) + open_eyes(30)
        self.assertEqual(feed(BlinkDetector(), left, open_eyes(len(left))), [])

    def test_a_squint_never_fires(self):
        self.assertEqual(feed(BlinkDetector(), open_eyes(30) + [0.5] * 20 + open_eyes(30)), [])

    def test_a_long_blink_with_wobbling_scores_is_one_click(self):
        wobble = [0.9 - 0.2 * abs(math.sin(i)) for i in range(20)]     # stays well over the open level
        self.assertEqual(len(feed(BlinkDetector(), open_eyes(30) + wobble + open_eyes(30))), 1)

    def test_cooldown_stops_a_second_click_right_after(self):
        levels = open_eyes(30) + shut(500) + open_eyes(5) + shut(500) + open_eyes(30)
        self.assertEqual(len(feed(BlinkDetector(), levels)), 1)

    def test_a_gap_in_the_frames_drops_the_close(self):
        d = BlinkDetector()
        feed(d, open_eyes(30) + shut(300))
        t = (30 + len(shut(300))) * FRAME_MS
        self.assertIsNone(d.update(t + 2000, {"eyeBlinkLeft": 0.05, "eyeBlinkRight": 0.05}))
        self.assertFalse(d.held)

    def test_no_blink_scores_means_nothing(self):
        d = BlinkDetector()
        self.assertIsNone(d.update(0, {}))
        self.assertFalse(d.held)

    def test_held_while_closed_for_the_cursor_freeze(self):
        d = BlinkDetector()
        feed(d, open_eyes(30) + shut(300))
        self.assertTrue(d.held)
        feed(d, open_eyes(3), t0=(30 + len(shut(300))) * FRAME_MS)
        self.assertFalse(d.held)


class TestModeManagerBlink(Base):
    def requests(self):
        return self.bus.of(topics.TARGET_REQUEST)

    def test_a_blink_with_nothing_waiting_is_the_snap_left_click(self):
        self.look(500, 300)
        self.gesture("blink")
        req = self.requests()[-1]
        self.assertTrue(req.snap)
        self.bus.publish(topics.TARGET_RESULT, TargetResult(req.label, True, 520, 310))
        self.assertEqual(self.os(), [(topics.OS_CLICK, {"x": 520, "y": 310})])

    def test_a_blink_answers_a_question_like_a_raise(self):
        self.say("start dictation")
        self.say("hello there")
        self.assertTrue(self.mm.preview)
        self.gesture("blink")
        self.assertEqual(self.os(), [(topics.OS_TYPE, {"text": "hello there "})])

    def test_a_blink_is_never_a_right_click_even_with_brow_click_right(self):
        self.mm._brow_click = "right"
        self.look(500, 300)
        self.gesture("blink")
        self.assertTrue(self.requests())
        self.assertEqual(self.os(), [])

    def test_a_blink_leaves_scroll_mode(self):
        self.say("scroll")
        self.assertEqual(self.mm.mode, Mode.SCROLL)
        self.gesture("blink")
        self.assertEqual(self.mm.mode, Mode.POINTER)

    def test_a_blink_does_nothing_while_paused(self):
        self.say("pause")
        self.gesture("blink")
        self.assertEqual(self.os(), [])
        self.assertEqual(self.requests(), [])


def blink_face(t_ms, closed, yaw=0.0):
    f = face(t_ms, yaw=yaw)
    f.blendshapes["eyeBlinkLeft"] = f.blendshapes["eyeBlinkRight"] = 0.85 if closed else 0.05
    return f


class TestEyesServiceBlink(unittest.TestCase):
    def setUp(self):
        self.r = Rig()
        self.r.os = self.r.out.os = ButtonOs()
        self.r.eyes.blink = BlinkDetector()
        self.r.frames(60)   # the one-look calibration

    def tearDown(self):
        self.r.out.stop()

    def blink_frames(self, n, closed, yaw=0.0):
        r = self.r
        for _ in range(n):
            r.t_ms += 33
            r.clock.t += 0.033
            r.eyes.process(blink_face(r.t_ms, closed, r.sit[0] + yaw), r.clock.t)

    def test_a_long_blink_clicks_once_where_the_cursor_was(self):
        self.blink_frames(20, False, yaw=4.0)                 # point, eyes open
        self.blink_frames(10, False, yaw=4.0)
        start = self.r.bus.of(topics.GAZE)[-1]
        self.blink_frames(18, True, yaw=7.0)                  # about 0.6 s shut; the head drifts meanwhile
        self.blink_frames(5, False, yaw=7.0)
        self.r.out.flush()
        req = self.r.bus.of(topics.TARGET_REQUEST)[-1]
        self.assertTrue(req.snap)
        self.assertLess(math.hypot(req.gaze_x - start.x, req.gaze_y - start.y), 15)   # frozen, not where the head went
        clicks = [c for c in self.r.os.clicks if len(c) == 2]
        self.assertEqual(len(clicks), 1)

    def test_normal_blinks_while_pointing_never_click(self):
        for _ in range(6):
            self.blink_frames(12, False, yaw=4.0)
            self.blink_frames(4, True, yaw=4.0)               # about 0.13 s
        self.blink_frames(10, False, yaw=4.0)
        self.r.out.flush()
        self.assertEqual(self.r.os.clicks, [])
        self.assertEqual(self.r.bus.of(topics.TARGET_REQUEST), [])

    def test_off_by_default(self):
        r = Rig()
        self.assertIsNone(r.eyes.blink)
        r.out.stop()


if __name__ == "__main__":
    unittest.main()
