"""The head pointer's feel: head as a mouse with acceleration (engine/eyes/pointer.py, 'head' mode).
Run: python -m unittest tests.test_pointer"""
import random
import unittest

from engine.core.types import EyeFeatures, HeadPose
from engine.eyes.pointer import MAX_STEP_PX, HeadRelPointer, Pointer

W, H = 1920, 1080


class Head:
    def __init__(self):
        self.p = HeadRelPointer(W, H)
        self.t = 0
        self.yaw = self.pitch = 0.0

    def go(self, d_yaw=0.0, d_pitch=0.0, seconds=0.5, noise=0.0, settle=0.3):
        """Turn by d_yaw, d_pitch degrees at an even speed over `seconds`, then hold still for `settle`."""
        n = max(1, int(seconds * 30))
        for i in range(n + int(settle * 30)):
            if i < n:
                self.yaw += d_yaw / n
                self.pitch += d_pitch / n
            self.t += 33
            pt = self.p.update(HeadPose(self.t, self.yaw + random.gauss(0, noise),
                                        self.pitch + random.gauss(0, noise), 0.0))
        return pt.x, pt.y


class TestHeadMouse(unittest.TestCase):
    def setUp(self):
        random.seed(5)
        self.h = Head()
        self.h.go(seconds=0.5)   # settle at the center

    def test_a_quick_turn_goes_far(self):
        x, _ = self.h.go(d_yaw=10, seconds=0.2)        # a quick 10-degree turn, about 50 deg/s
        self.assertGreater(x - 959.5, 330)
        self.assertLess(x - 959.5, 480)

    def test_a_slow_turn_is_precise(self):
        x, _ = self.h.go(d_yaw=2, seconds=2.0)         # 1 deg/s
        self.assertLess(x - 959.5, 35)
        self.assertGreater(x - 959.5, 10)

    def test_still_head_keeps_the_cursor_still(self):
        x0, y0 = self.h.p.cursor
        x, y = self.h.go(seconds=0.1, settle=3.0, noise=0.03)   # measured tracking noise on the demo PC
        self.assertLess(abs(x - x0) + abs(y - y0), 6)

    def test_reaching_a_corner_takes_three_quick_turns(self):
        for _ in range(2):
            self.h.go(d_yaw=-12, d_pitch=-8, seconds=0.25)      # up-left, quickly...
            self.h.go(d_yaw=12, d_pitch=8, seconds=2.5)         # ...head back slowly (cursor stays)...
        x, y = self.h.go(d_yaw=-12, d_pitch=-8, seconds=0.25)  # ...and again
        self.assertLess(x, 40)
        self.assertLess(y, 40)

    def test_no_fixed_center_turning_back_slowly_keeps_the_cursor(self):
        x1, _ = self.h.go(d_yaw=10, seconds=0.2)        # throw right
        x2, _ = self.h.go(d_yaw=-10, seconds=3.0)       # bring the head back slowly
        self.assertGreater(x2 - 959.5, 250)              # the cursor mostly stays right: like lifting a mouse
        self.assertLess(x2, x1)

    def test_a_tracking_glitch_is_ignored(self):
        x0, y0 = self.h.p.cursor
        self.h.t += 33
        self.h.p.update(HeadPose(self.h.t, 25.0, -20.0, 0.0))   # one wild frame
        self.h.t += 33
        x, y = self.h.p.update(HeadPose(self.h.t, 0.0, 0.0, 0.0)).x, self.h.p.cursor[1]
        self.assertLess(abs(x - x0) + abs(y - y0), 60)

    def test_stays_on_screen(self):
        for _ in range(3):
            x, y = self.h.go(d_yaw=40, d_pitch=40, seconds=0.3)
        self.assertEqual((x, y), (W - 1, H - 1))

    def test_no_single_frame_jump(self):
        prev = self.h.p.cursor
        for _ in range(10):
            self.h.t += 33
            self.h.yaw += 8.0                                    # 240 deg/s: as fast as a head turns
            pt = self.h.p.update(HeadPose(self.h.t, self.h.yaw, 0.0, 0.0))
            self.assertLessEqual(pt.x - prev[0], MAX_STEP_PX + 0.01)
            prev = (pt.x, pt.y)

    def test_a_real_fast_turn_is_not_stuck_as_a_glitch(self):
        x0 = self.h.p.cursor[0]
        for _ in range(3):                                       # the head really jumped 15 degrees (or a frame drop)
            self.h.t += 33
            self.h.p.update(HeadPose(self.h.t, 15.0, 0.0, 0.0))
        self.h.yaw = 15.0
        x1, _ = self.h.go(d_yaw=3, seconds=0.3)                  # it moves again from there
        self.assertLess(abs(self.h.p.cursor[0] - x1), 1)
        self.assertGreater(x1 - x0, 20)

    def test_pointer_place_and_modes(self):
        p = Pointer(W, H, "head")
        p.place(300, 200)
        pt = p.update(10, None, HeadPose(10, 0.0, 0.0, 0.0))
        self.assertEqual((pt.x, pt.y), (300, 200))
        self.assertTrue(p.set_mode("head-abs"))
        self.assertFalse(p.set_mode("hybrid"))          # no gaze model

    def test_looking_down_is_not_a_blink_freeze_then_jump(self):
        """Real log 2026-10-24: looking at the bottom edge drops the eyelids, read as a 16-frame 'blink'.
        The cursor froze, then jumped 500-800 px when the eyes opened."""
        p, t, pitch, prev = Pointer(W, H, "head"), 0, 0.0, None
        for i in range(90):
            t += 33
            closed = 20 <= i < 40                       # eyelids 'closed' for 0.66 s while the chin keeps moving
            pitch += 0.4 if i < 60 else 0.0
            pt = p.update(t, EyeFeatures(t, 0.5, 0.1, 0.3, usable=not closed), HeadPose(t, 0.0, pitch, 0.0))
            if prev is not None:
                self.assertLess(abs(pt.y - prev), 40, f"frame {i}")
            prev = pt.y
        self.assertGreater(prev, (H - 1) / 2 + 100)    # it kept moving down after the hold ended

    def test_a_short_blink_holds_the_cursor(self):
        p, t = Pointer(W, H, "head"), 0
        for i in range(30):
            t += 33
            blink = 10 <= i < 14                        # a real blink, 0.13 s, nudges the angles 2 degrees
            pt = p.update(t, EyeFeatures(t, 0.5, 0.1, 0.3, usable=not blink),
                          HeadPose(t, 0.0, 2.0 if blink else 0.0, 0.0))
        self.assertAlmostEqual(pt.y, (H - 1) / 2, delta=3)


if __name__ == "__main__":
    unittest.main()
