"""The head pointer's feel: head as a mouse with acceleration (engine/eyes/pointer.py, 'head' mode).
Run: python -m unittest tests.test_pointer
The edge sweep (where turns of 10-25 degrees at each speed and frame rate end):  python -m tests.test_pointer --sweep
"""
import random
import sys
import unittest

from engine.core.types import EyeFeatures, HeadPose
from engine.eyes.pointer import MAX_STEP_PX, HeadRelPointer, Pointer

W, H = 1920, 1080


class Head:
    def __init__(self, fps=30):
        self.p = HeadRelPointer(W, H)
        self.t = 0
        self.fps = fps
        self.yaw = self.pitch = 0.0

    def go(self, d_yaw=0.0, d_pitch=0.0, seconds=0.5, noise=0.0, settle=0.3):
        """Turn by d_yaw, d_pitch degrees at an even speed over `seconds`, then hold still for `settle`."""
        n = max(1, round(seconds * self.fps))
        for i in range(n + round(settle * self.fps)):
            if i < n:
                self.yaw += d_yaw / n
                self.pitch += d_pitch / n
            self.t += round(1000 / self.fps)
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


class TestEdges(unittest.TestCase):
    """H2: the cursor didn't reach the left and top edges. The sweep (--sweep) found the pointer math is the same
    in every direction, but (1) a hard throw moved the cursor LESS than a medium one (MAX_STEP_PX cut it), and
    (2) at a low frame rate (Hung's webcam at 8 fps in dim light, or dropped frames) fast turns were cut more or
    taken for glitches, so the cursor didn't move at all."""

    def setUp(self):
        random.seed(5)

    def throw(self, d_yaw=0.0, d_pitch=0.0, speed=150.0, fps=30):
        h = Head(fps)
        h.go(seconds=0.5)
        return h, h.go(d_yaw, d_pitch, seconds=max(abs(d_yaw), abs(d_pitch)) / speed, noise=0.03, settle=0.7)

    def test_a_hard_throw_reaches_the_left_and_top_edges(self):
        for speed in (60, 100, 150):
            self.assertLessEqual(self.throw(d_yaw=-22, speed=speed)[1][0], 4, f"left at {speed} deg/s")
            self.assertLessEqual(self.throw(d_pitch=-16, speed=speed)[1][1], 4, f"top at {speed} deg/s")
            self.assertGreaterEqual(self.throw(d_yaw=22, speed=speed)[1][0], W - 5, f"right at {speed} deg/s")

    def test_a_hard_throw_that_ends_on_the_screen_is_cut_as_before(self):
        _, (x, _) = self.throw(d_yaw=-10, speed=150)                 # the middle keeps its feel
        self.assertGreater(x, 959.5 - 330)
        self.assertLess(x, 959.5 - 150)

    def test_at_8_fps_a_fast_turn_still_moves(self):
        _, (x, _) = self.throw(d_yaw=-15, speed=100, fps=8)          # before: not at all (taken for glitches)
        self.assertLess(x, 959.5 - 250)
        _, (x, _) = self.throw(d_yaw=-25, speed=60, fps=8)           # before: stopped at x 664
        self.assertLessEqual(x, 2)

    def test_turning_back_drops_the_rest_of_the_throw(self):
        h = Head()
        h.go(seconds=0.5)
        h.go(d_yaw=-25, seconds=25 / 150, settle=0.0)                 # thrown at the left edge...
        x_mid = h.p.cursor[0]
        x, _ = h.go(d_yaw=8, seconds=0.3, settle=0.5)                 # ...and right back
        self.assertGreater(x_mid, 100)
        self.assertGreater(x, x_mid)

    def test_frame_limits_are_unchanged_at_25_fps_and_up(self):
        for dt in (1 / 60, 1 / 30, 1 / 25):
            self.assertEqual(HeadRelPointer.frame_scale(dt), 1.0)
        self.assertGreater(HeadRelPointer.frame_scale(1 / 8), 3.0)


def sweep() -> None:
    """Print where the cursor ends after turns from the center, and after a slow (5 deg/s) turn back."""
    random.seed(1)
    speeds = (5, 20, 60, 100, 150, 300)
    for fps in (30, 15, 8):
        print(f"\n{fps} fps. Each cell: cursor x (yaw) or y (pitch) after the turn / after a slow turn back. "
              f"Edges: x 0..{W - 1}, y 0..{H - 1}")
        print("               " + "".join(f"{s:>5} deg/s  " for s in speeds))
        for axis, sign, name in (("yaw", -1, "left"), ("pitch", -1, "up"), ("yaw", 1, "right"), ("pitch", 1, "down")):
            for deg in (10, 15, 20, 25):
                cells = []
                for speed in speeds:
                    h = Head(fps)
                    h.go(seconds=0.5)
                    d = sign * deg
                    kw = {"d_yaw": d} if axis == "yaw" else {"d_pitch": d}
                    out = h.go(seconds=deg / speed, noise=0.03, settle=0.7, **kw)
                    kw = {k: -v for k, v in kw.items()}
                    back = h.go(seconds=deg / 5, noise=0.03, settle=0.3, **kw)
                    i = 0 if axis == "yaw" else 1
                    cells.append(f"{out[i]:5.0f}/{back[i]:5.0f} ")
                print(f"  {name:<5} {deg:2d} deg " + " ".join(cells))


if __name__ == "__main__":
    if "--sweep" in sys.argv:
        sweep()
    else:
        unittest.main()
