"""The 'head-push' pointer (engine/eyes/pointer.py): a quick turn away from where the head rests pushes the cursor; the
way back (an overshoot past the start, wobble, settling back) moves nothing; wherever the head then settles, after a
split second, is where the next push is measured from. The 'head' mode must be unchanged.
Run: python -m unittest tests.test_head_push
"""
import math
import random
import unittest

from engine.core.types import HeadPose
from engine.eyes.pointer import HeadRelPointer, Pointer, _away

W, H = 1920, 1080
MID = (W - 1) / 2


class Head:
    """A head that turns smoothly (eased), with measured resting noise (Khang's logs: about 0.025 deg yaw, 0.04 pitch)."""

    def __init__(self, push=True, fps=32, noise=(0.025, 0.04), seed=7):
        self.p = HeadRelPointer(W, H)
        self.p.push = push
        self.t = 0
        self.fps = fps
        self.noise = noise
        self.rnd = random.Random(seed)
        self.yaw = self.pitch = 0.0

    def frame(self):
        self.t += round(1000 / self.fps)
        return self.p.update(HeadPose(self.t, self.yaw + self.rnd.gauss(0, self.noise[0]),
                                      self.pitch + self.rnd.gauss(0, self.noise[1]), 0.0))

    def to(self, yaw=None, pitch=None, seconds=0.25, hold=0.0):
        """Turn to (yaw, pitch) over `seconds` (eased in and out), then hold still for `hold` seconds."""
        y0, p0 = self.yaw, self.pitch
        y1 = y0 if yaw is None else yaw
        p1 = p0 if pitch is None else pitch
        n = max(1, round(seconds * self.fps))
        for i in range(1, n + 1):
            s = (1 - math.cos(math.pi * i / n)) / 2
            self.yaw, self.pitch = y0 + (y1 - y0) * s, p0 + (p1 - p0) * s
            self.frame()
        for _ in range(round(hold * self.fps)):
            self.frame()
        return self.p.cursor

    def rest(self, seconds):
        for _ in range(round(seconds * self.fps)):
            self.frame()
        return self.p.cursor

    def come_back(self, to=0.0, overshoot=2.0, seconds=0.4):
        """Back like a person: past the spot by `overshoot` degrees, then settle on it."""
        side = 1.0 if self.yaw >= to else -1.0
        self.to(yaw=to - side * overshoot, seconds=seconds)
        return self.to(yaw=to, seconds=0.3)


class TestAway(unittest.TestCase):
    def test_moving_out_counts(self):
        self.assertAlmostEqual(_away(2.0, 5.0), 3.0)
        self.assertAlmostEqual(_away(-2.0, -5.0), -3.0)

    def test_coming_back_counts_nothing(self):
        self.assertEqual(_away(5.0, 2.0), 0.0)
        self.assertEqual(_away(-5.0, -1.0), 0.0)

    def test_crossing_the_start_counts_from_the_start_on(self):
        self.assertAlmostEqual(_away(3.0, -2.0), -2.0)
        self.assertAlmostEqual(_away(-1.0, 4.0), 4.0)


class TestPush(unittest.TestCase):
    def setUp(self):
        self.h = Head()
        self.h.rest(0.6)

    def test_a_quick_push_moves_the_cursor_that_way(self):
        x, _ = self.h.to(yaw=10, seconds=0.25, hold=0.15)
        self.assertGreater(x - MID, 300)
        self.assertEqual(self.h.p.pushes, 1)

    def test_the_way_back_moves_nothing_even_with_an_overshoot(self):
        for overshoot in (0.0, 1.0, 2.0, 3.0, 4.0):
            for settle in (0.0, 0.2):
                h = Head(seed=int(overshoot * 10) + int(settle * 100))
                h.rest(0.6)
                x1, _ = h.to(yaw=10, seconds=0.25, hold=0.15)
                h.to(yaw=-overshoot, seconds=0.4)
                h.rest(settle)                         # a stop at the overshoot...
                h.to(yaw=0.0, seconds=0.3)             # ...then settling back toward the start
                h.rest(0.5)
                self.assertEqual(h.p.cursor[0], x1, f"overshoot {overshoot}, stop {settle}")

    def test_a_wobble_on_the_way_back_moves_nothing(self):
        x1, _ = self.h.to(yaw=10, seconds=0.25, hold=0.15)
        self.h.to(yaw=6, seconds=0.15)
        self.h.to(yaw=6.8, seconds=0.08)            # a little bounce outward on the way back
        self.h.to(yaw=0, seconds=0.25)
        x2, _ = self.h.rest(0.5)
        self.assertEqual(x2, x1)

    def test_states_go_ready_push_return_ready(self):
        seen = [self.h.p.push_state]
        orig = self.h.frame

        def frame():
            r = orig()
            if self.h.p.push_state != seen[-1]:
                seen.append(self.h.p.push_state)
            return r

        self.h.frame = frame
        self.h.to(yaw=10, seconds=0.25, hold=0.1)
        self.h.come_back(overshoot=1.5)
        self.h.rest(0.4)
        self.assertEqual(seen, ["ready", "push", "return", "ready"])

    def test_the_next_push_is_taken_a_split_second_after_stopping(self):
        x1, _ = self.h.to(yaw=10, seconds=0.25, hold=0.1)
        self.h.to(yaw=0.5, seconds=0.35)
        self.h.rest(0.2)                            # a short stop...
        x2, _ = self.h.to(yaw=-6, seconds=0.25)     # ...and a push the other way counts
        self.assertLess(x2, x1 - 100)

    def test_the_resting_spot_does_not_have_to_be_the_start(self):
        x1, _ = self.h.to(yaw=10, seconds=0.25, hold=0.1)
        self.h.to(yaw=4, seconds=0.35)              # back to where it's comfortable, not all the way
        self.h.rest(0.3)
        self.assertEqual(self.h.p.push_state, "ready")
        self.assertAlmostEqual(self.h.p.home[0], 4.0, delta=0.3)   # the new start point
        self.assertEqual(self.h.p.cursor[0], x1)
        x2, _ = self.h.to(yaw=-2, seconds=0.25)     # a 6-degree push left from there counts in full
        self.assertLess(x2, x1 - 100)

    def test_holding_out_there_then_coming_back_never_counts(self):
        self.h.to(yaw=10, seconds=0.25)
        x1, _ = self.h.rest(2.0)                    # hold the head turned...
        self.h.to(yaw=8.5, seconds=0.6)             # ...relax a little...
        self.h.rest(1.0)                            # ...and stay there a while
        self.assertEqual(self.h.p.push_state, "ready")
        self.h.come_back(overshoot=2.0)             # then come back, with an overshoot
        self.h.rest(0.5)
        self.assertLess(abs(self.h.p.cursor[0] - x1), 2.0)

    def test_held_out_there_a_push_further_still_counts(self):
        x1, _ = self.h.to(yaw=8, seconds=0.25, hold=1.0)   # push, then hold the head turned
        x2, _ = self.h.to(yaw=14, seconds=0.2)             # a quick turn further out: a new push
        self.assertGreater(x2 - x1, 100)
        x3, _ = self.h.come_back(overshoot=2.0)            # and the long way back counts nothing
        self.assertLess(abs(x3 - x2), 2.0)                 # (the filtered angle finishes the push's last pixel)

    def test_push_back_push_the_same_way_needs_no_pause(self):
        x1, _ = self.h.to(yaw=8, seconds=0.25)
        self.h.to(yaw=0, seconds=0.3)               # straight back, no stop...
        x2, _ = self.h.to(yaw=8, seconds=0.25)      # ...and straight out again
        self.assertGreater(x2 - x1, 0.6 * (x1 - MID))
        self.assertEqual(self.h.p.pushes, 2)

    def test_after_an_overshoot_a_push_the_same_way_counts_past_the_start(self):
        x1, _ = self.h.to(yaw=8, seconds=0.25, hold=0.1)
        self.h.to(yaw=-2, seconds=0.35)             # back with an overshoot, and a stop there
        self.h.rest(0.2)
        x2, _ = self.h.to(yaw=8, seconds=0.35)      # push right again from there
        self.assertGreater(x2 - x1, 120)            # counts (from just past the old start)
        self.assertLess(x2 - x1, 1.1 * (x1 - MID))  # but the overshoot's 2 degrees back don't add to it

    def test_slow_drifting_moves_nothing_and_the_start_follows(self):
        x0, y0 = self.h.p.cursor
        x, y = self.h.to(yaw=3, pitch=-2, seconds=2.0)    # 1.8 deg/s
        self.assertEqual((x, y), (x0, y0))
        self.h.rest(0.2)
        self.assertAlmostEqual(self.h.p.home[0], 3.0, delta=0.3)

    def test_a_small_quick_nudge_moves_a_little(self):
        x, _ = self.h.to(yaw=1.0, seconds=0.12)            # about 8 deg/s: fine aiming
        self.assertGreater(x - MID, 4)
        self.assertLess(x - MID, 40)

    def test_still_head_keeps_the_cursor_still(self):
        x0, y0 = self.h.p.cursor
        x, y = self.h.rest(5.0)
        self.assertEqual((x, y), (x0, y0))

    def test_up_and_down_work_the_same(self):
        _, y1 = self.h.to(pitch=6, seconds=0.25, hold=0.1)
        self.assertGreater(y1 - (H - 1) / 2, 100)
        self.h.to(pitch=-1.0, seconds=0.35)               # back, a little past the start...
        self.h.to(pitch=0.0, seconds=0.3)                  # ...and settle
        _, y3 = self.h.rest(0.4)
        self.assertEqual(y3, y1)

    def test_center_sets_a_new_start(self):
        self.h.to(yaw=8, seconds=0.25)
        self.h.p.center()
        self.h.rest(0.3)
        self.assertAlmostEqual(self.h.p.home[0], 8.0, delta=0.3)
        self.assertEqual(self.h.p.push_state, "ready")
        x, _ = self.h.to(yaw=3, seconds=0.25)              # a quick turn left from there moves left
        self.assertLess(x, MID - 40)

    def test_a_slow_camera_works_too(self):
        h = Head(fps=12)
        h.rest(0.6)
        x1, _ = h.to(yaw=10, seconds=0.25, hold=0.2)
        h.come_back(overshoot=2.0)
        h.rest(0.6)
        self.assertGreater(x1 - MID, 250)
        self.assertEqual(h.p.cursor[0], x1)
        x2, _ = h.to(yaw=-6, seconds=0.3)                  # and the next push still works
        self.assertLess(x2, x1 - 80)

    def test_a_rebase_keeps_the_state(self):
        self.h.to(yaw=10, seconds=0.25)
        self.h.to(yaw=7, seconds=0.15)
        self.assertEqual(self.h.p.push_state, "return")
        x1 = self.h.p.cursor[0]
        self.h.p.rebase()                                  # a blink hold or a UI-Venus snap
        self.h.to(yaw=-2, seconds=0.3)
        self.h.to(yaw=0, seconds=0.3)
        self.assertEqual(self.h.p.cursor[0], x1)


class TestHeadUnchanged(unittest.TestCase):
    def test_head_mode_still_moves_back(self):
        h = Head(push=False)
        h.rest(0.5)
        x1, _ = h.to(yaw=10, seconds=0.25, hold=0.1)
        x2, _ = h.to(yaw=0, seconds=0.25, hold=0.1)
        self.assertLess(x2, x1 - 200)                      # 'head' brings the cursor back on a quick return

    def test_head_mode_slow_aiming_still_moves(self):
        h = Head(push=False)
        h.rest(0.5)
        x, _ = h.to(yaw=2, seconds=2.0)
        self.assertGreater(x - MID, 10)


class TestPointerMode(unittest.TestCase):
    def test_mode_turns_push_on_and_off(self):
        p = Pointer(W, H, "head-push")
        self.assertTrue(p.rel.push)
        self.assertEqual(p.still_px, 2.0)
        p.set_mode("head")
        self.assertFalse(p.rel.push)

    def test_place_keeps_the_start_point(self):
        p = Pointer(W, H, "head-push")
        t = 0
        for _ in range(10):
            t += 33
            p.update(t, None, HeadPose(t, 3.0, 0.0, 0.0))
        home = p.rel.home
        p.place(100, 100)
        self.assertEqual(p.rel.home, home)
        self.assertEqual(p.rel.cursor, (100.0, 100.0))


if __name__ == "__main__":
    unittest.main()
