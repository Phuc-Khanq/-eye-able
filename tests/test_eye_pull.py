"""The head-eyes pointer (engine/eyes/pointer.py) and the head pointer's rest gate, with a made-up gaze model.

Run with: python -m unittest tests.test_eye_pull
"""
import math
import random
import unittest

import engine.eyes.pointer as P
from engine.core.types import EyeFeatures, HeadPose
from engine.eyes.pointer import EYE_RADIUS_PX, HeadRelPointer, Pointer

W, H = 1920, 1080
FPS = 30


class FakeGaze:
    """A gaze model that says you look at self.at (screen px), whatever the eyes and head do."""

    def __init__(self, at=(960.0, 540.0)):
        self.at = at

    def predict(self, feats, pose):
        return self.at


def eyes(t_ms, usable=True):
    return EyeFeatures(t_ms=t_ms, h=0.5, v=0.0, lid=0.3, usable=usable)


class Rig:
    def __init__(self, mode="head-eyes", look=(960.0, 540.0)):
        self.gaze = FakeGaze(look)
        self.p = Pointer(W, H, mode, model=self.gaze)
        self.t, self.yaw, self.pitch = 0, 0.0, 0.0
        self.point = None

    def go(self, d_yaw=0.0, d_pitch=0.0, seconds=0.4, settle=0.3, noise=0.0, usable=True):
        n = max(1, round(seconds * FPS))
        for i in range(n + round(settle * FPS)):
            if i < n:
                self.yaw += d_yaw / n
                self.pitch += d_pitch / n
            self.t += round(1000 / FPS)
            pose = HeadPose(self.t, self.yaw + random.gauss(0, noise), self.pitch + random.gauss(0, noise), 0.0)
            self.point = self.p.update(self.t, eyes(self.t, usable), pose)
        return self.point.x, self.point.y


class TestEyePull(unittest.TestCase):
    def setUp(self):
        random.seed(3)

    def test_the_eyes_carry_the_cursor_further_toward_where_you_look(self):
        head = Rig("head")
        head.go(seconds=0.3)
        hx, _ = head.go(d_yaw=8.0, seconds=0.4)
        both = Rig("head-eyes", look=(1700.0, 540.0))
        both.go(seconds=0.3)
        bx, _ = both.go(d_yaw=8.0, seconds=0.4)
        self.assertGreater(bx, hx + 150)                    # the same head turn got much further
        self.assertGreater(both.p.eye_pull_px, 150)

    def test_the_pull_stops_at_the_edge_of_the_gaze_error(self):
        r = Rig("head-eyes", look=(1700.0, 540.0))
        r.go(seconds=0.3)
        for _ in range(6):
            r.go(d_yaw=6.0, seconds=0.3)
            if r.point.x > 1700 - EYE_RADIUS_PX[0] - 5:
                break
        x_before = r.point.x
        self.assertGreater(x_before, 1500)
        r.go(d_yaw=0.5, seconds=0.3)                        # a small slow move: the head's own fine control
        self.assertLess(abs(r.point.x - x_before), 25)

    def test_a_resting_head_never_moves_while_the_eyes_look_around(self):
        r = Rig("head-eyes")
        r.go(seconds=0.5)
        start = (r.point.x, r.point.y)
        for spot in ((100.0, 100.0), (1800.0, 900.0), (960.0, 40.0)):   # reading the card, glancing at a corner
            r.gaze.at = spot
            r.go(seconds=0.5, noise=0.03)
        self.assertLess(math.hypot(r.point.x - start[0], r.point.y - start[1]), 3)

    def test_no_pull_when_the_head_goes_the_other_way(self):
        head, both = Rig("head"), Rig("head-eyes", look=(100.0, 540.0))   # looking left...
        for r in (head, both):
            r.go(seconds=0.3)
        hx, _ = head.go(d_yaw=6.0, seconds=0.4)                            # ...turning right
        bx, _ = both.go(d_yaw=6.0, seconds=0.4)
        self.assertAlmostEqual(hx, bx, delta=2)
        self.assertEqual(both.p.eye_pull_px, 0.0)

    def test_slow_fine_aiming_is_the_head_alone(self):
        head, both = Rig("head"), Rig("head-eyes", look=(1500.0, 540.0))
        for r in (head, both):
            r.go(seconds=0.3)
        hx, _ = head.go(d_yaw=1.0, seconds=0.5)                            # 2 deg/s
        bx, _ = both.go(d_yaw=1.0, seconds=0.5)
        self.assertAlmostEqual(hx, bx, delta=2)

    def test_no_eyes_no_pull(self):
        r = Rig("head-eyes", look=(1700.0, 540.0))
        r.go(seconds=0.3)
        r.go(d_yaw=8.0, seconds=0.4, usable=False)           # eyes unusable the whole time: the head alone
        self.assertEqual(r.p.eye_pull_px, 0.0)

    def test_head_eyes_needs_a_gaze_model(self):
        with self.assertRaises(ValueError):
            Pointer(W, H, "head-eyes", model=None)

    def test_place_carries_on_from_a_found_button(self):
        r = Rig("head-eyes")
        r.go(seconds=0.3)
        r.p.place(400, 300)
        x, y = r.go(seconds=0.3)
        self.assertEqual((round(x), round(y)), (400, 300))


class TestRestGate(unittest.TestCase):
    """A resting head's wobble no longer creeps the cursor; real moves are unchanged."""

    def drift(self, gate, noise_yaw=0.07, noise_pitch=0.16, seconds=10.0):
        random.seed(11)
        p = HeadRelPointer(W, H)
        p.rest_gate = gate
        t, xs = 0, []
        for _ in range(round(seconds * FPS)):
            t += round(1000 / FPS)
            pt = p.update(HeadPose(t, random.gauss(0, noise_yaw), random.gauss(0, noise_pitch), 0.0))
            xs.append((pt.x, pt.y))
        return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(xs, xs[1:])) / seconds   # px/s

    def test_resting_wobble_barely_moves_the_cursor(self):
        before, after = self.drift(False), self.drift(True)
        self.assertGreater(before, 5.0)
        self.assertLess(after, before / 4)

    def test_a_real_move_still_goes_the_whole_way(self):
        def move(gate):
            p = HeadRelPointer(W, H)
            p.rest_gate = gate
            t = 0
            for i in range(60):
                t += 33
                yaw = 0.0 if i < 10 else min(10.0, (i - 10) * 0.5)    # 15 deg/s for 0.66 s
                pt = p.update(HeadPose(t, yaw, 0.0, 0.0))
            return pt.x
        self.assertAlmostEqual(move(True), move(False), delta=0.03 * abs(move(False) - (W - 1) / 2))

    def test_a_sideways_move_drops_the_pitch_wobble(self):
        random.seed(2)
        p = HeadRelPointer(W, H)
        t, ys = 0, []
        for i in range(60):
            t += 33
            pt = p.update(HeadPose(t, i * 0.4, random.gauss(0, 0.16), 0.0))
            ys.append(pt.y)
        self.assertLess(max(ys) - min(ys), 6)

    def test_the_gate_has_sensible_thresholds(self):
        self.assertLess(P.REST_YAW_DEG[0], P.REST_YAW_DEG[1])
        self.assertLess(P.REST_PITCH_DEG[0], P.REST_PITCH_DEG[1])


if __name__ == "__main__":
    unittest.main()
