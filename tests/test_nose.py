"""Run with: python -m unittest tests.test_nose

The nose as a second reading of the head's turn (engine/eyes/nose.py) and the head-nose pointer
(engine/eyes/pointer.py). Made-up faces: no camera.
"""
import math
import unittest

import numpy as np

from engine.core.types import FaceFrame, HeadPose
from engine.eyes.nose import K_YAW_DEFAULT, LEARN_SAMPLES, NoseTracker
from engine.eyes.pointer import NOSE_EDGE_DEG, Pointer

W, H = 1920, 1080
K_TRUE = 0.45          # this face's real nose factor (the tracker starts from 0.55 and must learn it)


def face(t_ms: int, yaw: float, pitch: float = 0.0) -> FaceFrame:
    """A face whose nose sits where the geometry says for this yaw and pitch (yaw > 0: to the subject's right,
    which is toward image left)."""
    lm = np.zeros((478, 3), np.float32)
    lm[234] = (0.4, 0.5, 0)       # cheeks, 0.2 apart
    lm[454] = (0.6, 0.5, 0)
    lm[10] = (0.5, 0.3, 0)        # forehead and chin, 0.4 apart
    lm[152] = (0.5, 0.7, 0)
    lm[1] = (0.5 - K_TRUE * math.sin(math.radians(yaw)) * 0.2,
             0.5 + 0.35 * math.sin(math.radians(pitch)) * 0.4, 0)
    return FaceFrame(t_ms=t_ms, width=640, height=480, landmarks=lm, blendshapes={}, head_matrix=np.eye(4))


class TestNoseTracker(unittest.TestCase):
    t = 0

    def feed(self, tracker, yaws, pitch=0.0):
        out = None
        for y in yaws:
            self.t += 33                                   # time only moves forward (the filters need it)
            out = tracker.update(face(self.t, y, pitch), HeadPose(self.t, y, pitch, 0.0))
        return out

    def test_sign_and_scale_are_learned_from_the_matrix(self):
        t = NoseTracker()
        t.update(face(0, 0.0), HeadPose(0, 0.0, 0.0, 0.0))
        t.set_neutral()
        sweep = [0, 6, 9, 12, 15, 18, 15, 12, 9, 6, 0, -6, -9, -12, -15, -18, -12, -6, 0] * 3
        self.feed(t, sweep)
        self.assertTrue(t.learned)
        self.assertAlmostEqual(t.k(0), K_TRUE, delta=0.03)
        right = self.feed(t, [18] * 30)
        left = self.feed(t, [-18] * 30)
        self.assertAlmostEqual(right.yaw, 18.0, delta=1.5)
        self.assertAlmostEqual(left.yaw, -18.0, delta=1.5)

    def test_it_keeps_reading_where_the_matrix_stops(self):
        t = NoseTracker()
        self.feed(t, [0, 8, 12, 16] * 10)
        got = self.feed(t, [30] * 30)          # the nose is at 30 even if a matrix would say less
        self.assertAlmostEqual(got.yaw, 30.0, delta=2.5)

    def test_pitch_down_is_positive(self):
        t = NoseTracker()
        t.update(face(0, 0.0, 0.0), HeadPose(0, 0.0, 0.0, 0.0))
        t.set_neutral()
        down = self.feed(t, [0.0] * 30, pitch=10.0)
        self.assertGreater(down.pitch, 3.0)

    def test_no_landmarks_or_a_tiny_face_is_none(self):
        t = NoseTracker()
        self.assertIsNone(t.update(FaceFrame(0, 640, 480, np.zeros((5, 3), np.float32), {}, np.eye(4)),
                                   HeadPose(0, 0, 0, 0)))
        tiny = face(0, 0.0)
        tiny.landmarks[454] = (0.42, 0.5, 0)
        self.assertIsNone(t.update(tiny, HeadPose(0, 0, 0, 0)))

    def test_default_scale_before_it_has_learned(self):
        self.assertEqual(NoseTracker().k(0), K_YAW_DEFAULT)
        self.assertGreater(LEARN_SAMPLES, 5)


class Nose:
    def __init__(self, yaw, pitch=0.0):
        self.yaw, self.pitch = yaw, pitch


class TestHeadNosePointer(unittest.TestCase):
    def run_pointer(self, mode, yaw_to, nose_yaw=None, matrix_yaw=None, hold_s=2.0, nose=True):
        """Turn slowly to `yaw_to` over 1.5 s, then hold `hold_s`; returns the cursor x."""
        p = Pointer(W, H, mode)
        t = 0
        frames = int(1.5 * 30)
        for i in range(frames):
            yaw = yaw_to * (i + 1) / frames
            t += 33
            p.update(t, None, HeadPose(t, yaw, 0.0, 0.0), Nose(yaw if nose_yaw is None else nose_yaw) if nose else None)
        for _ in range(int(hold_s * 30)):
            t += 33
            p.update(t, None, HeadPose(t, yaw_to if matrix_yaw is None else matrix_yaw, 0.0, 0.0),
                     Nose(yaw_to if nose_yaw is None else nose_yaw) if nose else None)
        return p.rel.cursor[0], p

    def test_a_turn_held_near_the_edge_is_carried_to_it(self):
        x_head, _ = self.run_pointer("head", 17.0)
        x_nose, p = self.run_pointer("head-nose", 17.0)
        self.assertLess(x_head, 1500)                     # the plain head pointer stops short (slow turn, 10 px/deg)
        self.assertGreater(x_nose, 1700)                  # the nose carried it on toward the edge
        self.assertGreater(p.nose_assist_px, 100)

    def test_the_middle_of_the_screen_feels_exactly_like_head(self):
        x_head, _ = self.run_pointer("head", 8.0)
        x_nose, p = self.run_pointer("head-nose", 8.0)
        self.assertEqual(round(x_head, 3), round(x_nose, 3))
        self.assertEqual(p.nose_assist_px, 0.0)

    def test_without_a_nose_reading_it_is_the_head_pointer(self):
        x_head, _ = self.run_pointer("head", 17.0)
        x_nose, _ = self.run_pointer("head-nose", 17.0, nose=False)
        self.assertEqual(round(x_head, 3), round(x_nose, 3))

    def test_the_matrix_must_agree(self):
        _, p = self.run_pointer("head-nose", 8.0)          # a cursor in the middle zone, the nose then says "far"
        p._nose_t = 100.0
        before = p.rel.cursor
        point = p._nose_assist(100_033, Nose(17.0), HeadPose(100_033, 2.0, 0.0, 0.0),
                               type("P", (), {"confidence": 1.0})())
        self.assertEqual(p.rel.cursor, before)             # the matrix says the head is barely turned: no assist
        point = p._nose_assist(100_066, Nose(17.0), HeadPose(100_066, 15.0, 0.0, 0.0),
                               type("P", (), {"confidence": 1.0})())
        self.assertGreater(p.rel.cursor[0], before[0])     # the matrix agrees: assist

    def test_left_edge_too_and_never_past_it(self):
        x, _ = self.run_pointer("head-nose", -17.0, hold_s=4.0)
        self.assertLess(x, 250)
        self.assertGreaterEqual(x, 0.0)

    def test_it_never_pulls_back_toward_the_middle(self):
        x_nose, p = self.run_pointer("head-nose", 17.0, hold_s=4.0)
        before = p.rel.cursor[0]
        # the nose says "far right" but the cursor is already at the edge: nothing moves
        p.rel.cursor = (W - 1.0, p.rel.cursor[1])
        p.update(10_000, None, HeadPose(10_000, 17.0, 0.0, 0.0), Nose(17.0))
        p.update(10_033, None, HeadPose(10_033, 17.0, 0.0, 0.0), Nose(17.0))
        self.assertEqual(p.rel.cursor[0], W - 1.0)
        self.assertGreater(before, 960)

    def test_bottom_edge_with_pitch(self):
        p = Pointer(W, H, "head-nose")
        t = 0
        for i in range(45):
            t += 33
            p.update(t, None, HeadPose(t, 0.0, 10.0 * (i + 1) / 45, 0.0), Nose(0.0, 10.0 * (i + 1) / 45))
        for _ in range(90):
            t += 33
            p.update(t, None, HeadPose(t, 0.0, 10.0, 0.0), Nose(0.0, 10.0))
        self.assertGreater(p.rel.cursor[1], 900)

    def test_it_is_a_relative_mode_and_a_real_mode(self):
        from engine.eyes.pointer import MODES, REL_MODES, STILL_PX
        self.assertIn("head-nose", MODES)
        self.assertIn("head-nose", REL_MODES)
        self.assertEqual(STILL_PX["head-nose"], STILL_PX["head"])
        self.assertEqual(NOSE_EDGE_DEG, (19.0, 11.0))


if __name__ == "__main__":
    unittest.main()
