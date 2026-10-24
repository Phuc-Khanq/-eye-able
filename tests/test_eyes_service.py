"""The eyes in the engine, end to end with no camera, GPU or Windows: synthetic head poses go through
the real HeadPoseEstimator, Pointer, GestureDetector and ConfirmNod into the real ModeManager; UI-Venus
and os_input are fakes that record what they were asked to do.

Run with: python -m unittest tests.test_eyes_service
"""
import unittest
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from engine.core import messages, topics
from engine.core.mode_manager import Mode, ModeManager
from engine.core.types import EyeFeatures, FaceFrame, TargetResult
from engine.eyes.head_pose import HeadPoseEstimator
from engine.eyes.os_output import OsOutput
from engine.eyes.service import EyesService
from engine.eyes.target_service import TargetService
from tests.helpers import FakeBus

W, H = 1920, 1080
SETTINGS = SimpleNamespace(dwell_ms=800, sensitivity=1.0, use_ui_venus=True)


class FakeOs:
    """Stands in for engine.eyes.os_input."""

    def __init__(self):
        self.moves, self.clicks, self.typed, self.keys, self.scrolls = [], [], [], [], []

    def make_dpi_aware(self):
        pass

    def screen_size(self):
        return W, H

    def move(self, x, y):
        self.moves.append((x, y))

    def click(self, x, y):
        self.clicks.append((x, y))

    def type_text(self, text):
        self.typed.append(text)

    def combo(self, keys):
        self.keys.append(keys)

    def scroll(self, notches):
        self.scrolls.append(notches)


class FakeFinder:
    """UI-Venus stand-in: finds every label 30 px right and 10 px below the cursor point."""

    def __init__(self):
        self.asked = []

    def preload(self):
        f = Future()
        f.set_result(0.1)
        return f

    def submit(self, req):
        self.asked.append(req)
        f = Future()
        f.set_result(TargetResult(label=req.label, found=True, x=req.gaze_x + 30, y=req.gaze_y + 10))
        return f

    def close(self):
        pass


class FakeFeatures:
    def compute(self, face):
        return EyeFeatures(t_ms=face.t_ms, h=0.5, v=0.1, lid=0.3, usable=True)


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def face(t_ms, yaw=0.0, pitch=0.0, roll=0.0, brow=0.05):
    """A FaceFrame whose head matrix reads back as these angles (head_pose's sign conventions),
    with all three eyebrow scores at `brow` (0.05 = resting)."""
    m = np.eye(4)
    m[:3, :3] = Rotation.from_euler("xyz", [pitch, -yaw, roll], degrees=True).as_matrix()
    shapes = {"browInnerUp": brow, "browOuterUpLeft": brow, "browOuterUpRight": brow,
              "eyeBlinkLeft": 0.02, "eyeBlinkRight": 0.02}
    return FaceFrame(t_ms=t_ms, width=1280, height=720, landmarks=np.zeros((478, 3)), blendshapes=shapes,
                     head_matrix=m)


class Rig:
    def __init__(self, sit=(3.0, 2.0), confirm="brow"):
        self.bus, self.clock, self.os = FakeBus(), Clock(), FakeOs()
        self.mm = ModeManager(self.bus, clock=self.clock)
        self.eyes = EyesService(self.bus, SETTINGS, os_api=self.os, features=FakeFeatures(),
                                head_pose=HeadPoseEstimator(), clock=self.clock, model_path="no-such-file.json",
                                confirm=confirm)
        self.finder = FakeFinder()
        self.targets = TargetService(self.bus, SETTINGS, finder=self.finder)
        self.targets.start()
        self.out = OsOutput(self.bus, driver=self.eyes.driver, os_api=self.os)
        self.sit = sit          # raw yaw, pitch of the user's resting head (camera not straight ahead)
        self.t_ms = 0

    def frames(self, n, yaw=0.0, pitch=0.0, roll=0.0, brow=0.05):
        """n frames at 30 fps with the head this far (degrees) from where the user sits."""
        for _ in range(n):
            self.t_ms += 33
            self.clock.t += 0.033
            self.eyes.process(face(self.t_ms, self.sit[0] + yaw, self.sit[1] + pitch, roll, brow), self.clock.t)

    def path(self, pitches=(), yaws=(), roll=0.0):
        for i in range(max(len(pitches), len(yaws))):
            self.frames(1, yaw=yaws[i] if i < len(yaws) else 0.0, pitch=pitches[i] if i < len(pitches) else 0.0,
                        roll=roll)

    def say(self, text):
        self.bus.publish(topics.TRANSCRIPT, {"text": text, "words": [{"text": w, "conf": 0.95} for w in text.split()],
                                             "ms": 0.0})

    def nod(self):
        self.frames(40)                                     # still first: the resting level
        self.path(pitches=[3, 6, 8, 9, 8, 5, 2, 1, 0, 0])   # 9 degrees down and back, about 0.3 s
        self.frames(5)
        self.out.flush()

    def raise_brows(self, frames=12):
        self.frames(15)                     # resting brows
        self.frames(frames, brow=0.45)      # raised about 0.4 s (12 frames)
        self.frames(10)
        self.out.flush()

    def ui(self, kind):
        return self.bus.ui(kind)


class TestEyesService(unittest.TestCase):
    def setUp(self):
        self.r = Rig()
        self.r.frames(60)   # first face: the one-look calibration runs (about 1.5 s)

    def tearDown(self):
        self.r.out.stop()

    def test_first_face_calibrates_on_the_center(self):
        self.assertEqual(len(self.r.ui(messages.CALIB_TARGET)), 1)
        self.assertEqual(self.r.ui(messages.CALIB_RESULT)[-1]["ok"], True)
        self.r.frames(30)
        g = self.r.bus.of(topics.GAZE)[-1]
        self.assertAlmostEqual(g.x, (W - 1) / 2, delta=3)   # sitting turned 3 deg reads as the center
        self.assertAlmostEqual(g.y, (H - 1) / 2, delta=3)

    def test_publishes_the_mouth_every_frame(self):
        n = len(self.r.bus.of(topics.MOUTH))
        self.r.frames(10)
        self.assertEqual(len(self.r.bus.of(topics.MOUTH)), n + 10)

    def test_head_points(self):
        self.r.frames(60, yaw=5.0, pitch=-3.0)   # right and up
        g = self.r.bus.of(topics.GAZE)[-1]
        self.assertAlmostEqual(g.x, (W - 1) / 2 * (1 + 5 / 15), delta=12)
        self.assertAlmostEqual(g.y, (H - 1) / 2 * (1 - 3 / 10), delta=12)

    def test_click_by_name_then_raise_eyebrows(self):
        r = self.r
        r.frames(60, yaw=5.0)
        r.say("click subscribe")
        found = r.ui(messages.TARGET_FOUND)[-1]
        self.assertTrue(r.eyes.question_open and r.eyes.hold is not None)
        r.raise_brows()
        self.assertEqual(r.os.clicks, [(found["x"], found["y"])])
        self.assertIsNone(r.mm.question)
        self.assertEqual([g.type for g in r.bus.of(topics.GESTURE)], ["brow_raise"])

    def test_a_short_brow_flick_does_not_click(self):
        r = self.r
        r.say("click subscribe")
        r.raise_brows(frames=4)             # about 0.13 s: under the 250 ms hold
        self.assertEqual(r.os.clicks, [])

    def test_raise_with_nothing_waiting_does_nothing(self):
        r = self.r
        r.raise_brows()
        self.assertEqual(r.os.clicks, [])
        self.assertEqual(r.os.typed, [])

    def test_nod_does_not_say_yes_by_default(self):
        r = self.r
        r.say("click subscribe")
        r.nod()
        self.assertEqual(r.os.clicks, [])
        self.assertTrue(r.eyes.question_open)

    def test_click_by_name_then_nod_with_confirm_nod(self):
        self.r.out.stop()
        self.r = r = Rig(confirm="nod")
        r.frames(60)
        r.frames(60, yaw=5.0)
        r.say("click subscribe")
        req = r.finder.asked[-1]
        self.assertEqual(req.label.lower(), "subscribe")
        self.assertAlmostEqual(req.gaze_x, 1279, delta=12)
        found = r.ui(messages.TARGET_FOUND)[-1]
        self.assertTrue(r.eyes.question_open and r.eyes.hold is not None)
        self.assertEqual(r.os.moves[-1], (found["x"], found["y"]))   # the cursor jumped onto it
        r.nod()
        self.assertEqual(r.os.clicks, [(found["x"], found["y"])])
        self.assertIsNone(r.eyes.hold)
        self.assertIsNone(r.mm.question)

    def test_shake_cancels_the_click(self):
        r = self.r
        r.say("click like button")
        self.assertTrue(r.eyes.question_open)
        r.frames(10)
        r.path(yaws=[0, 5, 10, 13, 10, 5, 0, -5, -10, -13, -10, -5, 0, 5, 10, 13, 10, 5, 0])
        r.frames(10)
        r.out.flush()
        self.assertEqual(r.os.clicks, [])
        self.assertIsNone(r.mm.question)
        self.assertIsNone(r.eyes.hold)

    def test_dictation_brow_raise_types_the_preview(self):
        r = self.r
        r.say("start dictation")
        self.assertEqual(r.mm.mode, Mode.DICTATION)
        r.say("hello world")
        self.assertTrue(r.eyes.preview_waiting)
        r.raise_brows()
        self.assertEqual(r.os.typed, ["hello world "])
        self.assertFalse(r.eyes.preview_waiting)

    def test_a_small_nod_while_reading_does_nothing(self):
        r = self.r
        r.frames(40)
        r.path(pitches=[3, 6, 8, 9, 8, 5, 2, 1, 0, 0])   # nothing is waiting for a yes
        r.frames(10)
        self.assertEqual(r.bus.of(topics.GESTURE), [])

    def test_tilt_scrolls(self):
        r = self.r
        r.frames(20, roll=24.0)   # held about 0.66 s
        r.frames(20)
        r.out.flush()
        self.assertEqual(r.os.scrolls, [-5])

    def test_pause_leaves_the_cursor_alone(self):
        r = self.r
        r.say("pause")
        self.assertEqual(r.eyes.mode, "paused")
        n = len(r.os.moves)
        r.eyes.driver.snap(10, 10)
        self.assertEqual(len(r.os.moves), n)
        r.say("resume")
        self.assertTrue(r.eyes.driver.enabled)

    def test_calibrate_again_by_voice(self):
        r = self.r
        r.sit = (8.0, -1.0)      # the user moved
        r.say("calibrate")
        r.frames(60)
        self.assertEqual(len(r.ui(messages.CALIB_RESULT)), 2)
        r.frames(30)
        g = r.bus.of(topics.GAZE)[-1]
        self.assertAlmostEqual(g.x, (W - 1) / 2, delta=3)

    def test_ui_venus_off_answers_not_found(self):
        bus = FakeBus()
        svc = TargetService(bus, SimpleNamespace(use_ui_venus=False), finder=FakeFinder())
        svc.start()
        from engine.core.types import TargetRequest
        bus.publish(topics.TARGET_REQUEST, TargetRequest(10, 10, "Send"))
        self.assertFalse(bus.of(topics.TARGET_RESULT)[-1].found)


if __name__ == "__main__":
    unittest.main()
