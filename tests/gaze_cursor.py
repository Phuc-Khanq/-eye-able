"""The real cursor, end to end: look to move it, nod or dwell to click. Nothing is saved.

Run:  .\\.venv\\Scripts\\python.exe -m tests.gaze_cursor
      .\\.venv\\Scripts\\python.exe -m tests.gaze_cursor --no-dwell     (nod clicks only)
First calibrate:  .\\.venv\\Scripts\\python.exe -m tests.gaze_dot --points 16

Loop: camera -> FaceMesh -> EyeFeatureExtractor + HeadPoseEstimator -> GazeModel.predict
      -> PointSmoother -> HeadPointer.update (gaze jumps, head fine-tunes) -> os_input.move
Clicks: a nod clicks where the cursor was just before the nod started; a dwell clicks after the
cursor rests 800 ms inside a 60 px circle. The terminal says which one fired.

Keys work from any window (the cursor clicks other apps, so this can't rely on window focus):
  N = look at the screen center, then press: zero the head and start the cursor
  P = pause or resume the cursor     D = dwell clicks on or off     Q = quit
Keys typed by this program itself are ignored. Hold still on the center when pressing N:
the gaze model was fitted with the head zeroed on the center dot, same as gaze_dot does it.
"""
import argparse
import math
import queue
import time
from collections import deque
from dataclasses import dataclass


from engine.core.types import EyeFeatures, GazePoint, HeadPose
from engine.eyes.head_gestures import NOD_WINDOW_MS, GestureDetector
from engine.eyes.head_pointer import HeadPointer
from engine.eyes.smoothing import DwellDetector, PointSmoother

DWELL_RADIUS_PX = 60.0
DWELL_MS = 800
NOD_START_FRACTION = 0.25  # the nod "started" once pitch left its start by this share of the threshold


@dataclass
class Click:
    x: float
    y: float
    how: str  # "nod" or "dwell"


@dataclass
class Sample:
    t_ms: int
    x: float
    y: float
    pitch: float


class GazeCursor:
    """Everything after the camera, so it runs on recorded or made-up input too.
    step() returns the cursor point to move to (or None) and a Click (or None)."""

    def __init__(self, model, screen_w: int, screen_h: int, dwell: bool = True) -> None:
        self.model = model
        self.W, self.H = screen_w, screen_h
        self.smoother = PointSmoother(min_cutoff=1.0, beta=0.5)  # in screen fractions, like gaze_dot
        self.pointer = HeadPointer(screen_w, screen_h)
        self.gestures = GestureDetector()
        self.dwell = DwellDetector(radius_px=DWELL_RADIUS_PX, dwell_ms=DWELL_MS)
        self.dwell_on = dwell
        self.dwell_armed = True       # after a dwell click, wait until the cursor moves away
        self.dwell_click_at: tuple[float, float] | None = None
        self.paused = True            # starts paused until the head is zeroed (N)
        self.history: deque[Sample] = deque()
        self.cursor: tuple[float, float] | None = None

    def set_paused(self, paused: bool) -> None:
        self.paused = paused
        self.dwell.reset()
        self.gestures.reset()

    def restart(self) -> None:
        """After the head is zeroed: drop old smoothing and anchors so nothing jumps from stale state."""
        self.smoother.reset()
        self.pointer.clear_anchor()
        self.history.clear()
        self.dwell.reset()
        self.gestures.reset()
        self.dwell_armed = True

    def _pre_nod_point(self, now_ms: int) -> tuple[float, float]:
        """Where the cursor was just before the nod began; the nod itself moves the cursor down."""
        window = [s for s in self.history if now_ms - s.t_ms <= NOD_WINDOW_MS]
        if not window:
            return self.cursor
        start = window[0].pitch
        limit = NOD_START_FRACTION * self.gestures.nod_deg
        best = window[0]
        for s in window:
            if abs(s.pitch - start) > limit:
                break
            best = s
        return best.x, best.y

    def step(self, t_ms: int, eyes: EyeFeatures | None, pose: HeadPose | None):
        if pose is None:
            return None, None
        gaze = None
        if eyes is not None and eyes.usable:
            gx, gy = self.model.predict(eyes, pose)
            sx, sy = self.smoother(t_ms / 1000.0, gx / self.W, gy / self.H)
            gaze = GazePoint(t_ms=t_ms, x=sx * self.W, y=sy * self.H, confidence=1.0)
        point = self.pointer.update(pose, gaze)
        self.cursor = (point.x, point.y)
        self.history.append(Sample(t_ms, point.x, point.y, pose.pitch))
        while self.history and t_ms - self.history[0].t_ms > 2 * NOD_WINDOW_MS:
            self.history.popleft()

        gesture = self.gestures.update(pose)
        if self.paused:
            return None, None

        click = None
        if gesture is not None and gesture.type == "nod":
            x, y = self._pre_nod_point(t_ms)
            click = Click(x, y, "nod")
            self.dwell.reset()
        elif self.dwell_on:
            if not self.dwell_armed:
                cx, cy = self.dwell_click_at
                if math.hypot(point.x - cx, point.y - cy) > DWELL_RADIUS_PX:
                    self.dwell_armed = True
                    self.dwell.reset()
            elif self.dwell.update(t_ms, point.x, point.y) >= 1.0:
                click = Click(point.x, point.y, "dwell")
                self.dwell_armed = False
                self.dwell_click_at = (point.x, point.y)
                self.dwell.reset()
        return self.cursor, click


def start_key_listener(keys: "queue.Queue[str]"):
    """Global keys, so they work while the cursor clicks other windows."""
    from pynput import keyboard

    def on_press(key, injected=False):
        if injected:  # typed by os_input, not by a person
            return
        ch = getattr(key, "char", None)
        if ch and ch.lower() in "npdq":
            keys.put(ch.lower())

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener


def main() -> None:
    parser = argparse.ArgumentParser(description="Gaze cursor: look to move, nod or dwell to click")
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--iris", choices=["center", "ring"], default="center",
                        help="use the same setting you calibrated with")
    parser.add_argument("--model", default=None, help="gaze model file (default: calibration/gaze_model.json)")
    parser.add_argument("--no-dwell", action="store_true", help="nod clicks only")
    args = parser.parse_args()

    from engine.eyes import os_input
    from engine.eyes.calibration import DEFAULT_PATH, GazeModel
    from engine.eyes.camera import Camera
    from engine.eyes.eye_features import EyeFeatureExtractor
    from engine.eyes.face_mesh import FaceMesh
    from engine.eyes.head_pose import HeadPoseEstimator

    os_input.make_dpi_aware()  # first, so screen size, model and cursor all use physical pixels
    W, H = os_input.screen_size()
    try:
        model = GazeModel.load(args.model)
    except FileNotFoundError:
        print(f"No gaze model at {args.model or DEFAULT_PATH}. Calibrate first:")
        print("  .\\.venv\\Scripts\\python.exe -m tests.gaze_dot --points 16")
        return
    if (model.screen_w, model.screen_h) != (W, H):
        print(f"Warning: the model was calibrated on a {model.screen_w}x{model.screen_h} screen, "
              f"this one is {W}x{H}. Recalibrate for good results.")

    cam = Camera(index=args.camera, fps=args.fps)
    mode = cam.start()
    print(f"Screen {W}x{H}. Camera {mode.width}x{mode.height} via {mode.backend}, camera {mode.index}.")
    mesh = FaceMesh()
    eyes = EyeFeatureExtractor(iris_mode=args.iris)
    head = HeadPoseEstimator()
    app = GazeCursor(model, W, H, dwell=not args.no_dwell)

    keys: queue.Queue[str] = queue.Queue()
    listener = start_key_listener(keys)
    print("Look at the center of the screen and press N to start. P pause, D dwell on/off, Q quit.")
    print(f"Dwell clicks {'on' if app.dwell_on else 'off'}; nod clicks on.")

    frame_id = 0
    face = None
    clicks = {"nod": 0, "dwell": 0}
    last_status = time.monotonic()
    try:
        while True:
            while not keys.empty():
                k = keys.get()
                if k == "q":
                    return
                if k == "n" and face is not None:
                    head.set_neutral(face)
                    app.restart()
                    app.set_paused(False)
                    print("Head zeroed. Cursor on.")
                elif k == "n":
                    print("No face in view; face the camera and press N again.")
                elif k == "p":
                    app.set_paused(not app.paused)
                    print("Paused." if app.paused else "Cursor on.")
                elif k == "d":
                    app.dwell_on = not app.dwell_on
                    app.dwell.reset()
                    print(f"Dwell clicks {'on' if app.dwell_on else 'off'}.")

            frame, t_ms, frame_id = cam.wait_next(frame_id)
            if frame is None:
                continue
            face = mesh.process(frame, t_ms)
            if face is None:
                continue
            feats = eyes.compute(face)
            pose = head.compute(face)
            cursor, click = app.step(face.t_ms, feats, pose)
            if cursor is not None:
                os_input.move(*cursor)
            if click is not None:
                os_input.click(int(round(click.x)), int(round(click.y)))
                clicks[click.how] += 1
                print(f"click by {click.how:<5} at ({click.x:.0f}, {click.y:.0f})")

            now = time.monotonic()
            if now - last_status > 10:
                last_status = now
                print(f"[{cam.measured_fps:.0f} fps] clicks so far: nod {clicks['nod']}, dwell {clicks['dwell']}")
    finally:
        listener.stop()
        cam.stop()
        mesh.close()
        print(f"Done. Clicks: nod {clicks['nod']}, dwell {clicks['dwell']}.")


if __name__ == "__main__":
    main()
