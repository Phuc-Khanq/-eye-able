"""Calibrate, check on new dots, then watch a dot follow your gaze. Only calibration numbers are saved.

Run:  .\\.venv\\Scripts\\python.exe -m tests.gaze_dot
Options:  --fps 60   --points 16   --iris ring   --no-head   --camera N
Keys:  SPACE = start / next   R = redo calibration   M = move the real cursor on/off
       B = show the 1280x720 area UI-Venus would search   Q or Esc = quit

Steps
  1. Sit as you normally would, about an arm's length away. SPACE starts.
  2. Look at each dot until it turns green and disappears. Keep your head fairly still.
  3. Results show the leave-one-dot-out error. SPACE checks the model on 5 new dots.
  4. The check shows accuracy (how far off, on average) and wobble (how much it shakes). SPACE goes live.
  5. Live: the blue dot is your gaze. B shows the box UI-Venus would search around it.
The numbers are also printed in the terminal, so you can paste them.
"""
import argparse
import math
import time

import cv2
import numpy as np

from engine.config import settings
from engine.eyes.calibration import GazeModel, feature_row, grid_targets
from engine.eyes.smoothing import PointSmoother

SETTLE_S = 0.7      # look time before collecting, while the dot shrinks
COLLECT_S = 1.0     # collect this long once settled
MAX_S = 3.5         # give up on a dot after this long (blinks, no face)
MIN_SAMPLES = 12    # usable frames needed per dot
VALIDATION_DOTS = 5

BG = (24, 22, 20)
TEXT = (235, 235, 235)
QUIET = (150, 150, 150)
GOOD = (120, 210, 120)
WARN = (60, 170, 240)
BAD = (90, 90, 230)
DOT = (240, 240, 240)
GAZE = (250, 170, 60)
PRED = (60, 160, 250)
FONT = cv2.FONT_HERSHEY_SIMPLEX


class GazeDotApp:
    """The whole flow as a state machine, so it can run with a real camera or with fake input."""

    def __init__(self, screen_w: int, screen_h: int, points: int = 9, seed: int | None = None) -> None:
        self.W, self.H = screen_w, screen_h
        self.points = points
        self.rng = np.random.default_rng(seed)
        self.model = GazeModel(screen_w, screen_h)
        self.smoother = PointSmoother(min_cutoff=1.0, beta=0.5)  # in screen fractions
        self.state = "ready"
        self.move_cursor = False
        self.show_box = False
        self.messages: list[str] = []
        self.reset_calibration()

    # -- flow -------------------------------------------------------------------------------
    def reset_calibration(self) -> None:
        self.state = "ready"
        self.targets = grid_targets(self.W, self.H, self.points)
        self.i = 0
        self.dot_start = None
        self.rows: list[list[np.ndarray]] = [[] for _ in self.targets]
        self.need_neutral = True
        self.report = None
        self.val_targets: list[tuple[float, float]] = []
        self.val_preds: list[list[tuple[float, float]]] = []
        self.val_result = None
        self.gaze = None

    def start(self, now: float) -> None:
        if self.state == "ready":
            self.state, self.i, self.dot_start = "calib", 0, now
        elif self.state == "results" and self.model.ready:
            m = 0.15
            self.val_targets = [(float(self.rng.uniform(m, 1 - m) * self.W), float(self.rng.uniform(m, 1 - m) * self.H))
                                for _ in range(VALIDATION_DOTS)]
            self.val_preds = [[] for _ in self.val_targets]
            self.state, self.i, self.dot_start = "validate", 0, now
        elif self.state == "val_results":
            self.smoother.reset()
            self.state = "live"

    def _dot_phase(self, now: float, count: int) -> str:
        """'settle', 'collect' or 'done' for the current dot."""
        t = now - self.dot_start
        if t < SETTLE_S:
            return "settle"
        if (t >= SETTLE_S + COLLECT_S and count >= MIN_SAMPLES) or t >= MAX_S:
            return "done"
        return "collect"

    def step(self, now: float, row: np.ndarray | None, on_center_collect=None) -> None:
        """Feed one camera frame. row is the feature row, or None for no face or a blink."""
        if self.state == "calib":
            phase = self._dot_phase(now, len(self.rows[self.i]))
            if phase == "collect" and row is not None:
                if self.i == 0 and self.need_neutral and on_center_collect is not None:
                    row = on_center_collect()  # zero the head pose while looking at the middle dot
                    self.need_neutral = False
                if row is not None:
                    self.rows[self.i].append(row)
            elif phase == "done":
                self.i += 1
                self.dot_start = now
                if self.i == len(self.targets):
                    self._fit()
        elif self.state == "validate":
            phase = self._dot_phase(now, len(self.val_preds[self.i]))
            if phase == "collect" and row is not None:
                self.val_preds[self.i].append(self.model.predict_row(row))
            elif phase == "done":
                self.i += 1
                self.dot_start = now
                if self.i == len(self.val_targets):
                    self._score_validation()
        elif self.state == "live" and row is not None:
            x, y = self.model.predict_row(row)
            sx, sy = self.smoother(now, x / self.W, y / self.H)
            self.gaze = (sx * self.W, sy * self.H)

    def _fit(self) -> None:
        dots = [(t, np.array(r)) for t, r in zip(self.targets, self.rows) if len(r) >= MIN_SAMPLES // 2]
        try:
            self.report = self.model.fit(dots)
            path = self.model.save()
            self.messages = [f"Calibration: {self.report.summary()}", f"Saved to {path}"]
        except ValueError as e:
            self.report = None
            self.messages = [f"Calibration failed: {e}. Press R to try again (more light, face the camera)."]
        for m in self.messages:
            print(m)
        self.state = "results"

    def _score_validation(self) -> None:
        offsets, wobbles = [], []
        for t, preds in zip(self.val_targets, self.val_preds):
            if len(preds) < MIN_SAMPLES // 2:
                continue
            p = np.array(preds)
            med = np.median(p, axis=0)
            offsets.append(math.hypot(med[0] - t[0], med[1] - t[1]))
            wobbles.append(float(np.sqrt(np.mean(np.sum((p - med) ** 2, axis=1)))))
        if offsets:
            self.val_result = (float(np.mean(offsets)), float(np.mean(wobbles)), len(offsets))
            print(f"Check on {len(offsets)} new dots: accuracy {self.val_result[0]:.0f} px (average offset), "
                  f"wobble {self.val_result[1]:.0f} px (frame-to-frame, before smoothing)")
        else:
            self.val_result = None
            print("Check failed: no usable frames on the new dots.")
        self.state = "val_results"

    # -- drawing ----------------------------------------------------------------------------
    def draw(self, now: float, face_ok: bool, fps: float, canvas_scale: float) -> np.ndarray:
        cw, ch = int(round(self.W * canvas_scale)), int(round(self.H * canvas_scale))
        img = np.full((ch, cw, 3), BG, dtype=np.uint8)
        s = canvas_scale
        fs = 0.75 * cw / 1920  # font scale relative to a 1920-wide canvas

        def P(x, y):
            return int(round(x * s)), int(round(y * s))

        # Text sits in the empty band between the dot rows (dots are at 10%, 50% and 90% height).
        top = int(0.2 * ch) if self.state in ("results", "val_results") else int(60 * cw / 1920)

        def say(txt, row, color=TEXT, big=False, y0=None):
            y = (top if y0 is None else y0) + int(row * 40 * cw / 1920)
            cv2.putText(img, txt, (int(40 * cw / 1920), y), FONT,
                        fs * (1.3 if big else 1.0), color, 2 if big else 1, cv2.LINE_AA)

        if self.state == "ready":
            say("Gaze calibration", 0, TEXT, big=True)
            say(f"{len(self.targets)} dots. Look at each one until it turns green and disappears.", 2)
            say("Keep your head fairly still. Blinking is fine.", 3)
            say("SPACE to start   Q to quit", 5, QUIET)
        elif self.state in ("calib", "validate"):
            targets = self.targets if self.state == "calib" else self.val_targets
            counts = self.rows if self.state == "calib" else self.val_preds
            if self.i < len(targets):
                t = now - self.dot_start
                x, y = targets[self.i]
                if t < SETTLE_S:
                    r = int((34 - 22 * t / SETTLE_S) * cw / 1920)
                    cv2.circle(img, P(x, y), r, DOT, -1, cv2.LINE_AA)
                else:
                    cv2.circle(img, P(x, y), int(12 * cw / 1920), GOOD, -1, cv2.LINE_AA)
                cv2.circle(img, P(x, y), max(2, int(3 * cw / 1920)), BG, -1, cv2.LINE_AA)
                label = "Calibrating" if self.state == "calib" else "Checking on new dots"
                say(f"{label}: dot {self.i + 1} of {len(targets)}   samples {len(counts[self.i])}", 0, QUIET,
                    y0=ch - int(70 * cw / 1920))
        elif self.state == "results":
            if self.report:
                for d in self.report.dots:
                    cv2.circle(img, P(*d.target), int(14 * cw / 1920), DOT, 2, cv2.LINE_AA)
                    cv2.line(img, P(*d.target), P(*d.predicted), PRED, 2, cv2.LINE_AA)
                    cv2.circle(img, P(*d.predicted), int(7 * cw / 1920), PRED, -1, cv2.LINE_AA)
                say("Calibration done", 0, GOOD, big=True)
                say(self.report.summary(), 1)
                say("White ring = dot, orange = where the model would put it without having seen that dot.", 2, QUIET)
                say("SPACE: check on 5 new dots   R: redo   Q: quit", 3, QUIET)
            else:
                for m in self.messages:
                    say(m, 0, BAD)
                say("R: redo   Q: quit", 2, QUIET)
        elif self.state == "val_results":
            for t, preds in zip(self.val_targets, self.val_preds):
                cv2.circle(img, P(*t), int(14 * cw / 1920), DOT, 2, cv2.LINE_AA)
                if preds:
                    med = np.median(np.array(preds), axis=0)
                    cv2.line(img, P(*t), P(*med), GAZE, 2, cv2.LINE_AA)
                    cv2.circle(img, P(*med), int(7 * cw / 1920), GAZE, -1, cv2.LINE_AA)
            if self.val_result:
                acc, wob, n = self.val_result
                say("Check on new dots", 0, GOOD, big=True)
                say(f"Accuracy {acc:.0f} px (average offset)   wobble {wob:.0f} px (before smoothing)   {n} dots", 1)
                say("White ring = new dot, blue = where your gaze landed on average.", 2, QUIET)
            else:
                say("Check failed: no usable frames. R to recalibrate.", 0, BAD, big=True)
            say("SPACE: go live   R: recalibrate   Q: quit", 3, QUIET)
        elif self.state == "live":
            if self.gaze is not None:
                gx, gy = self.gaze
                if self.show_box:
                    bw, bh = settings.crop_width, settings.crop_height
                    left = min(max(gx - bw / 2, 0), max(self.W - bw, 0))
                    top_ = min(max(gy - bh / 2, 0), max(self.H - bh, 0))
                    cv2.rectangle(img, P(left, top_), P(left + bw, top_ + bh), WARN, 2, cv2.LINE_AA)
                if self.val_result:
                    cv2.circle(img, P(gx, gy), max(4, int(self.val_result[0] * s)), (90, 80, 70), 2, cv2.LINE_AA)
                cv2.circle(img, P(gx, gy), int(16 * cw / 1920), GAZE, -1, cv2.LINE_AA)
            say("Live gaze   M move real cursor: " + ("ON" if self.move_cursor else "off") +
                "   B UI-Venus box   R recalibrate   Q quit", 0, QUIET)
            if self.val_result:
                say(f"Ring = average offset from the check ({self.val_result[0]:.0f} px)", 1, QUIET)

        status = f"camera {fps:.0f} fps" + ("" if face_ok else "   NO FACE: face the camera")
        cv2.putText(img, status, (int(40 * cw / 1920), ch - int(30 * cw / 1920)), FONT, fs * 0.8,
                    QUIET if face_ok else BAD, 1, cv2.LINE_AA)
        return img


def main() -> None:
    parser = argparse.ArgumentParser(description="Gaze calibration and live gaze dot")
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--points", type=int, default=9, help="9 (3x3) or 16 (4x4)")
    parser.add_argument("--iris", choices=["center", "ring"], default="center")
    parser.add_argument("--no-head", action="store_true", help="leave head pose out of the model")
    args = parser.parse_args()

    from engine.eyes import os_input
    from engine.eyes.camera import Camera
    from engine.eyes.eye_features import EyeFeatureExtractor
    from engine.eyes.face_mesh import FaceMesh
    from engine.eyes.head_pose import HeadPoseEstimator

    os_input.make_dpi_aware()               # physical pixels, same as the cursor
    W, H = os_input.screen_size()
    canvas_scale = min(1.0, 1920 / W)       # draw at most 1920 wide; the window stretches it
    print(f"Screen {W}x{H}. Head pose {'off' if args.no_head else 'on'}, iris {args.iris}.")

    cam = Camera(index=args.camera, fps=args.fps)
    mode = cam.start()
    print(f"Camera: {mode.width}x{mode.height} {mode.fourcc} via {mode.backend}, camera {mode.index}")
    mesh = FaceMesh()
    eyes = EyeFeatureExtractor(iris_mode=args.iris)
    head = HeadPoseEstimator()
    app = GazeDotApp(W, H, points=args.points)

    window = "Diseyeble gaze"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window, 0, 0)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    frame_id = 0
    face = None
    try:
        while True:
            frame, t_ms, frame_id = cam.wait_next(frame_id)
            now = time.monotonic()
            row = None
            if frame is not None:
                face = mesh.process(frame, t_ms)
                if face is not None:
                    feats = eyes.compute(face)
                    pose = None if args.no_head else head.compute(face)
                    if feats.usable:
                        row = feature_row(feats, pose)

            def zero_head():
                if args.no_head or face is None:
                    return row
                head.set_neutral(face)
                return feature_row(eyes.last, head.compute(face)) if eyes.last else row

            app.step(now, row, on_center_collect=zero_head)
            if app.state == "live" and app.move_cursor and app.gaze is not None:
                os_input.move(*app.gaze)
            cv2.imshow(window, app.draw(now, face is not None, cam.measured_fps, canvas_scale))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                app.start(now)
            elif key == ord("r"):
                head.reset_neutral()
                app.reset_calibration()
            elif key == ord("m"):
                app.move_cursor = not app.move_cursor
            elif key == ord("b"):
                app.show_box = not app.show_box
    finally:
        cam.stop()
        mesh.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
