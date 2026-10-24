"""Calibrate (or re-center a trained model), check on new dots, then watch a dot follow your gaze.
Every run saves its numbers (no images) to calibration/sessions/<profile>/ and retrains the model on
all of that profile's sessions, so it gets better the more you use it.

Run:  .\\.venv\\Scripts\\python.exe -m tests.gaze_dot --profile khang            full: 25 dots (about 1 min)
      .\\.venv\\Scripts\\python.exe -m tests.gaze_dot --profile khang --quick    quick: 5 dots (about 6 s)
Other options:  --points 16   --sweep   --y-inputs eye|eye+mediapipe|mediapipe|full   --fps 60   --no-head
Keys:  SPACE = start / next   R = redo   C = re-center (live)   M = move the real cursor on/off
       B = show the 1280x720 area UI-Venus would search   Q or Esc = quit

A profile is one person on one setup. Use your own name, and a new name for a different screen or
camera position (data from another screen size is skipped automatically).

Full run
  1. Sit as you normally would, about an arm's length away. SPACE starts.
  2. Look at each dot until it turns green and disappears. Keep your head fairly still.
  3. Results. One session: leave-one-dot-out error. More sessions: the new-session test (train on the
     other sessions, re-center on one dot, measure the rest), next to "this session alone".
  4. SPACE: look at the middle dot (re-center), then 5 new dots. Shows accuracy with and without re-center.
  5. SPACE: live. The blue dot is your gaze. C re-centers on the middle dot if it has drifted.
Quick run: 5 re-center dots, then the same check and live view, using the trained model.
"""
import argparse
import math
import time

import cv2
import numpy as np

from engine.config import settings
from engine.eyes.calibration import GazeModel, feature_row, grid_targets
from engine.eyes.gaze_data import Session, load_sessions, save_session
from engine.eyes.smoothing import PointSmoother

SETTLE_S = 0.7      # look time before collecting, while the dot shrinks
COLLECT_S = 1.0     # collect this long once settled
MAX_S = 3.5         # give up on a dot after this long (blinks, no face)
MIN_SAMPLES = 12    # usable frames needed per dot
VALIDATION_DOTS = 5
MAX_ONE_DOT_DRIFT_PX = 300  # a one-dot re-center claiming more than this is ignored
QUICK_DOTS = [(0.5, 0.5), (0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)]  # quick re-center
SWEEP_S = 6.0       # optional head-sweep step: look at the middle dot, turn and nod a little
SWEEP_SETTLE_S = 1.0
SWEEP_CHUNK_S = 0.5

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

    def __init__(self, screen_w: int, screen_h: int, points: int = 25, seed: int | None = None,
                 y_inputs: str = "auto", sweep: bool = False, profile: str = "default",
                 trained: GazeModel | None = None, camera: str = "", iris_mode: str = "center",
                 save: bool = True) -> None:
        self.W, self.H = screen_w, screen_h
        self.points, self.y_inputs, self.sweep = points, y_inputs, sweep
        self.profile, self.camera, self.iris_mode, self.save = profile, camera, iris_mode, save
        self.quick = trained is not None
        self.rng = np.random.default_rng(seed)
        self.model = trained or GazeModel(screen_w, screen_h)
        self.smoother = PointSmoother(min_cutoff=1.0, beta=0.5)  # in screen fractions
        self.move_cursor = False
        self.show_box = False
        self.reset()

    # -- flow -------------------------------------------------------------------------------
    def reset(self) -> None:
        self.state = "ready"
        self.targets = grid_targets(self.W, self.H, self.points)
        self.i = 0
        self.dot_start = None
        self.rows: list[list[np.ndarray]] = [[] for _ in self.targets]
        self.sweep_rows: list[tuple[float, np.ndarray]] = []
        self.need_neutral = True
        self.report = None
        self.messages: list[str] = []
        self.rc_targets: list[tuple[float, float]] = []
        self.rc_rows: list[list[np.ndarray]] = []
        self.after_rc = "validate"
        self.bias_before = self.model.bias
        self.drift = (0.0, 0.0)
        self.val_targets: list[tuple[float, float]] = []
        self.val_rows: list[list[np.ndarray]] = []
        self.val_result = None
        self.gaze = None
        self.session = Session(profile=self.profile, screen=(self.W, self.H), camera=self.camera,
                               iris_mode=self.iris_mode, kind="quick" if self.quick else "calibration")

    def _begin_recenter(self, now: float, points, then: str) -> None:
        self.rc_targets = [(fx * self.W, fy * self.H) for fx, fy in points]
        self.rc_rows = [[] for _ in self.rc_targets]
        self.after_rc, self.bias_before = then, self.model.bias
        self.state, self.i, self.dot_start = "recenter", 0, now

    def start(self, now: float) -> None:
        if self.state == "ready":
            if self.quick:
                self._begin_recenter(now, QUICK_DOTS, "validate")
            else:
                self.state, self.i, self.dot_start = "calib", 0, now
        elif self.state == "results" and self.model.ready:
            self._begin_recenter(now, QUICK_DOTS[:1], "validate")
        elif self.state == "val_results":
            self.smoother.reset()
            self.state = "live"

    def recenter_live(self, now: float) -> None:
        if self.state == "live":
            self._begin_recenter(now, QUICK_DOTS[:1], "live")

    def _new_val_targets(self) -> None:
        m = 0.15
        self.val_targets = [(float(self.rng.uniform(m, 1 - m) * self.W), float(self.rng.uniform(m, 1 - m) * self.H))
                            for _ in range(VALIDATION_DOTS)]
        self.val_rows = [[] for _ in self.val_targets]

    def _dot_phase(self, now: float, count: int) -> str:
        """'settle', 'collect' or 'done' for the current dot."""
        t = now - self.dot_start
        if t < SETTLE_S:
            return "settle"
        if (t >= SETTLE_S + COLLECT_S and count >= MIN_SAMPLES) or t >= MAX_S:
            return "done"
        return "collect"

    def step(self, now: float, row: np.ndarray | None, on_center_collect=None) -> None:
        """Feed one camera frame. row is the feature row, or None for no face or a blink.
        on_center_collect: called once at the first middle dot to zero the head pose; returns a new row."""
        if self.state in ("calib", "recenter") and self.need_neutral and on_center_collect is not None \
                and row is not None and self.i == 0 and self._dot_phase(now, 0) == "collect":
            row = on_center_collect()
            self.need_neutral = False
        if self.state == "calib":
            phase = self._dot_phase(now, len(self.rows[self.i]))
            if phase == "collect" and row is not None:
                self.rows[self.i].append(row)
            elif phase == "done":
                self.i += 1
                self.dot_start = now
                if self.i == len(self.targets):
                    if self.sweep:
                        self.state = "sweep"
                    else:
                        self._fit()
        elif self.state == "sweep":
            t = now - self.dot_start
            if SWEEP_SETTLE_S <= t < SWEEP_SETTLE_S + SWEEP_S and row is not None:
                self.sweep_rows.append((t, row))
            elif t >= SWEEP_SETTLE_S + SWEEP_S:
                self._fit()
        elif self.state == "recenter":
            phase = self._dot_phase(now, len(self.rc_rows[self.i]))
            if phase == "collect" and row is not None:
                self.rc_rows[self.i].append(row)
            elif phase == "done":
                self.i += 1
                self.dot_start = now
                if self.i == len(self.rc_targets):
                    self._finish_recenter(now)
        elif self.state == "validate":
            phase = self._dot_phase(now, len(self.val_rows[self.i]))
            if phase == "collect" and row is not None:
                self.val_rows[self.i].append(row)
            elif phase == "done":
                self.i += 1
                self.dot_start = now
                if self.i == len(self.val_targets):
                    self._score_validation()
        elif self.state == "live" and row is not None:
            x, y = self.model.predict_row(row)
            sx, sy = self.smoother(now, x / self.W, y / self.H)
            self.gaze = (sx * self.W, sy * self.H)

    def _finish_recenter(self, now: float) -> None:
        dots = [(t, np.array(r)) for t, r in zip(self.rc_targets, self.rc_rows) if len(r) >= MIN_SAMPLES // 2]
        before = self.model.bias
        self.drift = self.model.recenter(dots)
        print(f"Re-center on {len(dots)} dot(s): gaze had drifted ({self.drift[0]:+.0f}, {self.drift[1]:+.0f}) px")
        if len(dots) == 1 and math.hypot(*self.drift) > MAX_ONE_DOT_DRIFT_PX:
            # One dot right after calibration can't honestly have moved this far: a blink or a glance away.
            self.model.bias = before
            self.drift = (0.0, 0.0)
            print(f"  ignored: more than {MAX_ONE_DOT_DRIFT_PX} px from one dot usually means you weren't on the dot")
            return self._after_recenter(now)
        for t, r in dots:
            self.session.add(t, r, "recenter")
        self._after_recenter(now)

    def _after_recenter(self, now: float) -> None:
        if self.after_rc == "live":
            self.smoother.reset()
            self.state = "live"
        else:
            self._new_val_targets()
            self.state, self.i, self.dot_start = "validate", 0, now

    def _train(self) -> None:
        """Saves this session and retrains on every session of the profile (same screen size)."""
        if self.save:
            save_session(self.session)
            others, notes = load_sessions(self.profile, (self.W, self.H))
            others = [s for s in others if s.path != self.session.path]
            for n in notes:
                print("  " + n)
        else:
            others = []
        self.report = self.model.fit_sessions(others + [self.session], current=len(others), y_inputs=self.y_inputs)
        if self.save:
            self.messages.append(f"Model saved to {self.model.save()}")

    def _fit(self) -> None:
        self.messages = []
        for t, r in zip(self.targets, self.rows):
            if len(r) >= MIN_SAMPLES // 2:
                self.session.add(t, r, "calib")
        chunks: dict[int, list[np.ndarray]] = {}
        for t, r in self.sweep_rows:
            chunks.setdefault(int((t - SWEEP_SETTLE_S) // SWEEP_CHUNK_S), []).append(r)
        for c in chunks.values():
            if len(c) >= 5:
                self.session.add((self.W / 2, self.H / 2), c, "sweep")
        try:
            self._train()
            self.messages[:0] = [f"Calibration: {self.report.summary()}", self.report.compare()]
        except ValueError as e:
            self.report = None
            self.messages = [f"Calibration failed: {e}. Press R to try again (more light, face the camera)."]
        for m in self.messages:
            print(m)
        self.state = "results"

    def _score_validation(self) -> None:
        offsets, raw_offsets, wobbles, self.val_meds = [], [], [], []
        for t, rows in zip(self.val_targets, self.val_rows):
            if len(rows) < MIN_SAMPLES // 2:
                self.val_meds.append(None)
                continue
            med_row = np.median(np.array(rows), axis=0)
            p = np.array(self.model.predict_row(med_row))
            q = np.array(self.model.predict_row(med_row, bias=self.bias_before))
            frames = np.array([self.model.predict_row(r) for r in rows])
            self.val_meds.append(p)
            offsets.append(math.hypot(p[0] - t[0], p[1] - t[1]))
            raw_offsets.append(math.hypot(q[0] - t[0], q[1] - t[1]))
            wobbles.append(float(np.sqrt(np.mean(np.sum((frames - np.median(frames, axis=0)) ** 2, axis=1)))))
            self.session.add(t, rows, "check")
        if offsets:
            self.val_result = (float(np.mean(offsets)), float(np.mean(wobbles)), len(offsets), float(np.mean(raw_offsets)))
            what = "the trained model's average shift" if self.quick else "the calibration"
            print(f"Check on {len(offsets)} new dots: accuracy {self.val_result[0]:.0f} px after re-center "
                  f"({self.val_result[3]:.0f} px with {what}), wobble {self.val_result[1]:.0f} px (frame-to-frame)")
            try:  # the check dots are new training data too
                self._train()
                print(f"Retrained on {self.report.sessions} session(s), {self.report.dots_total} dots: {self.report.summary()}")
            except ValueError as e:
                print(f"Not retrained: {e}")
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

        # Result text fits in the empty band under the top dot row (rows at 10% and, with 25 dots, 30% height).
        top = int(0.145 * ch) if self.state in ("results", "val_results") else int(60 * cw / 1920)

        def say(txt, row, color=TEXT, big=False, y0=None):
            y = (top if y0 is None else y0) + int(row * 34 * cw / 1920)
            x = int(40 * cw / 1920)
            scale, thick = fs * (1.3 if big else 1.0), 2 if big else 1
            (tw, th), base = cv2.getTextSize(txt, FONT, scale, thick)
            pad = int(6 * cw / 1920)
            cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + base + pad), BG, -1)  # readable over dots
            cv2.putText(img, txt, (x, y), FONT, scale, color, thick, cv2.LINE_AA)

        def dot(x, y, t):
            if t < SETTLE_S:
                cv2.circle(img, P(x, y), int((34 - 22 * t / SETTLE_S) * cw / 1920), DOT, -1, cv2.LINE_AA)
            else:
                cv2.circle(img, P(x, y), int(12 * cw / 1920), GOOD, -1, cv2.LINE_AA)
            cv2.circle(img, P(x, y), max(2, int(3 * cw / 1920)), BG, -1, cv2.LINE_AA)

        bottom = ch - int(70 * cw / 1920)
        if self.state == "ready":
            if self.quick:
                say("Quick start with the trained model", 0, TEXT, big=True)
                say(f"Profile '{self.profile}', trained on {self.model.sessions} session(s). Look at 5 dots to re-center.", 2)
            else:
                say("Gaze calibration", 0, TEXT, big=True)
                say(f"Profile '{self.profile}'. {len(self.targets)} dots: look at each until it turns green and disappears.", 2)
            say("Keep your head fairly still. Blinking is fine.", 3)
            say("SPACE to start   Q to quit", 5, QUIET)
        elif self.state in ("calib", "validate", "recenter"):
            targets = {"calib": self.targets, "validate": self.val_targets, "recenter": self.rc_targets}[self.state]
            counts = {"calib": self.rows, "validate": self.val_rows, "recenter": self.rc_rows}[self.state]
            if self.i < len(targets):
                dot(*targets[self.i], now - self.dot_start)
                label = {"calib": "Calibrating", "validate": "Checking on new dots", "recenter": "Re-centering"}[self.state]
                say(f"{label}: dot {self.i + 1} of {len(targets)}   samples {len(counts[self.i])}", 0, QUIET, y0=bottom)
        elif self.state == "sweep":
            t = now - self.dot_start
            dot(self.W / 2, self.H / 2, SETTLE_S)
            say("Last step: keep looking at the green dot", 0, GOOD, big=True)
            say("Slowly turn your head a little left and right, then nod a little. Eyes stay on the dot.", 1)
            say(f"{max(0.0, SWEEP_SETTLE_S + SWEEP_S - t):.0f} s left   samples {len(self.sweep_rows)}", 2, QUIET)
        elif self.state == "results":
            if self.report:
                for d in self.report.dots:
                    cv2.circle(img, P(*d.target), int(14 * cw / 1920), DOT, 2, cv2.LINE_AA)
                    cv2.line(img, P(*d.target), P(*d.predicted), PRED, 2, cv2.LINE_AA)
                    cv2.circle(img, P(*d.predicted), int(7 * cw / 1920), PRED, -1, cv2.LINE_AA)
                say("Calibration done", 0, GOOD, big=True)
                say(self.report.summary(), 1)
                say(self.report.compare(), 2)
                say("White ring = dot, orange = where the model put it without having seen that dot.", 3, QUIET)
                say("SPACE: re-center + check on 5 new dots   R: redo   Q: quit", 4, QUIET)
            else:
                for k, m in enumerate(self.messages):
                    say(m, k, BAD)
                say("R: redo   Q: quit", 2, QUIET)
        elif self.state == "val_results":
            for t, med in zip(self.val_targets, getattr(self, "val_meds", [])):
                cv2.circle(img, P(*t), int(14 * cw / 1920), DOT, 2, cv2.LINE_AA)
                if med is not None:
                    cv2.line(img, P(*t), P(*med), GAZE, 2, cv2.LINE_AA)
                    cv2.circle(img, P(*med), int(7 * cw / 1920), GAZE, -1, cv2.LINE_AA)
            if self.val_result:
                acc, wob, n, raw = self.val_result
                say("Check on new dots", 0, GOOD, big=True)
                say(f"Accuracy {acc:.0f} px after re-center ({raw:.0f} px without)   wobble {wob:.0f} px   {n} dots", 1)
                say("White ring = new dot, blue = where your gaze landed on average. Saved as training data.", 2, QUIET)
            else:
                say("Check failed: no usable frames. R to start over.", 0, BAD, big=True)
            say("SPACE: go live   R: start over   Q: quit", 3, QUIET)
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
                "   C re-center   B UI-Venus box   R start over   Q quit", 0, QUIET)
            if self.val_result:
                say(f"Ring = average offset from the check ({self.val_result[0]:.0f} px)", 1, QUIET)

        status = f"camera {fps:.0f} fps" + ("" if face_ok else "   NO FACE: face the camera")
        cv2.putText(img, status, (int(40 * cw / 1920), ch - int(30 * cw / 1920)), FONT, fs * 0.8,
                    QUIET if face_ok else BAD, 1, cv2.LINE_AA)
        return img


def main() -> None:
    parser = argparse.ArgumentParser(description="Gaze calibration, check and live gaze dot")
    parser.add_argument("--profile", default="default", help="your name; one profile per person and setup")
    parser.add_argument("--quick", action="store_true", help="use the trained model, re-center on 5 dots")
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--points", type=int, default=25, help="9 (3x3), 16 (4x4) or 25 (5x5)")
    parser.add_argument("--y-inputs", default="auto", choices=["auto", "eye", "eye+mediapipe", "mediapipe", "full"])
    parser.add_argument("--iris", choices=["center", "ring"], default="center")
    parser.add_argument("--no-head", action="store_true", help="leave head pose out of the model")
    parser.add_argument("--sweep", action="store_true", help="add the 6 s head-sweep step to calibration")
    args = parser.parse_args()

    from engine.eyes import os_input
    from engine.eyes.camera import Camera
    from engine.eyes.eye_features import EyeFeatureExtractor
    from engine.eyes.face_mesh import FaceMesh
    from engine.eyes.head_pose import HeadPoseEstimator

    os_input.make_dpi_aware()               # physical pixels, same as the cursor
    W, H = os_input.screen_size()
    canvas_scale = min(1.0, 1920 / W)       # draw at most 1920 wide; the window stretches it

    trained = None
    if args.quick:
        try:
            trained = GazeModel.load()
            if (trained.screen_w, trained.screen_h) != (W, H):
                print(f"The trained model is for a {trained.screen_w}x{trained.screen_h} screen, this one is {W}x{H}. "
                      "Running a full calibration instead.")
                trained = None
        except (FileNotFoundError, ValueError) as e:
            print(f"No usable trained model ({e}). Running a full calibration instead.")
    print(f"Screen {W}x{H}. Profile '{args.profile}'. {'Quick re-center' if trained else f'{args.points}-dot calibration'}, "
          f"head pose {'off' if args.no_head else 'on'}, iris {args.iris}, up/down inputs {args.y_inputs}.")

    cam = Camera(index=args.camera, fps=args.fps)
    mode = cam.start()
    camera = f"{mode.width}x{mode.height} {mode.fourcc} {mode.backend} camera {mode.index}"
    print(f"Camera: {camera}")
    mesh = FaceMesh()
    eyes = EyeFeatureExtractor(iris_mode=args.iris)
    head = HeadPoseEstimator()
    app = GazeDotApp(W, H, points=args.points, y_inputs=args.y_inputs, sweep=args.sweep and not args.no_head,
                     profile=args.profile, trained=trained, camera=camera, iris_mode=args.iris)

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
                app.reset()
            elif key == ord("c"):
                app.recenter_live(now)
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
