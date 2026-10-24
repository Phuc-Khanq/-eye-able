"""Numbers for the pitch: how well can you click buttons with gaze and a nod (or dwell)?

Run:  .\\.venv\\Scripts\\python.exe -m tests.click_test
      .\\.venv\\Scripts\\python.exe -m tests.click_test --trials 10 --no-dwell
First calibrate:  .\\.venv\\Scripts\\python.exe -m tests.gaze_dot --points 16

A full-screen grid of buttons. One lights up at a time; click it the gaze_cursor way (look, then
nod or dwell). Sizes run largest first: 160, 96, then 48 px, --trials buttons each. Each try ends
at the first click or after TIMEOUT_S. Hit = the click landed inside the lit button.
The terminal prints each try and a summary per size; the same numbers go to a CSV in calibration/
(git ignores it). No camera frames or video are saved.

Keys (this window): N = look at the screen center and press to zero the head, SPACE = start,
C = recenter (look at the center dot, nod), Q or Esc = quit.
"""
import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from tests.gaze_cursor import GazeCursor

SIZES = (160, 96, 48)    # button sides in physical pixels, largest first to warm up
COLS, ROWS = 5, 3
MARGIN = 0.12            # grid inset from the screen edges
GAP_S = 0.8              # pause between tries; clicks in it don't count
TIMEOUT_S = 15.0

BG = (32, 28, 26)
TEXT = (235, 235, 235)
QUIET = (150, 150, 150)
IDLE = (70, 64, 60)
LIT = (90, 200, 250)
GOOD = (120, 210, 120)
BAD = (90, 90, 230)
CURSOR = (250, 170, 60)
FONT = cv2.FONT_HERSHEY_SIMPLEX
REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Trial:
    size: int
    target: tuple[float, float]
    hit: bool = False
    seconds: float = 0.0
    how: str = ""            # "nod", "dwell" or "timeout"
    click: tuple[float, float] | None = None


def grid_slots(W: int, H: int) -> list[tuple[float, float]]:
    xs = np.linspace(MARGIN, 1 - MARGIN, COLS) * W
    ys = np.linspace(MARGIN, 1 - MARGIN, ROWS) * H
    return [(float(x), float(y)) for y in ys for x in xs]


def make_trials(W: int, H: int, per_size: int, seed: int | None = None) -> list[Trial]:
    """Per size, random slots without the same slot twice in a row."""
    rng = np.random.default_rng(seed)
    slots = grid_slots(W, H)
    trials, last = [], None
    for size in SIZES:
        for _ in range(per_size):
            choices = [i for i in range(len(slots)) if i != last]
            last = int(rng.choice(choices))
            trials.append(Trial(size=size, target=slots[last]))
    return trials


def inside(trial: Trial, x: float, y: float) -> bool:
    half = trial.size / 2
    return abs(x - trial.target[0]) <= half and abs(y - trial.target[1]) <= half


class ClickTest:
    """The flow as a state machine: ready -> (gap -> trial) x N -> done. Feed it clicks and time."""

    def __init__(self, W: int, H: int, per_size: int, seed: int | None = None) -> None:
        self.W, self.H = W, H
        self.trials = make_trials(W, H, per_size, seed)
        self.i = 0
        self.state = "ready"
        self.t0 = 0.0
        self.last_result: Trial | None = None

    @property
    def current(self) -> Trial | None:
        return self.trials[self.i] if self.i < len(self.trials) else None

    def start(self, now: float) -> None:
        self.i = 0
        self.state = "gap"
        self.t0 = now

    def update(self, now: float, click=None) -> Trial | None:
        """click: a gaze_cursor Click or None. Returns the finished Trial when a try ends."""
        if self.state == "gap":
            if now - self.t0 >= GAP_S:
                self.state = "trial"
                self.t0 = now
            return None
        if self.state != "trial":
            return None
        t = self.current
        if click is not None:
            t.hit = inside(t, click.x, click.y)
            t.how, t.click, t.seconds = click.how, (click.x, click.y), now - self.t0
        elif now - self.t0 >= TIMEOUT_S:
            t.hit, t.how, t.seconds = False, "timeout", TIMEOUT_S
        else:
            return None
        self.last_result = t
        self.i += 1
        self.state = "done" if self.i == len(self.trials) else "gap"
        self.t0 = now
        return t

    def summary(self) -> list[str]:
        lines = []
        done = [t for t in self.trials[:self.i]]
        for size in SIZES:
            ts = [t for t in done if t.size == size]
            if not ts:
                continue
            hits = [t for t in ts if t.hit]
            secs = [t.seconds for t in hits]
            how = ", ".join(f"{k} {sum(t.how == k for t in ts)}" for k in ("nod", "dwell", "timeout")
                            if any(t.how == k for t in ts))
            time_txt = f"median {np.median(secs):.1f} s, mean {np.mean(secs):.1f} s per hit" if secs else "no hits"
            lines.append(f"{size:>3} px: {len(hits)}/{len(ts)} hits ({100 * len(hits) / len(ts):.0f}%), "
                         f"{time_txt}   [{how}]")
        if done:
            hits = sum(t.hit for t in done)
            lines.append(f"all:    {hits}/{len(done)} hits ({100 * hits / len(done):.0f}%)")
        return lines

    def save_csv(self) -> Path:
        out = REPO_ROOT / "calibration" / f"click_test_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["size_px", "target_x", "target_y", "hit", "seconds", "how", "click_x", "click_y"])
            for t in self.trials[:self.i]:
                cx, cy = t.click if t.click else ("", "")
                w.writerow([t.size, round(t.target[0]), round(t.target[1]), int(t.hit), round(t.seconds, 2),
                            t.how, round(cx) if cx != "" else "", round(cy) if cy != "" else ""])
        return out

    def draw(self, cursor, recentering: bool, head_zeroed: bool, face_ok: bool, fps: float,
             canvas_scale: float) -> np.ndarray:
        s = canvas_scale
        cw, ch = int(round(self.W * s)), int(round(self.H * s))
        img = np.full((ch, cw, 3), BG, dtype=np.uint8)
        fs = 0.75 * cw / 1920

        def P(x, y):
            return int(round(x * s)), int(round(y * s))

        def say(txt, row, color=TEXT, big=False):
            y = int(60 * cw / 1920) + int(row * 40 * cw / 1920)
            cv2.putText(img, txt, (int(40 * cw / 1920), y), FONT, fs * (1.3 if big else 1.0), color,
                        2 if big else 1, cv2.LINE_AA)

        t = self.current
        if self.state in ("gap", "trial") and t is not None:
            half = t.size / 2
            for (x, y) in grid_slots(self.W, self.H):
                lit = self.state == "trial" and (x, y) == t.target
                cv2.rectangle(img, P(x - half, y - half), P(x + half, y + half), LIT if lit else IDLE,
                              -1 if lit else 2, cv2.LINE_AA)
            say(f"{t.size} px buttons   try {self.i + 1} of {len(self.trials)}", 0, QUIET)
            if self.last_result is not None and self.state == "gap":
                r = self.last_result
                say("hit" if r.hit else ("too slow" if r.how == "timeout" else "miss"), 1, GOOD if r.hit else BAD,
                    big=True)
        elif self.state == "ready":
            say("Click test", 0, TEXT, big=True)
            say("A button lights up. Look at it, then nod (or rest your gaze on it) to click.", 2)
            say(f"{len(self.trials)} buttons: {', '.join(str(z) for z in SIZES)} px. Hands in your lap.", 3)
            say(("Head zeroed." if head_zeroed else "Look at the center of the screen and press N.") +
                "   SPACE start   C recenter   Q quit", 5, QUIET if head_zeroed else TEXT)
        elif self.state == "done":
            say("Done", 0, GOOD, big=True)
            for k, line in enumerate(self.summary()):
                say(line, 2 + k)
            say("SPACE run again   Q quit", 3 + len(self.summary()), QUIET)

        if recentering:
            cv2.circle(img, P(self.W / 2, self.H / 2), int(26 * cw / 1920), LIT, 3, cv2.LINE_AA)
            cv2.circle(img, P(self.W / 2, self.H / 2), max(2, int(5 * cw / 1920)), TEXT, -1, cv2.LINE_AA)
            say("Recenter: look at the dot, then nod. Shake cancels.", 1, LIT)
        if cursor is not None:
            cv2.circle(img, P(*cursor), int(14 * cw / 1920), CURSOR, 2, cv2.LINE_AA)
        status = f"camera {fps:.0f} fps" + ("" if face_ok else "   NO FACE: face the camera")
        cv2.putText(img, status, (int(40 * cw / 1920), ch - int(30 * cw / 1920)), FONT, fs * 0.8,
                    QUIET if face_ok else BAD, 1, cv2.LINE_AA)
        return img


def main() -> None:
    parser = argparse.ArgumentParser(description="Click test: hit rate and time per click, per button size")
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--iris", choices=["center", "ring"], default="center",
                        help="use the same setting you calibrated with")
    parser.add_argument("--model", default=None, help="gaze model file (default: calibration/gaze_model.json)")
    parser.add_argument("--trials", type=int, default=8, help="buttons per size")
    parser.add_argument("--no-dwell", action="store_true", help="nod clicks only")
    args = parser.parse_args()

    from engine.eyes import os_input
    from engine.eyes.calibration import DEFAULT_PATH, GazeModel
    from engine.eyes.camera import Camera
    from engine.eyes.eye_features import EyeFeatureExtractor
    from engine.eyes.face_mesh import FaceMesh
    from engine.eyes.head_pose import HeadPoseEstimator

    os_input.make_dpi_aware()
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
    canvas_scale = min(1.0, 1920 / W)

    cam = Camera(index=args.camera, fps=args.fps)
    mode = cam.start()
    print(f"Screen {W}x{H}. Camera {mode.width}x{mode.height} via {mode.backend}, camera {mode.index}.")
    mesh = FaceMesh()
    eyes = EyeFeatureExtractor(iris_mode=args.iris)
    head = HeadPoseEstimator()
    cursor_app = GazeCursor(model, W, H, dwell=not args.no_dwell)
    test = ClickTest(W, H, per_size=args.trials)
    head_zeroed = False

    window = "Diseyeble click test"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window, 0, 0)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    print("Look at the center of the screen and press N, then SPACE to start.")

    frame_id = 0
    face = None
    try:
        while True:
            frame, t_ms, frame_id = cam.wait_next(frame_id)
            now = time.monotonic()
            click = None
            if frame is not None:
                face = mesh.process(frame, t_ms)
                if face is not None:
                    feats = eyes.compute(face)
                    pose = head.compute(face)
                    cursor, click = cursor_app.step(face.t_ms, feats, pose)
                    if cursor is not None:
                        os_input.move(*cursor)  # the real cursor follows too, so it feels like gaze_cursor
                    for msg in cursor_app.events:
                        print(msg)
                    cursor_app.events.clear()

            if test.state != "trial":
                click = None  # clicks between tries don't count
            done = test.update(now, click)
            if done is not None:
                result = "HIT " if done.hit else ("SLOW" if done.how == "timeout" else "MISS")
                where = f" at ({done.click[0]:.0f}, {done.click[1]:.0f})" if done.click else ""
                print(f"{test.i:>2}. {done.size:>3} px  {result}  {done.seconds:4.1f} s  {done.how}{where}")
                if test.state == "done":
                    print("\nSummary")
                    for line in test.summary():
                        print("  " + line)
                    print(f"Saved numbers to {test.save_csv()}")

            img = test.draw(cursor_app.cursor if not cursor_app.paused or cursor_app.recentering else None,
                            cursor_app.recentering, head_zeroed, face is not None, cam.measured_fps,
                            canvas_scale)
            cv2.imshow(window, img)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("n") and face is not None:
                head.set_neutral(face)
                cursor_app.restart()
                cursor_app.set_paused(False)
                head_zeroed = True
                print("Head zeroed.")
            elif key == ord(" ") and head_zeroed and test.state in ("ready", "done"):
                if test.state == "done":
                    test = ClickTest(W, H, per_size=args.trials)
                test.start(now)
                cursor_app.dwell.reset()
            elif key == ord("c"):
                if cursor_app.recentering:
                    cursor_app.cancel_recenter()
                else:
                    cursor_app.start_recenter()
    finally:
        if 0 < test.i < len(test.trials):
            print("\nStopped early. Summary so far")
            for line in test.summary():
                print("  " + line)
            print(f"Saved numbers to {test.save_csv()}")
        cam.stop()
        mesh.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
