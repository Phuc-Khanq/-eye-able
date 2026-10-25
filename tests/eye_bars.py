"""Live view for the eye-feature tests. Nothing is saved.

Run:  .\\.venv\\Scripts\\python.exe -m tests.eye_bars
      .\\.venv\\Scripts\\python.exe -m tests.eye_bars --fps 60      (if the camera check says it can)
Keys: C = clear ranges, I = iris center or ring average, M = mirror, Q or Esc = quit

How to measure (do both, in this order):
  1. Range: press C, then look at the left edge of your screen and the right edge, a few times.
  2. Steadiness: look at one point in the middle and hold still for 2 seconds.
The table then shows, for each signal, the edge-to-edge range, the jitter while holding still,
and the cursor wobble that jitter would cause on a 1920-pixel-wide screen. Lower wobble is better.

Eyelid check: "lid now" is this frame; "lid held" is what calibration gets, frozen at the last
open-eye value while "blink score" is above the threshold line. Close your eyes: lid now should
drop well below its open value, blink score should cross the line, and lid held should stay put.
"""
import argparse
import time
from collections import deque

import cv2
import numpy as np

from engine.config import settings
from engine.eyes.camera import Camera
from engine.eyes.eye_features import LEFT_EYE, RIGHT_EYE, EyeFeatureExtractor
from engine.eyes.face_mesh import FaceMesh
from engine.eyes.smoothing import OneEuroFilter

WINDOW = "EYEable - eye features (Q to quit)"
PANEL_W = 440
DISPLAY_H = 540
SCREEN_W = 1920  # used only to turn jitter into an example pixel figure

BG = (32, 28, 26)
TEXT = (235, 235, 235)
QUIET = (150, 150, 150)
GOOD = (120, 210, 120)
WARN = (60, 170, 240)
BAD = (90, 90, 230)
BAR_BG = (70, 64, 60)
BAR_FG = (240, 200, 90)
FONT = cv2.FONT_HERSHEY_SIMPLEX


class Signal:
    """Tracks one number: its range since the last reset, and its jitter over the last second."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.recent = deque()
        self.reset()

    def reset(self) -> None:
        self.lo, self.hi = np.inf, -np.inf
        self.recent.clear()

    def update(self, t_s: float, value: float) -> None:
        if not np.isfinite(value):
            return
        self.lo, self.hi = min(self.lo, value), max(self.hi, value)
        self.recent.append((t_s, value))
        while self.recent and t_s - self.recent[0][0] > 1.0:
            self.recent.popleft()

    @property
    def span(self) -> float:
        return self.hi - self.lo if np.isfinite(self.lo) else 0.0

    @property
    def jitter(self) -> float:
        """Standard deviation over the last second. Only meaningful while holding still."""
        if len(self.recent) < 5:
            return float("nan")
        return float(np.std([v for _, v in self.recent]))

    @property
    def wobble_px(self) -> float:
        """Jitter scaled by the edge-to-edge range to pixels on a SCREEN_W-wide screen."""
        if self.span < 0.02 or not np.isfinite(self.jitter):
            return float("nan")
        return self.jitter / self.span * SCREEN_W


def draw_eye_points(frame: np.ndarray, face) -> None:
    h, w = frame.shape[:2]
    for eye in (RIGHT_EYE, LEFT_EYE):
        for key, idx in eye.items():
            for i in (idx if isinstance(idx, tuple) else (idx,)):
                x, y = face.landmarks[i, 0] * w, face.landmarks[i, 1] * h
                if key in ("iris", "ring"):
                    color, r = (90, 230, 90), 3 if key == "iris" else 1
                elif key in ("a", "b"):
                    color, r = (60, 220, 240), 2
                else:
                    color, r = (240, 200, 90), 2
                cv2.circle(frame, (int(x), int(y)), r, color, -1, cv2.LINE_AA)


def text(img, s, x, y, color=TEXT, scale=0.5, thick=1) -> None:
    cv2.putText(img, s, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def bar(img, x, y, label, value, lo, hi, mid=None, width=230) -> None:
    text(img, label, x, y + 11, QUIET, 0.45)
    bx = x + 105
    cv2.rectangle(img, (bx, y), (bx + width, y + 14), BAR_BG, -1)
    if mid is not None:
        mx = bx + int((mid - lo) / (hi - lo) * width)
        cv2.line(img, (mx, y - 2), (mx, y + 16), QUIET, 1)
    if value is not None and np.isfinite(value):
        vx = bx + int(np.clip((value - lo) / (hi - lo), 0, 1) * width)
        cv2.rectangle(img, (vx - 3, y - 2), (vx + 3, y + 16), BAR_FG, -1)
        text(img, f"{value:+.3f}", bx + width + 8, y + 12, TEXT, 0.45)


def fmt(x: float, spec: str) -> str:
    return format(x, spec) if np.isfinite(x) else "-"


def render(preview: np.ndarray, info: dict) -> np.ndarray:
    """Builds the window image: camera preview on the left, numbers and bars on the right."""
    scale = DISPLAY_H / preview.shape[0]
    view = cv2.resize(preview, (int(preview.shape[1] * scale), DISPLAY_H), interpolation=cv2.INTER_AREA)
    panel = np.full((DISPLAY_H, PANEL_W, 3), BG, dtype=np.uint8)
    x0 = 16

    y = 28
    text(panel, "Eye features", x0, y, TEXT, 0.65, 2)
    y += 24
    text(panel, info["camera"], x0, y, QUIET, 0.45)
    y += 20
    fps_color = GOOD if info["cam_fps"] >= 25 else WARN
    text(panel, f"camera {info['cam_fps']:.1f} fps   processing {info['proc_ms']:.1f} ms", x0, y, fps_color, 0.45)
    y += 20
    if not info["face"]:
        text(panel, "No face found. Face the camera, add light.", x0, y, BAD, 0.45)
    elif info["blink"]:
        text(panel, "Blink: frame skipped, holding last value", x0, y, WARN, 0.45)
    else:
        text(panel, f"Face found   iris: {info['iris_mode']}", x0, y, GOOD, 0.45)
    y += 26

    feats = info["feats"]
    raw_h = feats.h if feats else None
    bar(panel, x0, y, "h raw", raw_h, 0.2, 0.8, 0.5)
    bar(panel, x0, y + 22, "h smoothed", info["h_smooth"], 0.2, 0.8, 0.5)
    bar(panel, x0, y + 44, "v smoothed", info["v_smooth"], -0.25, 0.25, 0.0)
    bar(panel, x0, y + 66, "lid now", info["lid_now"], 0.0, 0.5)                 # this frame
    bar(panel, x0, y + 88, "lid held", feats.lid if feats else None, 0.0, 0.5)  # frozen during blinks
    bar(panel, x0, y + 110, "blink score", info["blink_score"], 0.0, 1.0, settings.blink_threshold)
    y += 140
    text(panel, "Each eye (should move together)", x0, y, TEXT, 0.48)
    y += 10
    bar(panel, x0, y, "your right h", info["right"].h if info["right"] else None, 0.2, 0.8, 0.5)
    bar(panel, x0, y + 22, "your left h", info["left"].h if info["left"] else None, 0.2, 0.8, 0.5)
    y += 52
    text(panel, "MediaPipe eye-direction scores", x0, y, TEXT, 0.48)
    y += 10
    bar(panel, x0, y, "blend h", info["blend_h"], -0.6, 0.6, 0.0)
    bar(panel, x0, y + 22, "blend v", info["blend_v"], -0.6, 0.6, 0.0)
    y += 56

    text(panel, "Steadiness", x0, y, TEXT, 0.5)
    text(panel, "range", x0 + 140, y, QUIET, 0.42)
    text(panel, "jitter", x0 + 220, y, QUIET, 0.42)
    text(panel, "wobble", x0 + 300, y, QUIET, 0.42)
    y += 20
    for sig in info["signals"]:
        wob = sig.wobble_px
        color = GOOD if np.isfinite(wob) and wob < 100 else WARN if np.isfinite(wob) and wob < 250 else TEXT
        text(panel, sig.name, x0, y, TEXT, 0.45)
        text(panel, fmt(sig.span, ".3f"), x0 + 140, y, TEXT, 0.45)
        text(panel, fmt(sig.jitter, ".4f"), x0 + 220, y, TEXT, 0.45)
        text(panel, ("+-" + fmt(wob, ".0f") + " px") if np.isfinite(wob) else "-", x0 + 300, y, color, 0.45)
        y += 20
    y += 4
    text(panel, "1) C, then look edge to edge   2) hold still 2 s", x0, y, QUIET, 0.42)
    text(panel, f"wobble = jitter / range x {SCREEN_W} px", x0, y + 18, QUIET, 0.42)
    text(panel, "C clear   I iris mode   M mirror   Q quit", x0, DISPLAY_H - 14, QUIET, 0.45)
    return np.hstack([view, panel])


def main() -> None:
    parser = argparse.ArgumentParser(description="Live eye-feature view")
    parser.add_argument("--camera", type=int, default=None, help="camera index (default from config)")
    parser.add_argument("--fps", type=int, default=None, help="ask the camera for this frame rate")
    args = parser.parse_args()

    cam = Camera(index=args.camera, fps=args.fps)
    mode = cam.start()
    camera_label = f"{mode.width}x{mode.height} {mode.fourcc} via {mode.backend}, camera {mode.index}"
    print("Camera:", camera_label)
    mesh = FaceMesh()
    extractor = EyeFeatureExtractor()
    smooth_h, smooth_v = OneEuroFilter(), OneEuroFilter()
    signals = [Signal("h raw"), Signal("h smoothed"), Signal("blend h")]
    mirror = settings.preview_mirror
    proc_ms = deque(maxlen=30)
    frame_id = 0
    h_s = v_s = float("nan")
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            frame, t_ms, frame_id = cam.wait_next(frame_id)
            if frame is None:
                continue
            t0 = time.perf_counter()
            face = mesh.process(frame, t_ms)       # unmirrored frame in
            feats = extractor.compute(face) if face is not None else None
            proc_ms.append((time.perf_counter() - t0) * 1000)
            t_s = t_ms / 1000.0
            if feats is not None and feats.usable:
                h_s, v_s = smooth_h(t_s, feats.h), smooth_v(t_s, feats.v)
                signals[0].update(t_s, feats.h)
                signals[1].update(t_s, h_s)
                signals[2].update(t_s, extractor.blend_h)

            preview = frame.copy()
            if face is not None:
                draw_eye_points(preview, face)
            if mirror:
                preview = cv2.flip(preview, 1)     # mirror only what you see
            has_face = face is not None
            info = {
                "camera": camera_label, "cam_fps": cam.measured_fps,
                "proc_ms": sum(proc_ms) / len(proc_ms), "face": has_face,
                "blink": extractor.blinking if has_face else False, "iris_mode": extractor.iris_mode,
                "feats": feats, "h_smooth": h_s, "v_smooth": v_s,
                "right": extractor.right if has_face else None, "left": extractor.left if has_face else None,
                "blend_h": extractor.blend_h if has_face else None,
                "blend_v": extractor.blend_v if has_face else None,
                "lid_now": extractor.lid_now if has_face else None,
                "blink_score": extractor.blink_score if has_face else None,
                "signals": signals,
            }
            cv2.imshow(WINDOW, render(preview, info))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                for s in signals:
                    s.reset()
            if key == ord("i"):
                extractor.iris_mode = "ring" if extractor.iris_mode == "center" else "center"
                smooth_h.reset()
                smooth_v.reset()
                for s in signals:
                    s.reset()
            if key == ord("m"):
                mirror = not mirror
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cam.stop()
        mesh.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
