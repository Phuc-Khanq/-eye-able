"""Live view for the eye-feature quick test. Nothing is saved.

Run:  .\\.venv\\Scripts\\python.exe -m tests.eye_bars
Keys: C = clear the ranges, M = mirror on/off, Q or Esc = quit

Quick test (pass = both true):
  1. Press C. Look at the left edge of your screen, then the right edge, a few times.
     "h range" should reach about 0.2, and both eyes' h bars move the same way.
  2. Press C. Look at the screen center and lean in and out, then tilt your head sideways.
     "h range" should stay small (well under 0.1).
"""
import time
from collections import deque

import cv2
import numpy as np

from engine.config import settings
from engine.eyes.camera import Camera
from engine.eyes.eye_features import LEFT_EYE, RIGHT_EYE, EyeFeatureExtractor
from engine.eyes.face_mesh import FaceMesh

WINDOW = "Diseyeble - eye features (Q to quit)"
PANEL_W = 400
DISPLAY_H = 540

BG = (32, 28, 26)
TEXT = (235, 235, 235)
QUIET = (150, 150, 150)
GOOD = (120, 210, 120)
WARN = (60, 170, 240)
BAD = (90, 90, 230)
BAR_BG = (70, 64, 60)
BAR_FG = (240, 200, 90)
FONT = cv2.FONT_HERSHEY_SIMPLEX


class RangeTracker:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.h = [np.inf, -np.inf]
        self.v = [np.inf, -np.inf]

    def update(self, h: float, v: float) -> None:
        self.h = [min(self.h[0], h), max(self.h[1], h)]
        self.v = [min(self.v[0], v), max(self.v[1], v)]

    @staticmethod
    def span(r) -> float:
        return r[1] - r[0] if np.isfinite(r[0]) else 0.0


def draw_eye_points(frame: np.ndarray, face) -> None:
    h, w = frame.shape[:2]
    for eye in (RIGHT_EYE, LEFT_EYE):
        for key, idx in eye.items():
            x, y = face.landmarks[idx, 0] * w, face.landmarks[idx, 1] * h
            color = (90, 230, 90) if key == "iris" else (60, 220, 240) if key in ("a", "b") else (240, 200, 90)
            cv2.circle(frame, (int(x), int(y)), 3 if key == "iris" else 2, color, -1, cv2.LINE_AA)


def text(img, s, x, y, color=TEXT, scale=0.5, thick=1) -> None:
    cv2.putText(img, s, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def bar(img, x, y, label, value, lo, hi, mid=None, width=220) -> None:
    text(img, label, x, y + 11, QUIET, 0.45)
    bx = x + 70
    cv2.rectangle(img, (bx, y), (bx + width, y + 14), BAR_BG, -1)
    if mid is not None:
        mx = bx + int((mid - lo) / (hi - lo) * width)
        cv2.line(img, (mx, y - 2), (mx, y + 16), QUIET, 1)
    if value is not None and np.isfinite(value):
        vx = bx + int(np.clip((value - lo) / (hi - lo), 0, 1) * width)
        cv2.rectangle(img, (vx - 3, y - 2), (vx + 3, y + 16), BAR_FG, -1)
        text(img, f"{value:+.3f}", bx + width + 8, y + 12, TEXT, 0.45)


def eye_section(panel, y, title, m) -> int:
    text(panel, title, 16, y, TEXT, 0.5)
    y += 12
    bar(panel, 16, y, "h", m.h if m else None, 0.2, 0.8, 0.5)
    bar(panel, 16, y + 22, "v", m.v if m else None, -0.25, 0.25, 0.0)
    bar(panel, 16, y + 44, "lid", m.lid if m else None, 0.0, 0.5)
    return y + 74


def render(preview: np.ndarray, info: dict) -> np.ndarray:
    """Builds the window image: camera preview on the left, numbers and bars on the right."""
    scale = DISPLAY_H / preview.shape[0]
    view = cv2.resize(preview, (int(preview.shape[1] * scale), DISPLAY_H), interpolation=cv2.INTER_AREA)
    panel = np.full((DISPLAY_H, PANEL_W, 3), BG, dtype=np.uint8)

    y = 28
    text(panel, "Eye features", 16, y, TEXT, 0.65, 2)
    y += 26
    text(panel, info["camera"], 16, y, QUIET, 0.45)
    y += 20
    fps_color = GOOD if info["cam_fps"] >= 25 else WARN
    text(panel, f"camera {info['cam_fps']:.1f} fps   processing {info['proc_ms']:.1f} ms", 16, y, fps_color, 0.45)
    y += 20
    if not info["face"]:
        text(panel, "No face found. Face the camera, add light.", 16, y, BAD, 0.45)
    elif info["blink"]:
        text(panel, "Blink: frame skipped, holding last value", 16, y, WARN, 0.45)
    else:
        text(panel, "Face found", 16, y, GOOD, 0.45)
    y += 30

    feats = info["feats"]
    text(panel, "Both eyes (weighted by width)", 16, y, TEXT, 0.5)
    y += 12
    bar(panel, 16, y, "h", feats.h if feats else None, 0.2, 0.8, 0.5)
    bar(panel, 16, y + 22, "v", feats.v if feats else None, -0.25, 0.25, 0.0)
    bar(panel, 16, y + 44, "lid", feats.lid if feats else None, 0.0, 0.5)
    y += 82
    y = eye_section(panel, y, "Your right eye (image left)", info["right"])
    y += 8
    y = eye_section(panel, y, "Your left eye (image right)", info["left"])
    y += 8

    h_span, v_span = info["h_span"], info["v_span"]
    color = GOOD if h_span >= 0.15 else QUIET
    text(panel, f"h range {h_span:.3f}   (edge to edge: about 0.2)", 16, y, color, 0.5)
    text(panel, f"v range {v_span:.3f}", 16, y + 22, QUIET, 0.5)
    text(panel, "C clear ranges   M mirror   Q quit", 16, DISPLAY_H - 14, QUIET, 0.45)
    return np.hstack([view, panel])


def main() -> None:
    cam = Camera()
    mode = cam.start()
    camera_label = f"{mode.width}x{mode.height} {mode.fourcc} via {mode.backend}, camera {mode.index}"
    print("Camera:", camera_label)
    mesh = FaceMesh()
    extractor = EyeFeatureExtractor()
    ranges = RangeTracker()
    mirror = settings.preview_mirror
    proc_ms = deque(maxlen=30)
    frame_id = 0
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
            if feats is not None and feats.usable:
                ranges.update(feats.h, feats.v)

            preview = frame.copy()
            if face is not None:
                draw_eye_points(preview, face)
            if mirror:
                preview = cv2.flip(preview, 1)     # mirror only what you see
            info = {
                "camera": camera_label, "cam_fps": cam.measured_fps,
                "proc_ms": sum(proc_ms) / len(proc_ms),
                "face": face is not None, "blink": extractor.blinking if face is not None else False,
                "feats": feats, "right": extractor.right if face is not None else None,
                "left": extractor.left if face is not None else None,
                "h_span": RangeTracker.span(ranges.h), "v_span": RangeTracker.span(ranges.v),
            }
            cv2.imshow(WINDOW, render(preview, info))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                ranges.reset()
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
