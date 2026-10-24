"""Live view for head pose and gestures. Nothing is saved.

Run:  .\\.venv\\Scripts\\python.exe -m tests.head_bars
      .\\.venv\\Scripts\\python.exe -m tests.head_bars --camera 1   (if the picture is black)
Keys: N = set neutral (look at the screen center first), [ ] = nod threshold, ; ' = shake threshold,
      M = mirror on/off, Q or Esc = quit

Checks (pass = all true):
  1. Press N facing the screen. Turn to your right: yaw goes positive. Chin down: pitch goes
     positive. Tilt toward your right shoulder: roll goes positive. If one runs the wrong way,
     flip its sign at the top of engine/eyes/head_pose.py.
  2. Nod 5 times: 5 "nod". Shake 5 times: 5 "shake". Hold a tilt each way: tilt_left / tilt_right.
  3. Read, talk and look around the screen for a minute: no gestures at all.
     If any fire, raise the thresholds; if real nods are missed, lower them.
     Note the values that work for you (shown at the bottom of the window).
"""
import argparse
import time
from collections import deque

import cv2
import numpy as np

from engine.config import settings
from engine.eyes.camera import Camera
from engine.eyes.face_mesh import FaceMesh
from engine.eyes.head_gestures import GestureDetector
from engine.eyes.head_pose import HeadPoseEstimator
from tests.eye_bars import BAD, BG, DISPLAY_H, GOOD, QUIET, TEXT, WARN, bar, text

WINDOW = "Diseyeble - head pose (Q to quit)"
PANEL_W = 420


def render(preview: np.ndarray, info: dict) -> np.ndarray:
    scale = DISPLAY_H / preview.shape[0]
    view = cv2.resize(preview, (int(preview.shape[1] * scale), DISPLAY_H), interpolation=cv2.INTER_AREA)
    panel = np.full((DISPLAY_H, PANEL_W, 3), BG, dtype=np.uint8)

    y = 28
    text(panel, "Head pose and gestures", 16, y, TEXT, 0.65, 2)
    y += 26
    fps_color = GOOD if info["cam_fps"] >= 25 else WARN
    text(panel, f"camera {info['cam_fps']:.1f} fps   processing {info['proc_ms']:.1f} ms", 16, y, fps_color, 0.45)
    y += 20
    text(panel, "Face found" if info["pose"] else "No face found. Face the camera, add light.",
         16, y, GOOD if info["pose"] else BAD, 0.45)
    y += 30

    pose = info["pose"]
    text(panel, "Degrees from neutral (+ = your right / chin down)", 16, y, TEXT, 0.45)
    y += 12
    bar(panel, 16, y, "yaw", pose.yaw if pose else None, -30, 30, 0.0)
    bar(panel, 16, y + 22, "pitch", pose.pitch if pose else None, -30, 30, 0.0)
    bar(panel, 16, y + 44, "roll", pose.roll if pose else None, -30, 30, 0.0)
    y += 84

    det = info["detector"]
    text(panel, f"nod {det.nod_deg:.1f} deg   shake {det.shake_deg:.1f} deg   tilt {det.tilt_deg:.1f} deg",
         16, y, QUIET, 0.45)
    y += 30
    text(panel, "Gestures (newest first)", 16, y, TEXT, 0.5)
    y += 22
    now = info["now_ms"]
    for g in info["gestures"]:
        age = (now - g.t_ms) / 1000
        text(panel, f"{g.type:<10} {age:5.1f} s ago", 16, y, GOOD if age < 1.5 else QUIET, 0.5)
        y += 22
    counts = ", ".join(f"{k} {v}" for k, v in sorted(info["counts"].items())) or "none yet"
    text(panel, f"Totals: {counts}", 16, DISPLAY_H - 58, QUIET, 0.45)
    text(panel, f"Your thresholds: nod_deg={det.nod_deg:.1f}, shake_deg={det.shake_deg:.1f}",
         16, DISPLAY_H - 36, QUIET, 0.4)
    text(panel, "N neutral  [ ] nod  ; ' shake  M mirror  Q quit", 16, DISPLAY_H - 14, QUIET, 0.45)
    return np.hstack([view, panel])


def main() -> None:
    parser = argparse.ArgumentParser(description="Live head pose and gesture view")
    parser.add_argument("--camera", type=int, default=None, help="camera index (default from config)")
    parser.add_argument("--fps", type=int, default=None, help="ask the camera for this frame rate")
    args = parser.parse_args()

    cam = Camera(index=args.camera, fps=args.fps)
    mode = cam.start()
    print(f"Camera: {mode.width}x{mode.height} {mode.fourcc} via {mode.backend}, camera {mode.index}")
    mesh = FaceMesh()
    estimator = HeadPoseEstimator()
    detector = GestureDetector()
    gestures = deque(maxlen=8)
    counts: dict[str, int] = {}
    mirror = settings.preview_mirror
    proc_ms = deque(maxlen=30)
    frame_id = 0
    last_face = None
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            frame, t_ms, frame_id = cam.wait_next(frame_id)
            if frame is None:
                continue
            t0 = time.perf_counter()
            face = mesh.process(frame, t_ms)
            pose = estimator.compute(face) if face is not None else None
            if pose is not None:
                last_face = face
                g = detector.update(pose)
                if g is not None:
                    gestures.appendleft(g)
                    counts[g.type] = counts.get(g.type, 0) + 1
                    print(f"{g.type} at {g.t_ms} ms")
            proc_ms.append((time.perf_counter() - t0) * 1000)

            preview = cv2.flip(frame, 1) if mirror else frame
            info = {"cam_fps": cam.measured_fps, "proc_ms": sum(proc_ms) / len(proc_ms),
                    "pose": pose, "detector": detector, "gestures": list(gestures),
                    "counts": counts, "now_ms": t_ms}
            cv2.imshow(WINDOW, render(preview, info))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("n") and last_face is not None:
                estimator.set_neutral(last_face)
                detector.reset()
            if key == ord("["):
                detector.set_thresholds(nod_deg=max(2.0, detector.nod_deg - 1))
            if key == ord("]"):
                detector.set_thresholds(nod_deg=detector.nod_deg + 1)
            if key == ord(";"):
                detector.set_thresholds(shake_deg=max(2.0, detector.shake_deg - 1))
            if key == ord("'"):
                detector.set_thresholds(shake_deg=detector.shake_deg + 1)
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
