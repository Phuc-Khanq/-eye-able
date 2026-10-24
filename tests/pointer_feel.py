"""Head pointer feel test: compare 'head' (head as a mouse with acceleration) with 'head-abs' (fixed head
angle per screen spot), by feel and by the time it takes to reach targets. Nothing is saved but numbers.

Run:  .\\.venv\\Scripts\\python.exe -m tests.pointer_feel
      .\\.venv\\Scripts\\python.exe -m tests.pointer_feel --camera 1      (if the picture is black)

A full-screen dark window; the real cursor follows your head.
Keys (this window):
  1 = head (new)   2 = head-abs (old)   C = recenter (face the screen center, then press)
  SPACE = reach test: 8 targets, one at a time, near the corners and edges. Move the cursor onto
          the ring and hold it there for 0.3 s; the time is recorded. Do it once per mode.
  + / - = faster / slower overall (sensitivity)   [ / ] = less / more acceleration (fast-turn gain)
  , / . = slower / faster fine aim (slow-turn gain)   Q or Esc = quit, prints the comparison and the values
On quit, a numbers-only log (frame time, head angles, cursor position, mode, target) is written to
calibration/pointer_logs/ (git ignores it), so the pointer can be tuned on real camera data.
"""
import argparse
import math
import random
import statistics
import time

import cv2
import numpy as np

TARGETS = [(0.08, 0.08), (0.92, 0.08), (0.92, 0.92), (0.08, 0.92), (0.5, 0.06), (0.95, 0.5), (0.5, 0.94),
           (0.05, 0.5)]
RADIUS = 50
HOLD_S = 0.3
TIMEOUT_S = 10.0
WINDOW = "Diseyeble pointer feel"
BG, TEXT, QUIET, RING, GOOD = (32, 28, 26), (235, 235, 235), (150, 150, 150), (90, 200, 250), (120, 210, 120)


def main() -> None:
    p = argparse.ArgumentParser(description="Head pointer feel and reach test")
    p.add_argument("--camera", type=int, default=None)
    args = p.parse_args()

    from engine.config import settings
    from engine.eyes import os_input
    from engine.eyes.camera import Camera
    from engine.eyes.cursor_driver import CursorDriver
    from engine.eyes.eye_features import EyeFeatureExtractor
    from engine.eyes.face_mesh import FaceMesh
    from engine.eyes.head_pose import HeadPoseEstimator
    import engine.eyes.pointer as P
    from engine.eyes.pointer import Pointer

    os_input.make_dpi_aware()
    W, H = os_input.screen_size()
    cam = Camera(index=args.camera)
    cam.start()
    mesh, feats, head = FaceMesh(), EyeFeatureExtractor(), HeadPoseEstimator()
    pointer = Pointer(W, H, "head")
    driver = CursorDriver(os_input.move, still_px=pointer.still_px).start()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    scale = min(1.0, 1280 / W)
    w, h = int(W * scale), int(H * scale)

    results: dict[str, list[float]] = {"head": [], "head-abs": []}
    firsts: dict[str, list[float]] = {"head": [], "head-abs": []}    # time until the cursor first touched the ring
    entries: dict[str, list[int]] = {"head": [], "head-abs": []}     # times it entered the ring (1 = no overshoot)
    test = None            # {"order": [...], "i": int, "t0": float, "inside_since": float | None}
    frame_id, face, note = 0, None, "Face the screen center and press C, then move your head."
    log: list[tuple] = []   # (t_ms, mode, yaw, pitch, x, y, target index or -1, eyes usable 0/1)
    try:
        while True:
            frame, t_ms, new_id = cam.wait_next(frame_id)
            if frame is not None and new_id != frame_id:
                frame_id = new_id
                face = mesh.process(frame, t_ms)
                if face is not None:
                    pose = head.compute(face)
                    ef = feats.compute(face)
                    pt = pointer.update(face.t_ms, ef, pose)
                    driver.set_target(pt.x, pt.y)
                    log.append((face.t_ms, pointer.mode, round(pose.yaw, 3), round(pose.pitch, 3), round(pt.x, 1),
                                round(pt.y, 1), test["i"] if test else -1, int(ef.usable)))
            cx, cy = driver.position or (W / 2, H / 2)

            img = np.full((h, w, 3), BG, np.uint8)
            mode = pointer.mode
            cv2.putText(img, f"mode: {mode}   sensitivity {settings.sensitivity:.1f}   (1 head, 2 head-abs, C recenter, "
                             f"SPACE test, +/- speed, [ ] accel, , . aim, Q quit)", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT, 1,
                        cv2.LINE_AA)
            cv2.putText(img, note, (20, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.6, QUIET, 1, cv2.LINE_AA)
            if face is None:
                cv2.putText(img, "No face: face the camera", (20, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 90, 230), 1)

            if test is not None:
                tx, ty = test["order"][test["i"]]
                px, py = tx * W, ty * H
                inside = math.hypot(cx - px, cy - py) <= RADIUS
                now = time.monotonic()
                if inside and test["inside_since"] is None:
                    test["inside_since"] = now
                    test["entries"] += 1
                    if test["first"] is None:
                        test["first"] = now - test["t0"]
                elif not inside:
                    test["inside_since"] = None
                done = inside and now - test["inside_since"] >= HOLD_S
                timed_out = now - test["t0"] > TIMEOUT_S
                cv2.circle(img, (int(px * scale), int(py * scale)), int(RADIUS * scale), GOOD if inside else RING, 3,
                           cv2.LINE_AA)
                if done or timed_out:
                    secs = (test["inside_since"] if done else now) - test["t0"]
                    results[mode].append(secs if done else TIMEOUT_S)
                    if test["first"] is not None:
                        firsts[mode].append(test["first"])
                    entries[mode].append(test["entries"])
                    first = "-" if test["first"] is None else f"{test['first']:.2f} s"
                    print(f"  {mode:<8} target {test['i'] + 1}: {'%.2f s' % secs if done else 'timed out'}"
                          f"   (first touch {first}, entered {test['entries']}x)")
                    test["i"] += 1
                    if test["i"] == len(test["order"]):
                        r = results[mode][-len(test["order"]):]
                        note = f"{mode}: median {statistics.median(r):.2f} s per target. Try the other mode (1 / 2)."
                        print(f"  {mode} done: median {statistics.median(r):.2f} s")
                        test = None
                    else:
                        test["t0"], test["inside_since"] = now, None
                        test["first"], test["entries"] = None, 0

            cv2.imshow(WINDOW, img)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("1"):
                pointer.set_mode("head")
                driver.stable.still_px = pointer.still_px
                note = "head: quick turns go far, slow turns are precise; no fixed center."
            elif k == ord("2"):
                pointer.set_mode("head-abs")
                driver.stable.still_px = pointer.still_px
                note = "head-abs: each head angle is a fixed spot (about 15 degrees to a side edge)."
            elif k == ord("c") and face is not None:
                head.set_neutral(face)
                pointer.reset()
                note = "Recentered."
            elif k in (ord("+"), ord("=")):
                settings.sensitivity = min(settings.sensitivity + 0.1, 3.0)
            elif k == ord("-"):
                settings.sensitivity = max(settings.sensitivity - 0.1, 0.3)
            elif k in (ord("["), ord("]")):
                P.GAIN_FAST = min(max(P.GAIN_FAST + (10 if k == ord("]") else -10), P.GAIN_SLOW), 200.0)
                note = f"fast-turn gain {P.GAIN_FAST:.0f} px/deg (acceleration)"
            elif k in (ord(","), ord(".")):
                P.GAIN_SLOW = min(max(P.GAIN_SLOW + (2 if k == ord(".") else -2), 2.0), P.GAIN_FAST)
                note = f"slow-turn gain {P.GAIN_SLOW:.0f} px/deg (fine aim)"
            elif k == ord(" ") and test is None:
                order = TARGETS[:]
                random.shuffle(order)
                test = {"order": order, "i": 0, "t0": time.monotonic(), "inside_since": None, "first": None,
                        "entries": 0}
                note = f"Reach test ({mode}): move onto each ring and hold 0.3 s."
                print(f"Reach test, mode {mode}:")
    finally:
        driver.stop()
        cam.stop()
        mesh.close()
        cv2.destroyAllWindows()
        if log:
            from pathlib import Path
            folder = Path(__file__).resolve().parents[1] / "calibration" / "pointer_logs"
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / time.strftime("%Y%m%d-%H%M%S.csv")
            with open(path, "w", encoding="utf-8") as f:
                f.write("t_ms,mode,yaw,pitch,x,y,target,eyes\n")
                f.writelines(",".join(map(str, row)) + "\n" for row in log)
            print(f"Log (numbers only): {path}")
        print("\n=== Reach test (time to reach and hold each target) ===")
        for m, r in results.items():
            if r and firsts[m]:
                ft, en = statistics.median(firsts[m]), statistics.mean(entries[m])
                hint = ("overshooting: quick turns too strong (lower GAIN_FAST)" if en >= 2.0 else
                        "slow to arrive: quick turns too weak (raise GAIN_FAST)" if ft >= 1.8 else
                        "slow to settle: fine aim too weak or shaky (raise GAIN_SLOW a little)" if
                        statistics.median(r) - ft >= 1.0 else "balanced")
                print(f"  {m:<8} first touch median {ft:.2f} s, entered {en:.1f}x per target -> {hint}")
            if r:
                print(f"  {m:<8} {len(r)} targets, median {statistics.median(r):.2f} s, "
                      f"timed out {sum(x >= TIMEOUT_S for x in r)}")
        print(f"  values at the end: sensitivity {settings.sensitivity:.1f}, GAIN_SLOW {P.GAIN_SLOW:.0f}, "
              f"GAIN_FAST {P.GAIN_FAST:.0f}  (send these to Khang to make them the default)")


if __name__ == "__main__":
    main()
