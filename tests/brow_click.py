"""Eyebrow raise as a click: how well does it work? Head pointer + Hung's BrowRaiseDetector, the same
code the engine runs, but no real clicks: only numbers.

Run:  .\\.venv\\Scripts\\python.exe -m tests.brow_click
      .\\.venv\\Scripts\\python.exe -m tests.brow_click --camera 1      (if the picture is black)

A full-screen dark window; the real cursor follows your head (head mode).
Top left: your eyebrow score live. The white line is your resting level, the yellow line is where a
raise counts; it turns green while your brows are up, and "CLICK" flashes when a raise fires.

Keys (this window):
  SPACE = click test: 10 buttons (96 px, the size of a normal button), one at a time, anywhere on the
          screen. Move onto it and raise your eyebrows. A raise outside the button is a miss.
  T     = talk test: 30 s reading the text on screen out loud, with expression, NOT raising on purpose.
          Every raise that fires is a click you didn't want.
  F     = freeze the cursor while the brows are up (on/off). Raising your brows can tilt your head a
          little and move the cursor off the button; this holds it where it was when the raise started.
          Do the click test once with it off and once on to compare.
  C     = recenter (face the screen center, then press)
  + / - = raise threshold up / down (less / more sensitive)   [ / ] = hold time shorter / longer
  Q or Esc = quit: prints the summary and the values, and writes a numbers-only log to calibration/brow_logs/.
"""
import argparse
import math
import random
import statistics
import time

import cv2
import numpy as np

BUTTON = 96               # px, a normal button
N_BUTTONS = 10
TIMEOUT_S = 15.0
TALK_S = 30.0
WINDOW = "Diseyeble eyebrow click"
BG, TEXT, QUIET = (32, 28, 26), (235, 235, 235), (150, 150, 150)
RING, GOOD, BAD, YELLOW = (90, 200, 250), (120, 210, 120), (90, 90, 230), (60, 210, 230)
TALK_TEXT = [
    "Read this out loud, the way you'd talk to a friend. Don't raise your eyebrows on purpose.",
    "Wait, really? You're telling me the bus left early again? That's the third time this week!",
    "Okay, so first open the browser, then search for the hackathon page, and click on schedule.",
    "Honestly I'm not sure. Maybe tomorrow? Or, oh, what about Sunday afternoon instead?",
    "Wow, that's amazing. I can't believe it actually worked on the first try. Let's do it again.",
]


def main() -> None:
    p = argparse.ArgumentParser(description="Eyebrow raise as a click: hits, drift and false clicks")
    p.add_argument("--camera", type=int, default=None)
    args = p.parse_args()

    from engine.eyes import os_input
    from engine.eyes.brow_gesture import BrowRaiseDetector
    from engine.eyes.camera import Camera
    from engine.eyes.cursor_driver import CursorDriver
    from engine.eyes.eye_features import EyeFeatureExtractor
    from engine.eyes.face_mesh import FaceMesh
    from engine.eyes.head_pose import HeadPoseEstimator
    from engine.eyes.pointer import Pointer

    os_input.make_dpi_aware()
    W, H = os_input.screen_size()
    cam = Camera(index=args.camera)
    cam.start()
    mesh, feats, head = FaceMesh(), EyeFeatureExtractor(), HeadPoseEstimator()
    pointer = Pointer(W, H, "head")
    brows = BrowRaiseDetector()
    driver = CursorDriver(os_input.move, still_px=pointer.still_px).start()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    scale = min(1.0, 1280 / W)
    w, h = int(W * scale), int(H * scale)

    freeze = False
    cursor = ((W - 1) / 2, (H - 1) / 2)    # where the cursor is (frozen or not)
    onset = None                           # cursor when the current raise started
    frozen_at = None
    flash_until = 0.0
    test = None    # {"buttons": [(x, y)], "i", "t0", "misses", "freeze"}
    talk = None    # {"t0", "fires"}
    results: list[dict] = []               # one per button
    talks: list[tuple[float, int]] = []    # (seconds, fires)
    log: list[tuple] = []                  # (t_ms, x, y, score, rest, raised, fired, phase)
    frame_id, face, note = 0, None, "Face the screen center and press C. SPACE = click test, T = talk test."
    was_raised = False
    try:
        while True:
            frame, t_ms, new_id = cam.wait_next(frame_id)
            fired = False
            if frame is not None and new_id != frame_id:
                frame_id = new_id
                face = mesh.process(frame, t_ms)
                if face is not None:
                    pose = head.compute(face)
                    pt = pointer.update(face.t_ms, feats.compute(face), pose)
                    fired = brows.update(face.t_ms, face.blendshapes or {}) is not None
                    raised = brows.raised
                    if raised and not was_raised:
                        onset = (pt.x, pt.y) if frozen_at is None else frozen_at
                        if freeze:
                            frozen_at = onset
                    if not raised and frozen_at is not None:
                        pointer.place(*frozen_at)     # carry on from the held spot, no jump back
                        frozen_at = None
                    was_raised = raised
                    cursor = frozen_at if frozen_at is not None else (pt.x, pt.y)
                    driver.set_target(*cursor)
                    phase = "click" if test else "talk" if talk else "-"
                    log.append((face.t_ms, round(cursor[0], 1), round(cursor[1], 1),
                                round(brows.score, 3) if brows.score is not None else "",
                                round(brows.rest, 3) if brows.rest is not None else "", int(raised), int(fired),
                                phase))
                else:
                    brows.update(t_ms, {})
            now = time.monotonic()
            if fired:
                flash_until = now + 0.4
            cx, cy = cursor

            # ---- tests
            if test is not None:
                bx, by = test["buttons"][test["i"]]
                inside = abs(cx - bx) <= BUTTON / 2 and abs(cy - by) <= BUTTON / 2
                done = timed_out = False
                if fired:
                    if inside:
                        done = True
                    else:
                        test["misses"] += 1
                timed_out = now - test["t0"] > TIMEOUT_S
                if done or timed_out:
                    drift = math.hypot(cx - onset[0], cy - onset[1]) if (done and onset) else None
                    on_at_onset = (onset is not None and abs(onset[0] - bx) <= BUTTON / 2
                                   and abs(onset[1] - by) <= BUTTON / 2)
                    r = {"freeze": test["freeze"], "hit": done, "secs": now - test["t0"], "misses": test["misses"],
                         "drift": drift, "on_at_onset": on_at_onset}
                    results.append(r)
                    print(f"  button {test['i'] + 1}: {'hit in %.1f s' % r['secs'] if done else 'timed out'}, "
                          f"misses {r['misses']}" + (f", cursor moved {drift:.0f} px during the raise" if drift is not None else ""))
                    test["i"] += 1
                    if test["i"] == len(test["buttons"]):
                        mine = [x for x in results if x["freeze"] == test["freeze"]][-N_BUTTONS:]
                        note = summary_line(mine, test["freeze"]) + "  (F toggles freeze, SPACE again)"
                        print("  " + note)
                        test = None
                    else:
                        test["t0"], test["misses"] = now, 0
            if talk is not None:
                talk["fires"] += int(fired)
                if now - talk["t0"] >= TALK_S:
                    talks.append((TALK_S, talk["fires"]))
                    note = f"Talk test: {talk['fires']} unwanted clicks in {TALK_S:.0f} s."
                    print("  " + note)
                    talk = None

            # ---- draw
            img = np.full((h, w, 3), BG, np.uint8)
            cv2.putText(img, f"freeze while raised: {'ON' if freeze else 'off'}   threshold {brows.threshold:.2f}   "
                             f"hold {brows.hold_ms} ms   (SPACE click test, T talk test, F freeze, C recenter, "
                             f"+/- threshold, [ ] hold, Q quit)", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT, 1,
                        cv2.LINE_AA)
            cv2.putText(img, note, (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, QUIET, 1, cv2.LINE_AA)
            draw_brow_bar(img, brows, flash=now < flash_until)
            if face is None:
                cv2.putText(img, "No face: face the camera", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BAD, 1)
            if test is not None:
                bx, by = test["buttons"][test["i"]]
                inside = abs(cx - bx) <= BUTTON / 2 and abs(cy - by) <= BUTTON / 2
                half = BUTTON / 2 * scale
                cv2.rectangle(img, (int(bx * scale - half), int(by * scale - half)),
                              (int(bx * scale + half), int(by * scale + half)), GOOD if inside else RING, 3, cv2.LINE_AA)
                cv2.putText(img, f"{test['i'] + 1}/{N_BUTTONS}", (int(bx * scale - 16), int(by * scale + 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT, 1, cv2.LINE_AA)
            if talk is not None:
                left = TALK_S - (now - talk["t0"])
                cv2.putText(img, f"Talk test: {left:.0f} s left, unwanted clicks so far: {talk['fires']}",
                            (60, h // 2 - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, YELLOW, 2, cv2.LINE_AA)
                for k, line in enumerate(TALK_TEXT):
                    cv2.putText(img, line, (60, h // 2 - 60 + 44 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.75, TEXT, 1,
                                cv2.LINE_AA)
            cv2.circle(img, (int(cx * scale), int(cy * scale)), 6, YELLOW if frozen_at else TEXT, -1, cv2.LINE_AA)

            cv2.imshow(WINDOW, img)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("c") and face is not None:
                head.set_neutral(face)
                pointer.reset()
                note = "Recentered."
            elif k == ord("f"):
                freeze = not freeze
                note = f"Freeze while raised: {'ON' if freeze else 'off'}"
            elif k in (ord("+"), ord("=")):
                brows.set_thresholds(threshold=min(brows.threshold + 0.05, 0.6))
            elif k == ord("-"):
                brows.set_thresholds(threshold=max(brows.threshold - 0.05, 0.05))
            elif k in (ord("["), ord("]")):
                brows.set_thresholds(hold_ms=min(max(brows.hold_ms + (50 if k == ord("]") else -50), 100), 800))
            elif k == ord(" ") and test is None and talk is None:
                buttons = [(random.uniform(0.1, 0.9) * W, random.uniform(0.12, 0.9) * H) for _ in range(N_BUTTONS)]
                test = {"buttons": buttons, "i": 0, "t0": now, "misses": 0, "freeze": freeze}
                note = f"Click test (freeze {'ON' if freeze else 'off'}): move onto the button, raise your eyebrows."
                print(f"Click test, freeze {'ON' if freeze else 'off'}, threshold {brows.threshold:.2f}, "
                      f"hold {brows.hold_ms} ms:")
            elif k == ord("t") and test is None and talk is None:
                talk = {"t0": now, "fires": 0}
                note = "Talk test: read the text out loud, naturally."
    finally:
        driver.stop()
        cam.stop()
        mesh.close()
        cv2.destroyAllWindows()
        if log:
            from pathlib import Path
            folder = Path(__file__).resolve().parents[1] / "calibration" / "brow_logs"
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / time.strftime("%Y%m%d-%H%M%S.csv")
            with open(path, "w", encoding="utf-8") as f:
                f.write("t_ms,x,y,score,rest,raised,fired,phase\n")
                f.writelines(",".join(map(str, row)) + "\n" for row in log)
            print(f"Log (numbers only): {path}")
        print("\n=== Eyebrow click ===")
        for fr in (False, True):
            mine = [x for x in results if x["freeze"] == fr]
            if mine:
                print("  " + summary_line(mine, fr))
        for secs, n in talks:
            print(f"  talk test: {n} unwanted clicks in {secs:.0f} s ({n * 60 / secs:.1f} per minute)")
        print(f"  values at the end: threshold {brows.threshold:.2f}, hold {brows.hold_ms} ms "
              f"(send these to Khang with the summary)")


def summary_line(rs: list[dict], freeze: bool) -> str:
    hits = [r for r in rs if r["hit"]]
    misses = sum(r["misses"] for r in rs)
    drifts = [r["drift"] for r in hits if r["drift"] is not None]
    secs = statistics.median([r["secs"] for r in hits]) if hits else float("nan")
    lost = sum(1 for r in rs if r["on_at_onset"] and r["misses"])
    return (f"freeze {'ON' if freeze else 'off'}: hit {len(hits)}/{len(rs)}, median {secs:.1f} s, "
            f"{misses} misses (raise outside the button)"
            + (f", cursor moved median {statistics.median(drifts):.0f} px / max {max(drifts):.0f} px during a raise"
               if drifts else "")
            + (f", {lost} buttons where the raise slid off" if lost else ""))


def draw_brow_bar(img, brows, flash: bool) -> None:
    """Score bar (0 to 1): white = resting level, yellow = raise threshold."""
    x0, y0, bw, bh = 20, 90, 300, 22
    cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), QUIET, 1)
    if brows.score is not None:
        cv2.rectangle(img, (x0, y0), (x0 + int(bw * min(brows.score, 1.0)), y0 + bh),
                      GOOD if brows.raised else (120, 120, 120), -1)
    if brows.rest is not None:
        rx = x0 + int(bw * min(brows.rest, 1.0))
        tx = x0 + int(bw * min(brows.rest + brows.threshold, 1.0))
        cv2.line(img, (rx, y0 - 4), (rx, y0 + bh + 4), TEXT, 2)
        cv2.line(img, (tx, y0 - 4), (tx, y0 + bh + 4), YELLOW, 2)
    label = "eyebrows" if brows.rest is not None else "eyebrows (learning your resting level...)"
    cv2.putText(img, label, (x0, y0 + bh + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, QUIET, 1, cv2.LINE_AA)
    if flash:
        cv2.putText(img, "CLICK", (x0 + bw + 20, y0 + bh - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, GOOD, 2, cv2.LINE_AA)


if __name__ == "__main__":
    main()
