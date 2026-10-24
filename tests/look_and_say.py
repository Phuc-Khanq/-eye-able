"""Look and say, end to end: look at a thing, press Ctrl, UI-Venus finds it, nod to click.
Nothing is saved: camera frames and screenshots stay in memory.

Run:  .\\.venv\\Scripts\\python.exe -m tests.look_and_say
      .\\.venv\\Scripts\\python.exe -m tests.look_and_say --labels "channel avatar,search box,subscribe button"
      .\\.venv\\Scripts\\python.exe -m tests.look_and_say --auto        (click as soon as it's found, no nod)
Calibrate first (python -m tests.gaze_dot --profile khang --quick) and sit the same way.

How to use it:
  1. Look at the center of the screen and press N (zeroes the head, starts the gaze cursor).
  2. Press a number (1-9) to pick what you'll ask for; looking at the keyboard is fine for this.
  3. Look at the thing and tap Ctrl (either one, on its own) or F9, without looking down.
     Voice replaces this later. UI-Venus searches around where you were looking; the cursor jumps there.
  4. Nod to click it: chin down about 5 degrees and at least halfway back up within 2 s. X cancels; no nod in 10 s
     releases it.
Smoothness: the cursor ignores wobble under --still-px (8) and glides between camera frames at 120 updates a second
(--glide-ms 40). The panel shows "cursor wobble": hold your head still and read it; under 3 px is steady.
The status panel (bottom right, always on top) shows the state, your head angles, the nod meter and
every gesture the head detector fires, so you can see whether head tracking and nods are being read.
Between requests the gaze moves the cursor in jumps and small head turns fine-tune it (20 px per degree).
A nod with nothing found does nothing (looking down at the keyboard can look like a nod); --plain-clicks changes that.
Pointer modes (H switches live, --pointer picks the start):
  head    the head points: turn about 15 degrees to reach a side edge, tilt about 10 degrees for top or bottom.
          Steady and predictable; the eyes aren't used. C re-zeroes the head (look at the center first).
  hybrid  gaze jumps the cursor when you look more than --jump-px away, small head turns fine-tune it.
  gaze    gaze only, no head fine-tuning.
UI-Venus searches around the cursor in every mode, so the pointer only has to get close.
Keys work from any window. C = recenter, P = pause, Q = quit. On macOS, Ctrl is the Control key.
"""
import argparse
import queue
import time
from collections import deque

import numpy as np

from tests.gaze_cursor import GazeCursor, hide_mark, show_mark

DEFAULT_LABELS = ["channel avatar", "search box", "subscribe button", "like button", "settings icon"]
PANEL = "Diseyeble status"
PANEL_W, PANEL_H = 420, 230
MODES = ("head", "hybrid", "gaze")
BG, TEXT, QUIET, GOOD, WARN, BAD, ACCENT = (32, 28, 26), (235, 235, 235), (150, 150, 150), (120, 210, 120), (60, 170, 240), (90, 90, 230), (240, 200, 90)


def start_keys(keys: "queue.Queue[str]"):
    """Global keys. Ctrl counts only when tapped alone, so Ctrl+C and friends don't ask UI-Venus."""
    from pynput import keyboard

    ctrl = {keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r}
    held = {"ctrl": False, "other": False}

    def on_press(key, injected=False):
        if injected:  # typed by os_input, not by a person
            return
        if key in ctrl or getattr(key, "vk", None) in (17, 162, 163):
            if not held["ctrl"]:
                held["ctrl"], held["other"] = True, False
            return
        held["other"] = True
        if key == keyboard.Key.f9:
            keys.put("ask")
            return
        ch = getattr(key, "char", None)
        if ch and ch.lower() in "npcxqh123456789" and not held["ctrl"]:
            keys.put(ch.lower())

    def on_release(key, injected=False):
        if injected:
            return
        if (key in ctrl or getattr(key, "vk", None) in (17, 162, 163)) and held["ctrl"]:
            if not held["other"]:
                keys.put("ask")
            held["ctrl"] = False

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    return listener


def put(img, s, x, y, color=TEXT, scale=0.5, thick=1):
    import cv2
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_panel(info: dict) -> np.ndarray:
    import cv2
    img = np.full((PANEL_H, PANEL_W, 3), BG, dtype=np.uint8)
    state, color = info["state"]
    cv2.rectangle(img, (0, 0), (PANEL_W, 44), color, -1)
    put(img, state, 12, 30, (20, 20, 20), 0.75, 2)
    # the label itself stays out of the panel: UI-Venus could pick the panel's own text instead of the page's
    put(img, f"asking for item {info['item']} (see terminal; 1-9 to change, Ctrl to ask)", 12, 66, TEXT, 0.45)
    pose = info["pose"]
    if pose is None:
        put(img, "No face found: face the camera", 12, 94, BAD, 0.5, 1)
    else:
        put(img, f"{info['mode']} pointer   yaw {pose.yaw:+5.1f}  pitch {pose.pitch:+5.1f}  roll {pose.roll:+5.1f}",
            12, 94, TEXT, 0.48)
    # nod meter: how far the chin went down in the last second vs. what a nod needs
    need, dip = info["nod_need"], info["dip"]
    x0, y0, w = 12, 112, PANEL_W - 24
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + 18), (60, 60, 60), -1)
    frac = max(0.0, min(dip / (2 * need), 1.0))
    cv2.rectangle(img, (x0, y0), (x0 + int(frac * w), y0 + 18), GOOD if dip >= need else WARN, -1)
    mx = x0 + w // 2
    cv2.line(img, (mx, y0 - 3), (mx, y0 + 21), TEXT, 2)
    put(img, f"chin down in the last second: {dip:.1f} deg (nod needs {need:.0f})", 12, y0 + 40, QUIET, 0.45)
    put(img, f"last gesture: {info['gesture']}", 12, y0 + 64, ACCENT if info["gesture_new"] else QUIET, 0.5)
    put(img, f"cursor wobble {info['wobble']:.1f} px   shift ({info['shift'][0]:+.0f}, {info['shift'][1]:+.0f})   "
             f"clicks {info['clicks']}   {info['fps']:.0f} fps", 12, y0 + 90, QUIET, 0.45)
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description="Look and say: gaze + UI-Venus + nod")
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--iris", choices=["center", "ring"], default="center")
    parser.add_argument("--model", default=None, help="gaze model file (default: calibration/gaze_model.json)")
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS), help="comma list; label n is key n")
    parser.add_argument("--auto", action="store_true", help="click when found, no nod")
    parser.add_argument("--no-learn", action="store_true", help="don't correct the gaze from confirmed clicks")
    parser.add_argument("--no-zoom", action="store_true", help="skip the small zoom box, use the wide box only")
    parser.add_argument("--confirm-deg", type=float, default=5.0, help="chin-down degrees that confirm a found element")
    parser.add_argument("--still-px", type=float, default=8.0, help="cursor ignores wobble smaller than this (0 = off)")
    parser.add_argument("--glide-ms", type=float, default=40.0, help="cursor glides to each new point over about this long (0 = hop)")
    parser.add_argument("--confirm-s", type=float, default=10.0, help="how long a found element waits for a nod")
    parser.add_argument("--pointer", choices=MODES, default="head", help="what moves the cursor (H switches live)")
    parser.add_argument("--head-sens", type=float, default=1.0,
                        help="head pointer speed: 1.0 = 15 deg reaches a side edge; 0.7 = calmer, 1.3 = less turning")
    parser.add_argument("--jump-px", type=float, default=250.0,
                        help="gaze must land this far from the cursor to move it; below that, the head fine-tunes")
    parser.add_argument("--plain-clicks", action="store_true", help="a head-detector nod with nothing found clicks at the cursor")
    parser.add_argument("--no-panel", action="store_true", help="hide the status panel")
    args = parser.parse_args()
    labels = [s.strip() for s in args.labels.split(",") if s.strip()][:9]

    import cv2

    from engine.eyes import os_input
    from engine.eyes.calibration import DEFAULT_PATH, GazeModel
    from engine.eyes.camera import Camera
    from engine.eyes.eye_features import EyeFeatureExtractor
    from engine.eyes.face_mesh import FaceMesh
    from engine.eyes.head_pose import HeadPoseEstimator
    from engine.eyes.look_click import ConfirmNod, LookClick
    from engine.eyes.screen_target import ScreenTarget

    os_input.make_dpi_aware()
    W, H = os_input.screen_size()
    from engine.config import settings
    settings.sensitivity = args.head_sens  # HeadPointer reads it every frame
    try:
        model = GazeModel.load(args.model)
    except FileNotFoundError:
        if args.pointer != "head":
            print(f"No gaze model at {args.model or DEFAULT_PATH}. Calibrate first: python -m tests.gaze_dot --quick")
            return
        model = None
        print("No gaze model yet: head pointer only.")

    finder = ScreenTarget(zoom=None if args.no_zoom else (640, 360))
    print("Loading UI-Venus...", end=" ", flush=True)
    print(f"{finder.load():.0f} s on {finder.device}")
    from PIL import Image
    finder.ask(Image.new("RGB", (640, 360), (255, 255, 255)), "the button")  # warm-up, first call is slow

    cam = Camera(index=args.camera, fps=args.fps)
    mode = cam.start()
    print(f"Screen {W}x{H}. Camera {mode.width}x{mode.height} via {mode.backend}.")
    mesh, eyes, head = FaceMesh(), EyeFeatureExtractor(iris_mode=args.iris), HeadPoseEstimator()
    app = GazeCursor(model, W, H, dwell=False)
    mode_name = args.pointer

    def set_mode(name: str) -> None:
        nonlocal mode_name
        if name != "head" and model is None:
            print("No gaze model: calibrate first (python -m tests.gaze_dot --quick). Staying on head.")
            return
        mode_name = name
        lc.learn = name != "head" and not args.no_learn  # the head cursor goes where you point; nothing to learn
        app.pointer.jump_px = 0.0 if name == "gaze" else args.jump_px
        app.pointer.clear_anchor()   # head mode: neutral pose = screen center; others: next gaze jumps
        app.smoother.reset()
        lc.shift = (0.0, 0.0)        # a learned shift belongs to one pointer
        print(f"Pointer: {name}")
    lc = LookClick(finder, W, H, auto=args.auto, learn=not args.no_learn, confirm_s=args.confirm_s)
    yes = ConfirmNod(deg=args.confirm_deg)
    set_mode(mode_name)
    selected, last_state = 0, "idle"
    pitch_hist: deque[tuple[float, float]] = deque()
    gesture_text, gesture_at = "none yet", 0.0

    if not args.no_panel:
        cv2.namedWindow(PANEL, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(PANEL, W - PANEL_W - 20, H - PANEL_H - 90)
        try:
            cv2.setWindowProperty(PANEL, cv2.WND_PROP_TOPMOST, 1)
        except cv2.error:
            pass

    keys: queue.Queue[str] = queue.Queue()
    listener = start_keys(keys)
    print("Pick with: " + ", ".join(f"{i + 1} = {l}" for i, l in enumerate(labels)))
    print("Look at the center of the screen and press N to start. Then point at the thing and tap Ctrl (or F9).")
    print("H switches pointer (head / hybrid / gaze), C recenters, X cancels, P pauses, Q quits.")

    from engine.eyes.cursor_driver import CursorDriver
    driver = CursorDriver(os_input.move, glide_ms=args.glide_ms, still_px=args.still_px).start()

    def do_click(x, y, how):
        driver.click(x, y, os_input.click)
        print(f"click at ({x}, {y}) by {how}   gaze shift now ({lc.shift[0]:+.0f}, {lc.shift[1]:+.0f}) px")

    frame_id, face, pose, held_pose = 0, None, None, None
    from engine.core.types import HeadPose
    try:
        while True:
            now = time.monotonic()
            while not keys.empty():
                k = keys.get()
                if k == "q":
                    return
                if k == "n" and face is not None:
                    head.set_neutral(face)
                    app.restart()
                    app.set_paused(False)
                    print("Head zeroed. Gaze cursor on.")
                elif k == "n":
                    print("No face in view; face the camera and press N again.")
                elif k == "p":
                    app.set_paused(not app.paused)
                    print("Paused." if app.paused else "Cursor on.")
                elif k == "x":
                    msg = lc.cancel()
                    if msg:
                        print(msg)
                elif k == "h":
                    set_mode(MODES[(MODES.index(mode_name) + 1) % len(MODES)])
                elif k == "c" and mode_name == "head":
                    if face is not None:
                        head.set_neutral(face)
                        app.restart()
                        print("Head re-zeroed: where you face now is the screen center.")
                elif k == "c" and app.recentering:
                    app.cancel_recenter()
                    hide_mark()
                elif k == "c":
                    app.start_recenter()
                    show_mark(W, H)
                    print("Recenter: look at the dot in the middle, then nod.")
                elif k.isdigit():
                    if int(k) - 1 < len(labels):
                        selected = int(k) - 1
                        print(f"Asking for: {labels[selected]}")
                elif k == "ask":
                    if app.paused or app.recentering:
                        print("Ctrl heard, but the cursor is off: look at the center and press N first.")
                    else:
                        print(lc.request(labels[selected], now))

            for act in lc.poll(now):
                if act[0] == "say":
                    print(act[1])
                    if act[1].startswith("No nod"):
                        print(f"   deepest chin-down while waiting: {yes.deepest:.1f} deg (needed {yes.deg:.0f}). "
                              f"Try --confirm-deg {max(3, int(yes.deepest * 0.8))} if your nods are smaller.")
                elif act[0] == "move":
                    driver.snap(act[1], act[2])
                elif act[0] == "click":
                    do_click(act[1], act[2], "auto")
            if lc.state == "confirm" and last_state != "confirm":
                yes.arm()
            last_state = lc.state

            frame, t_ms, frame_id = cam.wait_next(frame_id)
            if frame is None:
                continue
            face = mesh.process(frame, t_ms)
            if face is None:
                pose = None
            else:
                feats, pose = eyes.compute(face), head.compute(face)
                # a blink can nudge MediaPipe's head angles; the head pointer holds its last open-eye pose
                if feats.usable or held_pose is None:
                    held_pose = pose
                point_pose = pose if feats.usable else HeadPose(pose.t_ms, held_pose.yaw, held_pose.pitch, held_pose.roll)
                was_recentering = app.recentering
                gestures_before = app.gestures.quiet_until_ms
                cursor, click = app.step(face.t_ms, None if mode_name == "head" else feats, point_pose)
                if app.gestures.quiet_until_ms != gestures_before:  # the head detector fired something
                    gesture_text, gesture_at = ("nod" if click is not None else "gesture") + f" at {time.strftime('%H:%M:%S')}", now
                for msg in app.events:
                    print(msg)
                app.events.clear()
                if was_recentering and not app.recentering:
                    hide_mark()
                pitch_hist.append((now, pose.pitch))
                while pitch_hist and now - pitch_hist[0][0] > 1.0:
                    pitch_hist.popleft()
                confirmed = yes.update(face.t_ms, pose.pitch)

                if lc.state == "confirm" and (confirmed or (click is not None and click.how == "nod")):
                    act = lc.confirm()
                    if act:
                        do_click(act[1], act[2], "nod")
                elif lc.busy:
                    if click is not None and lc.state == "searching":
                        print("   nod came before UI-Venus answered; nod again once the cursor jumps.")
                else:
                    if cursor is not None:
                        lc.add_gaze(now, *cursor)
                        driver.set_target(*lc.shifted(*cursor))
                    if click is not None and args.plain_clicks:
                        x, y = lc.shifted(click.x, click.y)
                        do_click(int(round(x)), int(round(y)), "plain nod")
                    elif click is not None:
                        print("   (nod seen, nothing to click: look at a thing and tap Ctrl first)")

            if not args.no_panel:
                if app.paused:
                    state = ("PRESS N TO START", QUIET)
                elif app.recentering:
                    state = ("RECENTER: LOOK AT DOT, NOD", ACCENT)
                elif lc.state == "searching":
                    state = ("SEARCHING...", WARN)
                elif lc.state == "confirm":
                    state = (f"NOD TO CLICK ({max(0, lc._deadline - now):.0f} s)", GOOD)
                else:
                    state = ("READY: POINT, THEN CTRL", TEXT)
                ps = [p for _, p in pitch_hist]
                dip = (max(ps) - min(ps[:int(np.argmax(ps)) + 1])) if ps else 0.0
                cv2.imshow(PANEL, draw_panel({
                    "state": state, "mode": mode_name, "item": selected + 1, "pose": pose, "nod_need": yes.deg, "dip": dip,
                    "gesture": gesture_text, "gesture_new": now - gesture_at < 2.0, "shift": lc.shift,
                    "clicks": lc.stats["clicked"], "fps": cam.measured_fps, "wobble": driver.wobble_px()}))
            if not args.no_panel or app.recentering:
                cv2.waitKey(1)  # draws the panel and the recenter mark
    finally:
        driver.stop()
        listener.stop()
        cam.stop()
        mesh.close()
        finder.close()
        print(f"Done. {lc.stats}")


if __name__ == "__main__":
    main()
