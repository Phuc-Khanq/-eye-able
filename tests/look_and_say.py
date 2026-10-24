"""Look and say, end to end: look at a thing, press Right Ctrl, UI-Venus finds it, nod to click.
Nothing is saved: camera frames and screenshots stay in memory.

Run:  .\\.venv\\Scripts\\python.exe -m tests.look_and_say
      .\\.venv\\Scripts\\python.exe -m tests.look_and_say --labels "channel avatar,search box,subscribe button"
      .\\.venv\\Scripts\\python.exe -m tests.look_and_say --auto        (click as soon as it's found, no nod)
Calibrate first (python -m tests.gaze_dot --profile khang --quick) and use the same camera setup.

How to use it:
  1. Look at the center of the screen and press N (zeroes the head, starts the gaze cursor).
  2. Open the page you want (a YouTube channel page works well).
  3. Press a number (1-9) to pick what you'll ask for; you can look at the keyboard for this.
  4. Look at the thing and press Right Ctrl without looking down (voice replaces this later).
     UI-Venus searches the box around where you were looking (zoom, then wide) and the cursor jumps to it.
  5. Nod to click it: chin down and back up, quick, from a still head. X cancels. No nod in 10 s releases it.
     If no nod is seen, the terminal says how far your head dipped, so you can tune --nod-deg.
Nods only click a found element here; a nod at other times does nothing (use --plain-clicks to change that),
because looking down at the keyboard can look like a nod.
Every confirmed click also corrects the gaze (see engine/eyes/look_click.py), so accuracy improves as you use it.
Keys work from any window. C = recenter, P = pause, Q = quit. Windows and macOS (Right Ctrl = right Control) both work.
"""
import argparse
import queue
import time

from tests.gaze_cursor import GazeCursor, hide_mark, show_mark

DEFAULT_LABELS = ["channel avatar", "search box", "subscribe button", "like button", "settings icon"]


def start_keys(keys: "queue.Queue[str]"):
    from pynput import keyboard

    def on_press(key, injected=False):
        if injected:  # typed by os_input, not by a person
            return
        if key == keyboard.Key.ctrl_r:
            keys.put("ask")
            return
        ch = getattr(key, "char", None)
        if ch and ch.lower() in "npcxq123456789":
            keys.put(ch.lower())

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener


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
    parser.add_argument("--nod-deg", type=float, default=None, help="how far a nod must dip (default: head_gestures.NOD_DEG)")
    parser.add_argument("--confirm-s", type=float, default=10.0, help="how long a found element waits for a nod")
    parser.add_argument("--plain-clicks", action="store_true", help="a nod with nothing found clicks at the gaze cursor")
    args = parser.parse_args()
    labels = [s.strip() for s in args.labels.split(",") if s.strip()][:9]

    from engine.eyes import os_input
    from engine.eyes.calibration import DEFAULT_PATH, GazeModel
    from engine.eyes.camera import Camera
    from engine.eyes.eye_features import EyeFeatureExtractor
    from engine.eyes.face_mesh import FaceMesh
    from engine.eyes.head_pose import HeadPoseEstimator
    from engine.eyes.look_click import LookClick
    from engine.eyes.screen_target import ScreenTarget

    os_input.make_dpi_aware()
    W, H = os_input.screen_size()
    try:
        model = GazeModel.load(args.model)
    except FileNotFoundError:
        print(f"No gaze model at {args.model or DEFAULT_PATH}. Calibrate first: python -m tests.gaze_dot --quick")
        return

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
    if args.nod_deg is not None:
        app.gestures.set_thresholds(nod_deg=args.nod_deg)
    lc = LookClick(finder, W, H, auto=args.auto, learn=not args.no_learn, confirm_s=args.confirm_s)
    selected = 0
    dip = None  # (pitch when the element was found, lowest, highest) while waiting for a nod

    keys: queue.Queue[str] = queue.Queue()
    listener = start_keys(keys)
    print("Pick with: " + ", ".join(f"{i + 1} = {l}" for i, l in enumerate(labels)))
    print(f"Asking for: {labels[0]}. A nod needs {app.gestures.nod_deg:.0f} degrees down and back.")
    print("Look at the center of the screen and press N to start. Then look at the thing and press Right Ctrl.")

    frame_id, face = 0, None
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
                        print("Press N first (cursor is off).")
                    else:
                        print(lc.request(labels[selected], now))

            for act in lc.poll(now):
                if act[0] == "say":
                    print(act[1])
                    if act[1].startswith("No nod") and dip is not None:
                        print(f"   your head moved {dip[0] - dip[1]:.1f} deg up and {dip[2] - dip[0]:.1f} deg down meanwhile; "
                              f"a nod needs {app.gestures.nod_deg:.0f} deg down and back within 0.6 s. Try --nod-deg "
                              f"{max(5, round((dip[2] - dip[0]) * 0.8))} if your nods are smaller." if dip[2] - dip[0] < app.gestures.nod_deg
                              else f"   your head dipped {dip[2] - dip[0]:.1f} deg, enough, but too slowly or without a still start.")
                    dip = None
                elif act[0] == "move":
                    os_input.move(act[1], act[2])
                elif act[0] == "click":
                    os_input.click(act[1], act[2])
                    print(f"click at ({act[1]}, {act[2]})   shift now ({lc.shift[0]:.0f}, {lc.shift[1]:.0f}) px   "
                          f"{lc.stats['clicked']} clicks, {lc.stats['missed']} not found")

            frame, t_ms, frame_id = cam.wait_next(frame_id)
            if frame is None:
                continue
            face = mesh.process(frame, t_ms)
            if face is None:
                continue
            feats, pose = eyes.compute(face), head.compute(face)
            was_recentering = app.recentering
            cursor, click = app.step(face.t_ms, feats, pose)
            for msg in app.events:
                print(msg)
            app.events.clear()
            if was_recentering and not app.recentering:
                hide_mark()
            elif app.recentering:
                cv2_wait()
            if lc.state == "confirm" and pose is not None:
                dip = (pose.pitch, pose.pitch, pose.pitch) if dip is None else (dip[0], min(dip[1], pose.pitch), max(dip[2], pose.pitch))
            if lc.busy:
                if click is not None and click.how == "nod":
                    if lc.state == "searching":
                        print("   nod came before UI-Venus answered; nod again once the cursor jumps.")
                    act = lc.confirm()
                    dip = None
                    if act:
                        os_input.click(act[1], act[2])
                        print(f"click at ({act[1]}, {act[2]})   shift now ({lc.shift[0]:.0f}, {lc.shift[1]:.0f}) px   "
                              f"{lc.stats['clicked']} clicks, {lc.stats['missed']} not found")
                continue  # cursor stays on the found element
            if cursor is not None:
                lc.add_gaze(now, *cursor)
                os_input.move(*lc.shifted(*cursor))
            if click is not None and args.plain_clicks:
                x, y = lc.shifted(click.x, click.y)
                os_input.click(int(round(x)), int(round(y)))
                print(f"plain click at ({x:.0f}, {y:.0f})")
            elif click is not None:
                print("   (nod seen, nothing to click: look at a thing and press Right Ctrl first)")
    finally:
        listener.stop()
        cam.stop()
        mesh.close()
        finder.close()
        print(f"Done. {lc.stats}")


def cv2_wait() -> None:
    import cv2
    cv2.waitKey(1)  # keep the recenter mark drawn


if __name__ == "__main__":
    main()
