"""Live check for the mouth right click: does open-and-close fire, and do talking and yawning not?
No clicks are made and nothing is saved. Run:  python -m tests.mouth_click_check   (--camera 1 if black)

Three 10-second phases, each announced in the terminal:
  1. click     open your mouth wide and close it again (about half a second), 5 times
  2. talk      read something aloud at your normal pace
  3. yawn      yawn, or hold your mouth open for 3 seconds, twice
Pass: 4 or 5 fire in "click", none in "talk" and "yawn". The live line shows jawOpen over the resting
level. If it fails, change OPEN_ABOVE / MIN_OPEN_MS / MAX_OPEN_MS in engine/eyes/mouth_gesture.py.
(In the engine, a mouth gesture is also ignored while the mic hears speech, so "talk" is stricter here.)
"""
import argparse
import threading
import time

from engine.eyes.mouth_gesture import MouthOpenDetector

PHASES = [("click", "open your mouth wide and close it, 5 times", (4, 5)),
          ("talk", "read something aloud", (0, 0)),
          ("yawn", "yawn, or hold your mouth open 3 s, twice", (0, 0))]
PHASE_S = 10.0


def main() -> None:
    p = argparse.ArgumentParser(description="Live check of the mouth right click")
    p.add_argument("--camera", type=int, default=None)
    args = p.parse_args()

    from engine.eyes.camera import Camera
    from engine.eyes.face_mesh import FaceMesh

    cam = Camera(index=args.camera)
    mode = cam.start()
    print(f"Camera {mode.width}x{mode.height}. Face the camera.")
    mesh, det = FaceMesh(), MouthOpenDetector()
    stop = threading.Event()
    fired = []

    def loop():
        frame_id = 0
        while not stop.is_set():
            frame, t_ms, new_id = cam.wait_next(frame_id)
            if frame is None or new_id == frame_id:
                continue
            frame_id = new_id
            face = mesh.process(frame, t_ms)
            if det.update(face.t_ms if face else t_ms, face.blendshapes if face else {}) is not None:
                fired.append(time.monotonic())
                print("   <- RIGHT CLICK (none made)", flush=True)

    threading.Thread(target=loop, daemon=True).start()
    counts = {}
    try:
        time.sleep(2.0)   # the resting level
        for name, what, _ in PHASES:
            for s in (3, 2, 1):
                print(f"\nNext: {name.upper()} ({what}) in {s}...", end="", flush=True)
                time.sleep(1)
            print(f"\n>>> {name.upper()}: {what}")
            t0, end = time.monotonic(), time.monotonic() + PHASE_S
            while time.monotonic() < end:
                over = (det.jaw or 0.0) - (det.rest or 0.0)
                print(f"\r  jawOpen {det.jaw if det.jaw is not None else float('nan'):4.2f}  over rest "
                      f"{over:+.2f} {'#' * max(0, int(over * 40)):<40}", end="", flush=True)
                time.sleep(0.1)
            counts[name] = sum(t0 <= t < end for t in fired)
        print("\n\n=== Result ===")
        ok = True
        for name, _, (lo, hi) in PHASES:
            good = lo <= counts[name] <= hi
            ok &= good
            print(f"  {name:<6} fired {counts[name]}  (want {lo}{'' if lo == hi else f'-{hi}'})  {'ok' if good else 'X'}")
        print("PASS" if ok else "FAIL: paste this to Khang's chat")
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        cam.stop()
        mesh.close()


if __name__ == "__main__":
    main()
