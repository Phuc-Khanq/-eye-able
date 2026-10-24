"""Live check for the mouth signal: does "moving" follow your talking and stay off while you're quiet?
Nothing is saved. Run:  python -m tests.mouth_check   (add --camera 1 if the picture is black)

Three 6-second phases, each announced in the terminal:
  1. quiet     face the screen, mouth closed, don't talk
  2. talk      read something aloud at your normal pace
  3. listen    stay quiet while a teammate talks next to you (or just stay quiet)
Pass: talking reads at least 0.5, the quiet phases at most 0.15. The live line shows jawOpen and the
moving flag 4 times a second. If it fails, change MOVE_RANGE / OPEN_ABOVE_REST in engine/eyes/mouth.py.
"""
import argparse
import threading
import time

from engine.core.types import MouthSample
from engine.eyes.mouth import MouthTracker

PHASES = [("quiet", "mouth closed, don't talk"), ("talk", "read something aloud"),
          ("listen", "stay quiet while a teammate talks")]
PHASE_S = 6.0


def main() -> None:
    p = argparse.ArgumentParser(description="Live check of the mouth-moving signal")
    p.add_argument("--camera", type=int, default=None)
    args = p.parse_args()

    from engine.eyes.camera import Camera
    from engine.eyes.face_mesh import FaceMesh

    cam = Camera(index=args.camera)
    mode = cam.start()
    print(f"Camera {mode.width}x{mode.height}. Face the camera.")
    mesh, mouth = FaceMesh(), MouthTracker()
    stop = threading.Event()
    last = {"open": None}

    def loop():
        frame_id = 0
        while not stop.is_set():
            frame, t_ms, new_id = cam.wait_next(frame_id)
            if frame is None or new_id == frame_id:
                continue
            frame_id = new_id
            face = mesh.process(frame, t_ms)
            if face is not None:
                o = float(face.blendshapes.get("jawOpen", 0.0))
                mouth.add(MouthSample(t_ms=face.t_ms, open=o))
                last["open"] = o
            else:
                last["open"] = None

    threading.Thread(target=loop, daemon=True).start()
    results = {}
    try:
        time.sleep(1.5)
        for name, what in PHASES:
            for s in (3, 2, 1):
                print(f"\nNext: {name.upper()} ({what}) in {s}...", end="", flush=True)
                time.sleep(1)
            print(f"\n>>> {name.upper()}: {what}")
            t0 = int(time.monotonic() * 1000)
            end = time.monotonic() + PHASE_S
            while time.monotonic() < end:
                now = int(time.monotonic() * 1000)
                f = mouth.moving_fraction(now - 500, now)
                o = last["open"]
                bar = "#" * int((o or 0) * 40)
                print(f"\r  jawOpen {o if o is not None else float('nan'):4.2f} {bar:<40} moving(0.5 s) "
                      f"{'n/a ' if f is None else f'{f:4.2f}'}", end="", flush=True)
                time.sleep(0.25)
            t1 = int(time.monotonic() * 1000)
            results[name] = mouth.moving_fraction(t0 + 300, t1)
        print("\n\n=== Result ===")
        for name, _ in PHASES:
            f = results[name]
            print(f"  {name:<7} moving {('no face' if f is None else f'{f:.2f}')}")
        ok = (results["talk"] or 0) >= 0.5 and all((results[n] if results[n] is not None else 1) <= 0.15
                                                    for n in ("quiet", "listen"))
        print("PASS: the mouth signal can tell your speech from others'" if ok else
              "NOT YET: send these numbers to Khang (thresholds are in engine/eyes/mouth.py)")
    finally:
        stop.set()
        cam.stop()
        mesh.close()


if __name__ == "__main__":
    main()
