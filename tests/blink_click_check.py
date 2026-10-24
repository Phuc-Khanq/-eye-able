"""Live check for the long-blink click: does a deliberate blink fire, and do normal blinking, reading and talking not?
No clicks are made and nothing is saved. Run:  python -m tests.blink_click_check   (--camera 1 if black)

Four phases, each announced in the terminal:
  1. click    close BOTH eyes on purpose for about half a second (count "one-one-thousand"), then open. 5 times
  2. blink    just blink normally, or even quickly twice in a row, for 15 seconds
  3. read     read a paragraph on screen for 20 seconds
  4. talk     read something aloud for 10 seconds
Pass: 4 or 5 fire in "click", none in the other three. The live line shows the lower of the two eye scores
(open is about 0.1, shut about 0.8) and how long they have been shut.
If it fails: too few fire in "click" -> lower MIN_CLOSED_MS (try 350) or CLOSE_ABOVE (try 0.45) in
engine/eyes/blink_gesture.py; fires while reading or talking -> raise MIN_CLOSED_MS (try 450).
Then turn it on in the engine with  python -m engine.main --blink  (or "blink_click": true in config.local.json).
"""
import argparse
import threading
import time

from engine.eyes.blink_gesture import BlinkDetector

PHASES = [("click", "close both eyes on purpose for half a second, then open, 5 times", 10.0, (4, 5)),
          ("blink", "blink normally, even twice in a row", 15.0, (0, 0)),
          ("read", "read the screen like you normally do", 20.0, (0, 0)),
          ("talk", "read something aloud", 10.0, (0, 0))]


def main() -> None:
    p = argparse.ArgumentParser(description="Live check of the long-blink click")
    p.add_argument("--camera", type=int, default=None)
    args = p.parse_args()

    from engine.eyes.camera import Camera
    from engine.eyes.face_mesh import FaceMesh

    cam = Camera(index=args.camera)
    mode = cam.start()
    print(f"Camera {mode.width}x{mode.height}. Face the camera.")
    from engine.eyes.head_pose import HeadPoseEstimator

    mesh, det, head = FaceMesh(), BlinkDetector(), HeadPoseEstimator()
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
            pitch = head.compute(face).pitch if face is not None else None
            if det.update(face.t_ms if face else t_ms, face.blendshapes if face else {}, pitch) is not None:
                fired.append(time.monotonic())
                print("\n   <- CLICK (none made)", flush=True)

    threading.Thread(target=loop, daemon=True).start()
    counts = {}
    try:
        time.sleep(2.0)
        for name, what, secs, _ in PHASES:
            for s in (3, 2, 1):
                print(f"\nNext: {name.upper()} ({what}) in {s}...", end="", flush=True)
                time.sleep(1)
            print(f"\n>>> {name.upper()}: {what}")
            t0, end = time.monotonic(), time.monotonic() + secs
            while time.monotonic() < end:
                sc = det.score if det.score is not None else float("nan")
                print(f"\r  eyes shut {sc:4.2f} {'#' * max(0, int(sc * 30)):<30} open level {det.open_level:4.2f}"
                      f"  closed {det.closed_ms:5d} ms",
                      end="", flush=True)
                time.sleep(0.1)
            counts[name] = sum(t0 <= t < end for t in fired)
        print("\n\n=== Result ===")
        ok = True
        for name, _, _, (lo, hi) in PHASES:
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
