"""Checks which modes the webcam really supports on this PC and how fast they actually run.
Nothing is saved. Close other apps that use the camera first. Takes 1 to 2 minutes.

Run:  .\\.venv\\Scripts\\python.exe -m tests.camera_probe
"""
import json
import time

from engine.eyes.camera import open_capture

MODES = [(1280, 720, 30), (1280, 720, 60), (1920, 1080, 30), (640, 480, 30)]
TRIES = [("dshow", "MJPG"), ("dshow", ""), ("msmf", "MJPG"), ("msmf", "")]
DARK = 10  # mean brightness below this usually means a virtual camera or a covered lens


def find_cameras(max_index: int = 5) -> list[int]:
    found = []
    for index in range(max_index):
        cap, _ = open_capture(index, "dshow", "", 640, 480, 30)
        if cap is not None:
            cap.release()
            found.append(index)
    return found


def measure(cap, seconds: float = 2.0) -> tuple[float, float]:
    for _ in range(5):  # let exposure settle
        cap.read()
    frames, brightness = 0, 0.0
    start = time.monotonic()
    while time.monotonic() - start < seconds:
        ok, frame = cap.read()
        if ok and frame is not None:
            frames += 1
            if frames % 10 == 1:
                brightness = float(frame.mean())
    return frames / (time.monotonic() - start), brightness


def pick_best(results: list[dict]) -> dict | None:
    usable = [r for r in results if r["fps"] >= 25 and r["brightness"] >= DARK]
    if not usable:
        usable = [r for r in results if r["brightness"] >= DARK] or results
        return max(usable, key=lambda r: r["fps"]) if usable else None
    hd = [r for r in usable if (r["width"], r["height"]) == (1280, 720)]
    pool = hd or usable
    # Prefer 720p near 30 fps (keeps MediaPipe fast); earlier entries in TRIES win ties.
    return min(pool, key=lambda r: (abs(r["fps"] - 30) > 5, -r["width"], r["order"]))


def main() -> None:
    print("Looking for cameras...")
    cameras = find_cameras()
    if not cameras:
        print("No camera opened. Check the USB cable, close apps using the camera, and check "
              "Windows Settings > Privacy & security > Camera (allow desktop apps).")
        raise SystemExit(1)
    print(f"Camera indexes that open: {cameras}\n")

    results = []
    order = 0
    for index in cameras:
        for backend, fourcc in TRIES:
            for width, height, fps in MODES:
                label = f"camera {index}  {backend:5s} {fourcc or 'auto':4s}  ask {width}x{height}@{fps}"
                print(label, end="  ->  ", flush=True)
                cap, mode = open_capture(index, backend, fourcc, width, height, fps)
                if cap is None:
                    print("won't open")
                    continue
                try:
                    real_fps, brightness = measure(cap)
                finally:
                    cap.release()
                note = "  (dark: virtual camera or covered lens?)" if brightness < DARK else ""
                print(f"got {mode.width}x{mode.height} {mode.fourcc}, {real_fps:.1f} fps measured{note}")
                results.append({"index": index, "backend": backend, "fourcc": fourcc,
                                "width": mode.width, "height": mode.height,
                                "fps": round(real_fps, 1), "brightness": brightness, "order": order})
                order += 1

    best = pick_best(results)
    print("\n=== Recommendation ===")
    if best is None:
        print("No mode delivered frames.")
        return
    fps_setting = 60 if best["fps"] > 45 else 30
    settings = {"camera_index": best["index"], "camera_backend": best["backend"],
                "camera_fourcc": best["fourcc"], "frame_width": best["width"],
                "frame_height": best["height"], "fps": fps_setting}
    print(f"camera {best['index']}, {best['backend']} {best['fourcc'] or 'auto'}, "
          f"{best['width']}x{best['height']} at {best['fps']} fps measured")
    if best["fps"] < 25:
        print("Warning: under 25 fps. Try more light (exposure gets longer in the dark) or another USB port.")
    print("Settings:")
    print(json.dumps(settings))
    print("\nOn the demo PC, these become the defaults in engine/config.py. "
          "On any other laptop, put them in config.local.json.")


if __name__ == "__main__":
    main()
