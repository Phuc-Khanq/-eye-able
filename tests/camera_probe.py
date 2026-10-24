"""Checks which modes the webcam really supports on this PC and how fast they actually run.
Nothing is saved. Close other apps that use the camera first. Takes 1 to 2 minutes.

Run:  .\\.venv\\Scripts\\python.exe -m tests.camera_probe
"""
import json
import time

import cv2

from engine.eyes.camera import open_capture

MODES = [(1280, 720, 30), (1280, 720, 60), (1920, 1080, 30), (640, 480, 30)]
TRIES = [("dshow", "MJPG"), ("dshow", ""), ("msmf", "MJPG"), ("msmf", "")]
DARK = 10  # average brightness (0-255) below this: a virtual camera, a closed privacy shutter or a covered lens

try:  # hide OpenCV's "can't be used to capture by index" noise for indexes that don't exist
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except AttributeError:
    pass


def measure(cap, seconds: float = 2.0) -> tuple[float, float]:
    """Frames per second actually delivered, and the median brightness of the frames."""
    for _ in range(5):  # let exposure settle
        cap.read()
    frames, samples = 0, []
    start = time.monotonic()
    while time.monotonic() - start < seconds:
        ok, frame = cap.read()
        if ok and frame is not None:
            frames += 1
            if frames % 5 == 1:
                samples.append(float(frame.mean()))
    samples.sort()
    brightness = samples[len(samples) // 2] if samples else 0.0
    return frames / (time.monotonic() - start), brightness


def find_cameras(max_index: int = 5) -> list[int]:
    found = []
    for index in range(max_index):
        cap, _ = open_capture(index, "dshow", "MJPG", 1280, 720, 30)
        if cap is None:
            cap, _ = open_capture(index, "dshow", "", 640, 480, 30)
        if cap is not None:
            cap.release()
            found.append(index)
    return found


def pick_best(results: list[dict]) -> dict | None:
    """Prefer a bright 1280x720 mode near 30 fps; earlier entries in TRIES win ties."""
    usable = [r for r in results if r["brightness"] >= DARK and r["fps"] >= 25]
    if not usable:
        return None
    hd = [r for r in usable if (r["width"], r["height"]) == (1280, 720)]
    pool = hd or usable
    return min(pool, key=lambda r: (abs(r["fps"] - 30) > 5, -r["width"], r["order"]))


def main() -> None:
    print("Looking for cameras...")
    cameras = find_cameras()
    if not cameras:
        print("No camera opened. Check the USB cable, close apps using the camera, and check "
              "Windows Settings > Privacy & security > Camera (allow desktop apps).")
        raise SystemExit(1)
    print(f"Camera indexes that open: {cameras}\n")

    results, order = [], 0
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
                note = "  DARK (virtual camera, shutter or covered lens?)" if brightness < DARK else ""
                print(f"got {mode.width}x{mode.height} {mode.fourcc}, {real_fps:.1f} fps, "
                      f"brightness {brightness:.0f}{note}")
                results.append({"index": index, "backend": backend, "fourcc": fourcc,
                                "width": mode.width, "height": mode.height,
                                "fps": round(real_fps, 1), "brightness": round(brightness), "order": order})
                order += 1

    print("\n=== Summary ===")
    for index in cameras:
        rows = [r for r in results if r["index"] == index]
        if rows and all(r["brightness"] < DARK for r in rows):
            print(f"camera {index}: every frame was dark. Probably a virtual camera "
                  "(OBS, NVIDIA Broadcast, a capture app) or a closed shutter. Don't use it.")
        elif rows:
            fast = sorted({(r["width"], r["height"], round(r["fps"])) for r in rows if r["brightness"] >= DARK})
            print(f"camera {index}: real picture. Modes seen (w, h, fps): {fast}")

    best = pick_best(results)
    if best is None:
        print("\nNo bright mode at 25 fps or more. Open the shutter, add light, or try another USB port.")
        return
    settings = {"camera_index": best["index"], "camera_backend": best["backend"],
                "camera_fourcc": best["fourcc"], "frame_width": best["width"],
                "frame_height": best["height"], "fps": 30}
    print(f"\nRecommended: camera {best['index']}, {best['backend']} {best['fourcc'] or 'auto'}, "
          f"{best['width']}x{best['height']} at {best['fps']} fps measured")
    sixty = [r for r in results if r["index"] == best["index"] and r["brightness"] >= DARK
             and (r["width"], r["height"]) == (best["width"], best["height"]) and r["fps"] >= 50]
    if sixty:
        print("This camera also does 60 fps at that size. Try it with: python -m tests.eye_bars --fps 60")
    print("Settings:")
    print(json.dumps(settings))
    print("\nOn the demo PC, these become the defaults in engine/config.py. "
          "On any other laptop, put them in config.local.json.")


if __name__ == "__main__":
    main()
