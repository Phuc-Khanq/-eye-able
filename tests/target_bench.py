"""UI-Venus bench: ask for on-screen targets, see where the model points, count hits and time each call.
Screenshots stay in memory and are never saved.

Run:  .\\.venv\\Scripts\\python.exe -m tests.target_bench
      .\\.venv\\Scripts\\python.exe -m tests.target_bench "Subscribe" "Search" "Home"
Options:  --no-zoom   skip the small zoom box, ask about the full 1280x720 box only (to compare)
          --move      also move the real cursor onto the answer
          --wait 3    seconds to switch windows before each capture

How it works
  1. Type a target (or pass a list). You get a few seconds: switch to the window you want and rest
     the mouse ON or right next to the target. The mouse stands in for your gaze for now.
  2. It asks UI-Venus about a 640x360 box around the mouse first (enlarged 2x), then the 1280x720 box
     if the target isn't there. The model only sees the box; it doesn't know where the mouse is.
  3. The window shows the 1280x720 box: blue ring = mouse, thin box = zoom box, red cross = answer.
     Close it with any key, then answer in the terminal: y if the cross was on the target, n if not.
Name targets the way they look: visible text works best ("Subscribe", "Search", "Shorts").
For icons, say what it looks like: "bell icon", "profile picture at the top right".
"""
import argparse
import statistics
import time

import cv2
import numpy as np

from engine.core.types import TargetRequest
from engine.eyes import os_input
from engine.eyes.screen_target import ScreenTarget, crop_box, grab, screen_bounds

WINDOW = "UI-Venus answer (any key to close)"


def show(screen, origin, mouse, finder, result) -> str:
    """Shows the wide box around the mouse with the zoom box and the answer, then asks in the terminal.
    Returns 'y', 'n' or ''."""
    ox, oy = origin
    left, top, w, h = crop_box(mouse[0] - ox, mouse[1] - oy, screen.width, screen.height, finder.crop_w, finder.crop_h)
    img = cv2.cvtColor(np.array(screen.crop((left, top, left + w, top + h))), cv2.COLOR_RGB2BGR)

    def local(x, y):
        return int(round(x - ox - left)), int(round(y - oy - top))

    for st in finder.trace:
        if st.name == "zoom":
            bl, bt, bw, bh = st.box
            cv2.rectangle(img, local(bl, bt), local(bl + bw, bt + bh), (200, 200, 200), 1, cv2.LINE_AA)
    cv2.circle(img, local(*mouse), 18, (255, 160, 40), 3, cv2.LINE_AA)
    if result.found:
        cv2.drawMarker(img, local(result.x, result.y), (40, 40, 255), cv2.MARKER_CROSS, 44, 3, cv2.LINE_AA)
        cv2.circle(img, local(result.x, result.y), 26, (40, 40, 255), 2, cv2.LINE_AA)
    found_by = finder.trace[-1].name if result.found else ""
    label = (f"{result.label}: " + (f"found in the {found_by} box" if result.found else "NOT FOUND")
             + f"   {result.ms:.0f} ms   blue = mouse, red = answer")
    cv2.rectangle(img, (0, 0), (img.shape[1], 40), (24, 22, 20), -1)
    cv2.putText(img, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(WINDOW, img)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_TOPMOST, 1)
    cv2.waitKey(0)
    cv2.destroyWindow(WINDOW)
    cv2.waitKey(1)
    answer = input("  Was the red cross on the target? [y/n, Enter = skip]: ").strip().lower()
    return answer[:1] if answer[:1] in ("y", "n") else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="UI-Venus target bench")
    parser.add_argument("targets", nargs="*", help="targets to ask for; none = type them one by one")
    parser.add_argument("--no-zoom", action="store_true", help="ask about the full 1280x720 box only")
    parser.add_argument("--move", action="store_true", help="move the real cursor onto each answer")
    parser.add_argument("--wait", type=float, default=3.0, help="seconds to switch windows before each capture")
    args = parser.parse_args()

    os_input.make_dpi_aware()
    finder = ScreenTarget(zoom=None if args.no_zoom else (640, 360))
    print(f"Loading {finder.model_id} on {finder.device} (the first load reads about 5 GB from disk)...")
    secs = finder.load()
    print(f"Loaded in {secs:.1f} s with {finder.model_class}.")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU memory in use: {torch.cuda.memory_allocated() / 2**30:.1f} GB")
    except Exception:
        pass
    from PIL import Image
    warm = finder.ask(Image.new("RGB", (640, 360), (255, 255, 255)), "the Send button")
    print(f"Warm-up: {warm.ms:.0f} ms, {warm.image_tokens} image tokens, model said {warm.raw!r} "
          "(blank image, so [-1,-1] is right)")
    if warm.image_tokens == 0:
        print("WARNING: the image didn't reach the model. Paste this whole output to Khang.")
    print("Steps: " + ("zoom 640x360 (enlarged 2x), then " if finder.zoom else "") + "wide 1280x720\n")

    queue = list(args.targets)
    times, hits, misses, skipped, not_found = [], 0, 0, 0, 0
    while True:
        if args.targets:
            if not queue:
                break
            label = queue.pop(0)
        else:
            label = input("Target to find (Enter to finish): ").strip()
            if not label:
                break
        for s in range(int(args.wait), 0, -1):
            print(f"  switch windows and rest the mouse on '{label}'... {s}  ", end="\r", flush=True)
            time.sleep(1)
        mouse = os_input.position()
        sl, st, sw, sh = screen_bounds()
        screen = grab(sl, st, sw, sh)  # one grab; both steps crop from it
        result = finder.find(TargetRequest(gaze_x=mouse[0], gaze_y=mouse[1], label=label), screen=screen, origin=(sl, st))
        times.append(result.ms)
        print(f"  {label}: " + (f"({result.x}, {result.y})" if result.found else "NOT FOUND")
              + f" in {result.ms:.0f} ms, mouse at {mouse}")
        for stage in finder.trace:
            a = stage.answer
            print(f"    {stage.name:4s} box {stage.box[2]}x{stage.box[3]}: said {a.raw!r} in {a.ms:.0f} ms, "
                  f"{a.image_tokens} image tokens" + ("  <- unreadable" if a.point is None else ""))
        if result.found:
            print(f"    answer is {np.hypot(result.x - mouse[0], result.y - mouse[1]):.0f} px from the mouse")
        if args.move and result.found:
            os_input.move(result.x, result.y)
        verdict = show(screen, (sl, st), mouse, finder, result)
        not_found += 0 if result.found else 1
        hits += verdict == "y"
        misses += verdict == "n"
        skipped += verdict == ""
        print(f"  marked: {'right' if verdict == 'y' else 'wrong' if verdict == 'n' else 'skipped'}\n")

    if times:
        judged = hits + misses
        print("=== Summary ===")
        print(f"{len(times)} targets, {hits}/{judged} right" + (f" ({100 * hits / judged:.0f}%)" if judged else "")
              + f", {not_found} answered 'not found', {skipped} skipped")
        print(f"time per target: median {statistics.median(times):.0f} ms, slowest {max(times):.0f} ms"
              + ("  (zoom, then wide if needed)" if finder.zoom else "  (wide box only)"))
    finder.close()


if __name__ == "__main__":
    main()
