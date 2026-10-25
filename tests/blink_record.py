"""Records your own blink scores at the top, middle and bottom of the screen, for tests.blink_replay.
About 2 minutes. Only numbers are saved (eye scores and head angles, 30 a second), never video, to
calibration/blink_data/blink_<date>_<time>.csv (git-ignored).

Run:  .\\.venv\\Scripts\\python.exe -m tests.blink_record          (--camera 1 if the picture is black)
A dark full-screen page shows a yellow dot and what to do. Point with your HEAD, the way you use EYEable:
  LOOK   look at the dot; blink normally, don't close your eyes on purpose        (3 s)
  CLICK  look at the dot and close BOTH eyes on purpose for about half a second, twice  (6 s)
at 9 spots (top, middle and bottom rows), then:
  UP AND DOWN   move your head between the top and the bottom of the screen; no blinks on purpose (15 s)
  READ          read the text at the bottom of the screen; no blinks on purpose (20 s)
Esc stops early (what was recorded so far is kept). Then run  python -m tests.blink_replay
"""
import argparse
import csv
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "calibration" / "blink_data"
SPOTS = [(r, c) for r in ("top", "middle", "bottom") for c in ("left", "center", "right")]
ROW_Y = {"top": 0.12, "middle": 0.5, "bottom": 0.88}
COL_X = {"left": 0.15, "center": 0.5, "right": 0.85}
SHAPES = ["eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
          "eyeSquintLeft", "eyeSquintRight"]
READ_TEXT = ("Read this slowly, the way you would read an email. EYEable lets people use a computer with "
             "their head, eyes and voice. A long blink clicks, and the card at the top shows what you said. "
             "Keep reading until the page moves on, and only blink the way you normally would.")


def plan():
    """[(phase, row, col, seconds, prompt)]"""
    steps = []
    for row, col in SPOTS:
        steps.append(("look", row, col, 3.0, "LOOK at the dot (blink normally)"))
        steps.append(("click", row, col, 6.0, "CLICK: close both eyes for half a second, TWICE"))
    steps.append(("updown", "", "", 15.0, "UP AND DOWN: move your head between the top and bottom dots, no blinks on purpose"))
    steps.append(("read", "bottom", "", 20.0, "READ the text below, no blinks on purpose"))
    return steps


def main() -> None:
    p = argparse.ArgumentParser(description="Record blink scores across the screen")
    p.add_argument("--camera", type=int, default=None)
    args = p.parse_args()

    import tkinter as tk

    from engine.eyes.camera import Camera
    from engine.eyes.face_mesh import FaceMesh
    from engine.eyes.head_pose import HeadPoseEstimator

    cam = Camera(index=args.camera)
    mode = cam.start()
    mesh, head = FaceMesh(), HeadPoseEstimator()
    rows: list[list] = []
    state = {"phase": "warmup", "row": "", "col": ""}
    lock, stop = threading.Lock(), threading.Event()

    def loop():
        frame_id = 0
        while not stop.is_set():
            frame, t_ms, new_id = cam.wait_next(frame_id)
            if frame is None or new_id == frame_id:
                continue
            frame_id = new_id
            face = mesh.process(frame, t_ms)
            with lock:
                ph, r, c = state["phase"], state["row"], state["col"]
            if face is None:
                rows.append([t_ms, ph, r, c, 0] + [""] * (len(SHAPES) + 2))
                continue
            pose = head.compute(face)
            bs = face.blendshapes or {}
            rows.append([t_ms, ph, r, c, 1] + [round(float(bs.get(k, 0.0)), 4) for k in SHAPES]
                        + [round(pose.pitch, 2), round(pose.yaw, 2)])

    threading.Thread(target=loop, daemon=True).start()

    root = tk.Tk()
    root.attributes("-fullscreen", True)
    root.configure(bg="#111111")
    W, H = root.winfo_screenwidth(), root.winfo_screenheight()
    canvas = tk.Canvas(root, width=W, height=H, bg="#111111", highlightthickness=0)
    canvas.pack()
    root.bind("<Escape>", lambda e: finish())
    steps = plan()
    total = sum(s[3] for s in steps)
    print(f"Camera {mode.width}x{mode.height}. About {total / 60:.0f} minutes. Esc stops early.")

    def draw(prompt, row, col, left_s, idx):
        canvas.delete("all")
        canvas.create_text(W // 2, H * 0.3 if row != "middle" else H * 0.3, text=prompt, fill="#ffffff",
                           font=("Segoe UI", 26, "bold"), width=W * 0.8)
        canvas.create_text(W // 2, H * 0.3 + 60, text=f"{left_s:.0f} s   ({idx}/{len(steps)})", fill="#a1a1aa",
                           font=("Segoe UI", 16))
        if row and col:
            x, y = W * COL_X[col], H * ROW_Y[row]
            canvas.create_oval(x - 18, y - 18, x + 18, y + 18, fill="#ffd400", outline="#000000", width=3)
        elif row == "bottom":
            canvas.create_text(W // 2, H * 0.86, text=READ_TEXT, fill="#e5e5e5", font=("Segoe UI", 18),
                               width=W * 0.7)
        else:   # up and down
            for y in (H * ROW_Y["top"], H * ROW_Y["bottom"]):
                canvas.create_oval(W / 2 - 18, y - 18, W / 2 + 18, y + 18, fill="#ffd400", outline="#000000", width=3)

    def run(i=0, started=None):
        if stop.is_set():
            return
        if i >= len(steps):
            finish()
            return
        phase, row, col, secs, prompt = steps[i]
        now = time.monotonic()
        if started is None:
            started = now
            with lock:
                state.update(phase=phase, row=row, col=col)
        left = secs - (now - started)
        if left <= 0:
            root.after(1, run, i + 1, None)
            return
        draw(prompt, row, col, left, i + 1)
        root.after(100, run, i, started)

    def finish():
        if stop.is_set():
            return
        stop.set()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / f"blink_{datetime.now():%Y%m%d_%H%M%S}.csv"
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t_ms", "phase", "row", "col", "face"] + SHAPES + ["pitch", "yaw"])
            w.writerows(rows)
        print(f"Saved {len(rows)} frames to {out}")
        print("Next:  python -m tests.blink_replay")
        try:
            root.destroy()
        except Exception:
            pass

    with lock:
        state.update(phase="warmup")
    root.after(2000, run)   # 2 s to settle (the open level learns)
    try:
        root.mainloop()
    finally:
        stop.set()
        cam.stop()
        mesh.close()


if __name__ == "__main__":
    main()
