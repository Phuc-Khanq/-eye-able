"""Record a scripted head session and score nod, shake and tilt detection on it. Saves angles only.

Record (about 3 minutes, needs the webcam):
    .\\.venv\\Scripts\\python.exe -m tests.head_eval
Score a saved session again with the current code, or try other thresholds on it:
    .\\.venv\\Scripts\\python.exe -m tests.head_eval --replay calibration\\head_session_XXXX.json
    .\\.venv\\Scripts\\python.exe -m tests.head_eval --replay calibration\\head_session_XXXX.json --sweep
    .\\.venv\\Scripts\\python.exe -m tests.head_eval --replay FILE --nod-deg 6 --shake-deg 7

The session follows on-screen prompts: a quiet stretch (nothing should fire), then 10 nods, 10 shakes,
5 tilts each way, a stretch of natural movement (nothing should fire), then gentle nods and shakes.
Camera frames are never saved, only yaw, pitch and roll over time, in calibration/ (git ignores it).
Recording once is enough: replay re-scores the same motion after any code change, so before and after
are measured on identical input.

Keys while recording: SPACE = start, Q or Esc = quit (the session is saved only if it finished).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from engine.core.types import HeadPose
from engine.eyes.head_gestures import GestureDetector

REPO_ROOT = Path(__file__).resolve().parents[1]
SLOT_S = 3.0
COUNTDOWN_S = 4.0

# (label, gesture that should fire once per slot or None, slots or seconds, what the person sees)
SEGMENTS = [
    ("idle", None, 25.0, "Look around the screen, read, scratch your nose. No nods, no shakes."),
    ("nod", "nod", 10, "NOD"),
    ("shake", "shake", 10, "SHAKE your head no"),
    ("tilt_left", "tilt_left", 5, "TILT your head LEFT and hold"),
    ("tilt_right", "tilt_right", 5, "TILT your head RIGHT and hold"),
    ("natural", None, 30.0, "Move naturally: stretch, lean in, look at the keyboard or your phone, turn to a friend."),
    ("nod_soft", "nod", 6, "small, gentle NOD"),
    ("shake_soft", "shake", 6, "small, gentle SHAKE"),
]
QUIET_LABELS = ("idle", "natural")


def build_schedule() -> list[dict]:
    """Segments with start and end in ms from the end of the countdown, and their slots."""
    t = 0.0
    out = []
    for label, expected, size, text in SEGMENTS:
        if expected is None:
            start, end, slots = t, t + size, []
            t = end
        else:
            start = t
            slots = [[(t + i * SLOT_S) * 1000, (t + (i + 1) * SLOT_S) * 1000] for i in range(size)]
            t += size * SLOT_S
            end = t
        out.append({"label": label, "expected": expected, "text": text, "start_ms": start * 1000,
                    "end_ms": end * 1000, "slots": slots})
    return out


def run_detector(poses, make_detector) -> list[tuple[float, str]]:
    det = make_detector()
    found = []
    for t, yaw, pitch, roll in poses:
        g = det.update(HeadPose(t_ms=int(t), yaw=yaw, pitch=pitch, roll=roll))
        if g is not None:
            found.append((float(g.t_ms), g.type))
    return found


def score(session: dict, make_detector=GestureDetector) -> dict:
    """Replay the recorded angles through a fresh detector and compare with the prompts."""
    gestures = run_detector(session["poses"], make_detector)
    used = set()
    rows = {}
    for seg in session["segments"]:
        row = {"label": seg["label"], "expected": seg["expected"], "n": len(seg["slots"]), "hit": 0, "miss": 0,
               "wrong": 0, "extra": 0, "latency_ms": [], "false": 0}
        if seg["expected"] is None:
            inside = [i for i, (t, _) in enumerate(gestures) if seg["start_ms"] <= t < seg["end_ms"]]
            row["false"] = len(inside)
            used.update(inside)
        for start, end in seg["slots"]:
            inside = [i for i, (t, _) in enumerate(gestures) if start <= t < end]
            used.update(inside)
            right = [i for i in inside if gestures[i][1] == seg["expected"]]
            if right:
                row["hit"] += 1
                row["latency_ms"].append(gestures[right[0]][0] - start)
                row["extra"] += len(inside) - 1
            elif inside:
                row["wrong"] += 1
                row["extra"] += len(inside) - 1
            else:
                row["miss"] += 1
        rows[seg["label"]] = row
    stray = [g for i, g in enumerate(gestures) if i not in used]
    minutes_quiet = sum(r["end_ms"] - r["start_ms"] for r in session["segments"] if r["label"] in QUIET_LABELS) / 60000
    false_total = sum(r["false"] + r["extra"] for r in rows.values()) + len(stray)
    return {"rows": rows, "stray": stray, "false_total": false_total, "quiet_minutes": minutes_quiet,
            "false_per_min": false_total / minutes_quiet if minutes_quiet else float("nan")}


def print_report(result: dict, title: str = "") -> None:
    if title:
        print(title)
    print(f"{'part':<11} {'asked':>5} {'hit':>4} {'miss':>5} {'wrong':>6} {'extra':>6} {'median s':>9}")
    for r in result["rows"].values():
        if r["expected"] is None:
            print(f"{r['label']:<11} {'-':>5} {'-':>4} {'-':>5} {'-':>6} {r['false']:>6}   (nothing should fire)")
        else:
            lat = f"{np.median(r['latency_ms']) / 1000:.2f}" if r["latency_ms"] else "-"
            print(f"{r['label']:<11} {r['n']:>5} {r['hit']:>4} {r['miss']:>5} {r['wrong']:>6} {r['extra']:>6} {lat:>9}")
    asked = sum(r["n"] for r in result["rows"].values())
    hits = sum(r["hit"] for r in result["rows"].values())
    print(f"hits {hits}/{asked}   false triggers {result['false_total']} "
          f"({result['false_per_min']:.1f} per quiet minute)   outside any prompt: {len(result['stray'])}")


def sweep(session: dict) -> None:
    """Try other thresholds on the same recording. 'extra' and 'false' count as false triggers."""
    grids = [("nod_deg", [5, 6, 7, 8, 10, 12], ("nod", "nod_soft")),
             ("shake_deg", [5, 6, 7, 8, 10, 12], ("shake", "shake_soft")),
             ("tilt_deg", [10, 12, 15, 18, 20], ("tilt_left", "tilt_right"))]
    for name, values, labels in grids:
        print(f"\n{name}: hits on {', '.join(labels)} / false triggers (whole session)")
        for v in values:
            res = score(session, lambda v=v, name=name: GestureDetector(**{name: float(v)}))
            hits = sum(res["rows"][l]["hit"] for l in labels if l in res["rows"])
            asked = sum(res["rows"][l]["n"] for l in labels if l in res["rows"])
            print(f"  {v:>4}: {hits:>3}/{asked:<3}  false {res['false_total']}")


def make_detector_from(args) -> "callable":
    kw = {k: v for k, v in (("nod_deg", args.nod_deg), ("shake_deg", args.shake_deg), ("tilt_deg", args.tilt_deg))
          if v is not None}
    return lambda: GestureDetector(**kw)


def record(args) -> None:
    import cv2

    from engine.eyes.camera import Camera
    from engine.eyes.face_mesh import FaceMesh
    from engine.eyes.head_pose import HeadPoseEstimator

    BG, TEXT, QUIET = (32, 28, 26), (235, 235, 235), (150, 150, 150)
    GOOD, WARN, BAD, ACCENT = (120, 210, 120), (60, 170, 240), (90, 90, 230), (240, 200, 90)
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    schedule = build_schedule()
    total_ms = schedule[-1]["end_ms"]

    cam = Camera(index=args.camera, fps=args.fps)
    mode = cam.start()
    camera_label = f"{mode.width}x{mode.height} {mode.fourcc} via {mode.backend}, camera {mode.index}"
    print("Camera:", camera_label)
    mesh = FaceMesh()
    estimator = HeadPoseEstimator()
    live = GestureDetector()
    window = "Diseyeble head session (Q to quit)"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    def put(img, s, y, scale=0.8, color=TEXT, thick=1, x=None):
        size = cv2.getTextSize(s, FONT, scale, thick)[0]
        cv2.putText(img, s, ((img.shape[1] - size[0]) // 2 if x is None else x, y), FONT, scale, color, thick,
                    cv2.LINE_AA)

    state, start_wall = "ready", 0.0
    t0_ms = None
    poses, live_gestures = [], []
    last_gesture = ("", -10.0)
    neutral_done = False
    frame_id = 0
    saved = None
    try:
        while True:
            frame, t_ms, frame_id = cam.wait_next(frame_id)
            now = time.monotonic()
            face = pose = None
            if frame is not None:
                face = mesh.process(frame, t_ms)
                if face is not None:
                    pose = estimator.compute(face)

            elapsed = now - start_wall if state in ("countdown", "running") else 0.0
            if state == "countdown" and elapsed >= COUNTDOWN_S:
                if face is not None:
                    estimator.set_neutral(face)
                    pose = estimator.compute(face)
                    neutral_done = True
                state, t0_ms = "running", t_ms
                live.reset()
                print("Recording...")
            if state == "running":
                rel = t_ms - t0_ms
                if pose is not None:
                    poses.append([rel, round(pose.yaw, 3), round(pose.pitch, 3), round(pose.roll, 3)])
                    g = live.update(HeadPose(t_ms=int(rel), yaw=pose.yaw, pitch=pose.pitch, roll=pose.roll))
                    if g is not None:
                        live_gestures.append([g.t_ms, g.type])
                        last_gesture = (g.type, now)
                if rel >= total_ms:
                    state = "done"
                    session = {"version": 1, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "camera": camera_label,
                               "neutral": [round(v, 3) for v in estimator.neutral], "segments": schedule,
                               "poses": poses, "live_gestures": live_gestures}
                    out = REPO_ROOT / "calibration" / f"head_session_{time.strftime('%Y%m%d_%H%M%S')}.json"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(json.dumps(session), encoding="utf-8")
                    saved = out
                    print(f"Saved {out} ({len(poses)} frames, angles only)\n")
                    print_report(score(session, make_detector_from(args)), "Score with the current code")

            img = np.full((540, 960, 3), BG, dtype=np.uint8)
            if not face and state != "done":
                put(img, "NO FACE: face the camera, add light", 520, 0.6, BAD)
            if state == "ready":
                put(img, "Head gesture session", 90, 1.2, TEXT, 2)
                put(img, "About 3 minutes. Sit as you normally would, hands off the keyboard.", 170, 0.7)
                put(img, "You will be asked to nod, shake, tilt, and also to just move naturally.", 210, 0.7)
                put(img, "Only angles are saved. No camera pictures.", 250, 0.7, QUIET)
                put(img, "SPACE to start   Q to quit", 330, 0.8, GOOD if face else QUIET)
            elif state == "countdown":
                put(img, "Look at the center of the screen and hold still", 150, 0.9)
                put(img, f"{max(0.0, COUNTDOWN_S - elapsed):.0f}", 300, 3.0, ACCENT, 4)
            elif state == "running":
                rel_s = (t_ms - t0_ms) / 1000.0
                seg = next((s for s in schedule if s["start_ms"] <= (t_ms - t0_ms) < s["end_ms"]), schedule[-1])
                put(img, f"{seg['label']}   {100 * (t_ms - t0_ms) / total_ms:.0f}% done", 40, 0.6, QUIET)
                if seg["expected"] is None:
                    put(img, "Natural movement" if seg["label"] == "natural" else "Quiet", 160, 1.2, ACCENT, 2)
                    put(img, seg["text"], 230, 0.65)
                else:
                    k = int(((t_ms - t0_ms) - seg["start_ms"]) // (SLOT_S * 1000))
                    k = min(max(k, 0), len(seg["slots"]) - 1)
                    slot_start = seg["slots"][k][0]
                    into = ((t_ms - t0_ms) - slot_start) / (SLOT_S * 1000)
                    put(img, seg["text"], 190, 1.6, ACCENT, 3)
                    put(img, f"{k + 1} of {len(seg['slots'])}: do it once, now", 260, 0.8)
                    cv2.rectangle(img, (240, 300), (720, 316), (70, 64, 60), -1)
                    cv2.rectangle(img, (240, 300), (240 + int(480 * min(into, 1.0)), 316), ACCENT, -1)
                if now - last_gesture[1] < 1.0:
                    put(img, f"detected: {last_gesture[0]}", 440, 0.9, GOOD)
            elif state == "done":
                put(img, "Done. Saved to calibration/", 130, 1.0, GOOD, 2)
                put(img, "Results are in the PowerShell window. Send that table and the file", 200, 0.65)
                put(img, f"{saved.name if saved else ''}", 240, 0.65, QUIET)
                put(img, "Q to quit", 320, 0.8, QUIET)
            cv2.imshow(window, img)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" ") and state == "ready" and face is not None:
                state, start_wall = "countdown", now
                print("Look at the center of the screen and hold still...")
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cam.stop()
        mesh.close()
        cv2.destroyAllWindows()
        if state in ("countdown", "running"):
            print("Stopped before the end; nothing was saved.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Record or replay a head gesture session and score detection")
    parser.add_argument("--replay", default=None, help="score this saved session instead of recording")
    parser.add_argument("--sweep", action="store_true", help="with --replay: try other thresholds on the recording")
    parser.add_argument("--nod-deg", type=float, default=None)
    parser.add_argument("--shake-deg", type=float, default=None)
    parser.add_argument("--tilt-deg", type=float, default=None)
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    args = parser.parse_args()

    if args.replay:
        session = json.loads(Path(args.replay).read_text(encoding="utf-8"))
        print_report(score(session, make_detector_from(args)), f"{args.replay}  ({session.get('camera', '')})")
        if args.sweep:
            sweep(session)
    else:
        record(args)


if __name__ == "__main__":
    main()
