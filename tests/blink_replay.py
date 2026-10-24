"""Replays recorded blink scores (tests.blink_record) through the blink detector and says how it did,
per row of the screen: clicks that should fire (2 per CLICK spot) and clicks that must not (everything else).
It also runs the old fixed-threshold detector, so you can see what changed, and suggests settings.

Run:  python -m tests.blink_replay                       (every file in calibration/blink_data)
      python -m tests.blink_replay path\\to\\file.csv --rise 0.15 --closed-abs 0.5    (try other values)
If the suggestions differ from the defaults, put them in config.local.json as "blink_rise" and
"blink_closed_abs" and paste this output to Khang's chat.
"""
import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import engine.eyes.blink_gesture as B

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "calibration" / "blink_data"


def load(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            row = {"t": int(float(r["t_ms"])), "phase": r["phase"], "row": r["row"], "col": r["col"],
                   "face": r["face"] == "1"}
            if row["face"]:
                row["L"], row["R"] = float(r["eyeBlinkLeft"]), float(r["eyeBlinkRight"])
                row["pitch"] = float(r["pitch"])
            out.append(row)
    return out


def segments(frames):
    """[(phase, row, first_index, last_index)] in order."""
    def key(f):
        return f["phase"], f["row"], f["col"]

    segs, start = [], 0
    for i in range(1, len(frames) + 1):
        if i == len(frames) or key(frames[i]) != key(frames[start]):
            segs.append((frames[start]["phase"], frames[start]["row"], start, i - 1))
            start = i
    return segs


def run_new(frames, rise, closed_abs):
    d = B.BlinkDetector(rise=rise, closed_abs=closed_abs)
    fired = []
    for i, f in enumerate(frames):
        bs = {"eyeBlinkLeft": f["L"], "eyeBlinkRight": f["R"]} if f["face"] else {}
        if d.update(f["t"], bs, f.get("pitch")) is not None:
            fired.append(i)
    return fired


def run_old(frames):
    """The fixed thresholds from before Oct 24 PM (0.55 to close, 0.35 to open again)."""
    fired, since, quiet, last = [], None, 0, None
    for i, f in enumerate(frames):
        if not f["face"] or (last is not None and f["t"] - last > B.MAX_GAP_MS):
            since = None
        last = f["t"]
        if not f["face"]:
            continue
        s = min(f["L"], f["R"])
        if since is None:
            if s >= B.OLD_CLOSE_ABOVE and f["t"] >= quiet:
                since = f["t"]
            continue
        if s > B.OLD_OPEN_BELOW:
            continue
        lasted, since = f["t"] - since, None
        if B.MIN_CLOSED_MS <= lasted <= B.MAX_CLOSED_MS:
            quiet = f["t"] + B.COOLDOWN_MS
            fired.append(i)
    return fired


def report(frames, rise, closed_abs) -> dict:
    segs = segments(frames)
    new, old = set(run_new(frames, rise, closed_abs)), set(run_old(frames))
    # a click fires when the eyes open again, which can be just after its CLICK segment ends: count
    # fires up to 15 frames (0.5 s) after a click segment as that segment's
    table = defaultdict(lambda: {"want": 0, "new": 0, "old": 0, "false_new": 0, "false_old": 0})
    for phase, row, a, b in segs:
        if phase == "warmup":
            continue
        key = row if phase in ("look", "click") else phase
        end = b + (15 if phase == "click" else 0)
        n_new = sum(a <= i <= end for i in new)
        n_old = sum(a <= i <= end for i in old)
        if phase == "click":
            table[key]["want"] += 2
            table[key]["new"] += min(n_new, 2)
            table[key]["old"] += min(n_old, 2)
            table[key]["false_new"] += max(0, n_new - 2)
            table[key]["false_old"] += max(0, n_old - 2)
        else:
            if phase == "look":   # a click that ended just before this LOOK began was counted already
                n_new = sum(a + 15 < i <= b for i in new)
                n_old = sum(a + 15 < i <= b for i in old)
            table[key]["false_new"] += n_new
            table[key]["false_old"] += n_old
    return table


def levels(frames):
    """Per row: median open score while looking, and the median peak of each deliberate close."""
    out = {}
    for row in ("top", "middle", "bottom"):
        look = [min(f["L"], f["R"]) for f in frames if f["face"] and f["phase"] == "look" and f["row"] == row]
        if not look:
            continue
        shut_from = statistics.median(look) + 0.15     # over this = a close in progress
        peaks, cur = [], []
        for f in frames + [{"face": False, "phase": "", "row": ""}]:
            s = min(f["L"], f["R"]) if f["face"] and f["phase"] == "click" and f["row"] == row else None
            if s is not None and s >= shut_from:
                cur.append(s)
            elif cur:
                if len(cur) >= 5:                         # about 0.15 s or more: a deliberate close
                    peaks.append(max(cur))
                cur = []
        pitch = [f["pitch"] for f in frames if f["face"] and f["phase"] == "look" and f["row"] == row]
        if look:
            out[row] = (statistics.median(look), statistics.median(peaks) if peaks else None, len(peaks),
                        statistics.median(pitch) if pitch else 0.0)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Replay recorded blink scores through the detector")
    p.add_argument("files", nargs="*")
    p.add_argument("--rise", type=float, default=B.RISE)
    p.add_argument("--closed-abs", type=float, default=B.CLOSED_ABS)
    args = p.parse_args()
    files = [Path(f) for f in args.files] or sorted(DATA.glob("blink_*.csv"))
    if not files:
        raise SystemExit(f"No recordings in {DATA}. Run  python -m tests.blink_record  first.")
    lv_all = defaultdict(list)
    for path in files:
        frames = load(path)
        seen = sum(f["face"] for f in frames)
        print(f"\n=== {path.name}: {len(frames)} frames, face in {seen} ({100 * seen / max(1, len(frames)):.0f}%)")
        t = report(frames, args.rise, args.closed_abs)
        print(f"{'where':<8} {'clicks found (new / old)':<26} {'wrong clicks (new / old)':<24}")
        for key in ("top", "middle", "bottom", "updown", "read"):
            if key in t:
                r = t[key]
                found = f"{r['new']}/{r['want']}  /  {r['old']}/{r['want']}" if r["want"] else "-"
                print(f"{key:<8} {found:<26} {r['false_new']}  /  {r['false_old']}")
        for row, (open_s, peak, n, pitch) in levels(frames).items():
            lv_all[row].append((open_s, peak))
            print(f"  {row:<7} head pitch {pitch:+5.1f} deg   open {open_s:.2f}   shut peak "
                  f"{'-' if peak is None else f'{peak:.2f}'} ({n} closes)")
    # suggestions
    gaps = [peak - open_s for row in lv_all for open_s, peak in lv_all[row] if peak is not None]
    peaks = [peak for row in lv_all for _, peak in lv_all[row] if peak is not None]
    if gaps and peaks:
        rise = round(max(0.12, min(0.3, 0.4 * min(gaps))), 2)
        closed = round(max(0.4, min(0.65, min(peaks) - 0.15)), 2)
        print(f"\nSmallest gap between open and shut: {min(gaps):.2f}; lowest shut peak: {min(peaks):.2f}")
        print(f"Suggested: \"blink_rise\": {rise}, \"blink_closed_abs\": {closed}   "
              f"(now {B.RISE}, {B.CLOSED_ABS}). Try them:  python -m tests.blink_replay --rise {rise} --closed-abs {closed}")


if __name__ == "__main__":
    main()
