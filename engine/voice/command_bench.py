"""Stock vs fine-tuned Whisper on 20 short commands: the same synthetic clips, decoded the way the app does.

    python -m engine.voice.command_bench                                   # stock large-v3-turbo
    python -m engine.voice.command_bench --model models/whisper-elder-ct2  # the fine-tuned one
    python -m engine.voice.command_bench --device cpu --compute-type int8  # on a laptop

Each clip goes through audio_in (Silero VAD) -> Dictation.transcribe -> commands.parse, exactly like
voice_check (engine/voice/voice_check.py check_clip). Clips: data/voice/synth, every voice x normal /
slow / quiet / noisy (scripts/make_test_speech.ps1, then python -m tests.add_noise). Synthetic voices,
not elderly speakers: it compares the two models, it doesn't say how well an older user is heard.

Two scores per clip:
  text     right if the words match after normalizing (case, punctuation, numbers)
  command  right if the app would do the right thing: commands.parse gives the phrase's command
           ("stop" / "done" count when they'd leave scroll mode; "open" alone is no command)
Prints right / wrong / empty per phrase and in total; what a wrong clip became is printed, never saved.
Saves the numbers only (phrase ids and counts, no audio or transcripts) to checkpoints/command_bench_<model>.json.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import time
from collections import defaultdict
from pathlib import Path

from engine.voice import commands
from engine.voice.voice_check import SYNTH, command_name, normalize_words

# (phrase id in tests/speech_phrases.csv, text): Khang's seven, then the demo's and the other short commands
PHRASES = [
    ("command_080", "yes"), ("command_092", "no"), ("short_001", "click"), ("short_002", "scroll"),
    ("short_003", "stop"), ("command_056", "send it"), ("short_004", "open"),
    ("short_005", "done"), ("command_044", "cancel"), ("command_035", "menu"), ("command_019", "pause"),
    ("command_026", "resume"), ("command_050", "scroll up"), ("command_053", "scroll down"),
    ("command_001", "start dictation"), ("command_010", "stop dictation"), ("command_068", "undo"),
    ("command_074", "new line"), ("short_006", "click send"), ("target_001", "click compose"),
]
SCROLL_EXIT = "scroll_exit"
REPO = Path(__file__).resolve().parents[2]


def heard_command(heard: str, mode: str) -> str:
    """What the app does with this text in the phrase's mode: a command name, target:<Label>, scroll_exit,
    none or empty. The phrase itself gives what it should do."""
    from engine.core.mode_manager import SCROLL_EXIT as EXIT

    if mode == "scroll" and commands.normalize(heard) in EXIT:
        return SCROLL_EXIT
    return command_name(commands.parse(heard, allow_targets=mode == "pointer")) if heard.strip() else "empty"


def score(text: str, mode: str, heard: list[str]) -> dict:
    """One clip: heard is every transcript Whisper gave for it (usually one)."""
    said = " ".join(heard).strip()
    if not said:
        return {"text": False, "command": False, "empty": True, "became": "empty"}
    want = heard_command(text, mode)
    got = [heard_command(h, mode) for h in heard if h.strip()]
    return {"text": normalize_words(said) == normalize_words(text), "command": got == [want], "empty": False,
            "became": " | ".join(got)}


def clips(manifest: Path = SYNTH / "manifest.csv") -> dict[str, list[dict]]:
    """phrase id -> its manifest rows (one per voice and variant)."""
    wanted = {pid for pid, _ in PHRASES}
    out: dict[str, list[dict]] = defaultdict(list)
    with manifest.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["id"] in wanted and (SYNTH / row["file"]).exists():
                out[row["id"]].append(row)
    return out


def run(dictation, settings, rows_by_id: dict[str, list[dict]], transcribe=None, log=print) -> dict:
    """Score every clip. transcribe(row) -> [texts] (default: voice_check's audio_in -> Whisper path)."""
    if transcribe is None:
        from engine.voice.voice_check import check_clip

        def transcribe(row):
            heard = check_clip(row, dictation, settings)["heard"]
            return [h for h in heard.split(" | ") if h] if heard else []

    per: dict[str, dict] = {}
    for pid, text in PHRASES:
        rows = rows_by_id.get(pid, [])
        n = {"clips": len(rows), "text_right": 0, "command_right": 0, "command_wrong": 0, "empty": 0}
        for row in rows:
            heard = transcribe(row)
            s = score(text, row["mode"], heard)
            n["text_right"] += s["text"]
            n["command_right"] += s["command"]
            n["empty"] += s["empty"]
            n["command_wrong"] += not s["command"] and not s["empty"]
            if not s["command"]:
                log(f"    {text!r} ({row['voice']}/{row['variant']}): heard {' | '.join(heard)!r} -> {s['became']}")
        per[pid] = {"phrase": text, **n}
    total = {k: sum(p[k] for p in per.values()) for k in ("clips", "text_right", "command_right",
                                                          "command_wrong", "empty")}
    return {"per_phrase": per, "total": total}


def table(result: dict) -> str:
    lines = [f"{'phrase':<18}{'clips':>6}{'text ok':>9}{'cmd ok':>8}{'wrong':>7}{'empty':>7}"]
    for p in list(result["per_phrase"].values()) + [{"phrase": "TOTAL", **result["total"]}]:
        lines.append(f"{p['phrase']:<18}{p['clips']:>6}{p['text_right']:>9}{p['command_right']:>8}"
                     f"{p['command_wrong']:>7}{p['empty']:>7}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from engine.config import settings
    from engine.voice.dictation import Dictation

    p = argparse.ArgumentParser(description="Stock vs fine-tuned Whisper on 20 short commands.")
    p.add_argument("--model", default="large-v3-turbo", help="a Whisper name or a CTranslate2 folder")
    p.add_argument("--device", default=None, help="cuda or cpu (default: the config's)")
    p.add_argument("--compute-type", default=None, help="float16 on the GPU, int8 on the CPU")
    args = p.parse_args(argv)
    changes = {"whisper_model": args.model}
    if args.device:
        changes["whisper_device"] = args.device
    if args.compute_type:
        changes["whisper_compute_type"] = args.compute_type
    settings = dataclasses.replace(settings, **changes)

    rows = clips()
    missing = [text for pid, text in PHRASES if not rows.get(pid)]
    if missing:
        print(f"no clips for {missing}: run scripts/make_test_speech.ps1, then python -m tests.add_noise")
        return 1
    dictation = Dictation(settings)
    t0 = time.perf_counter()
    dictation.load()
    print(f"Whisper {args.model} on {settings.whisper_device} ({settings.whisper_compute_type}) loaded in "
          f"{time.perf_counter() - t0:.1f} s; {sum(len(r) for r in rows.values())} clips")
    t0 = time.perf_counter()
    result = run(dictation, settings, rows)
    print(table(result))
    t = result["total"]
    print(f"{args.model}: {t['command_right']}/{t['clips']} commands right, {t['text_right']}/{t['clips']} "
          f"exact text, {t['empty']} empty ({time.perf_counter() - t0:.0f} s; synthetic voices)")
    out = REPO / "checkpoints" / f"command_bench_{Path(args.model).name}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"model": args.model, "device": settings.whisper_device,
                               "compute_type": settings.whisper_compute_type, **result}, indent=2),
                   encoding="utf-8")
    print(f"saved {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
