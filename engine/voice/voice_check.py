"""How well the voice chain hears the synthetic test clips. No mic needed.

Route: Khoi and Khoa (Khoi leads)
Each clip goes through the app's own path: engine.main.play_wav -> audio_in (Silero VAD)
-> Dictation (Whisper, settings from engine/config.py + config.local.json) -> commands.parse,
with allow_targets on only in Pointer mode, like the mode manager.

    .\\scripts\\make_test_speech.ps1 ; python -m tests.add_noise     # make the clips once
    python -m engine.voice.voice_check                              # every clip (CPU: ~20 min)
    python -m engine.voice.voice_check --variant normal --voice david   # a quick subset
    python -m engine.voice.voice_check --category near_miss --limit 20

Prints failures and a summary; writes the full table to data/voice/synth/report.md
(and the raw results to report.json next to it).
Reads clips from data/voice/synth/ only; nothing is recorded or sent anywhere.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import threading
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTH = REPO_ROOT / "data" / "voice" / "synth"

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
         "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def number_words(n: int) -> str:
    """0-999 as words: 21 -> 'twenty one', 105 -> 'one hundred five'."""
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _TENS[tens] + ("" if ones == 0 else " " + _ONES[ones])
    hundreds, rest = divmod(n, 100)
    return _ONES[hundreds] + " hundred" + ("" if rest == 0 else " " + number_words(rest))


def _digits(match: re.Match) -> str:
    s = match.group(0)
    if len(s) >= 4:                      # phone-like: "4729" == "4 7 2 9"
        return " ".join(_ONES[int(c)] for c in s)
    return number_words(int(s))


def normalize_words(text: str) -> list[str]:
    """For scoring only: lowercase, no punctuation, digits as words ("7" == "seven"),
    "3:30" == "three thirty", "10 a.m." == "ten am", "o'clock" kept as one word."""
    t = text.lower().replace("’", "'")
    t = re.sub(r"\b(\d{1,2}):00\b", r"\1", t)              # "8:00" == "8"
    t = re.sub(r"\b(\d{1,2}):(\d{2})\b", r"\1 \2", t)       # "3:30" -> "3 30"
    t = re.sub(r"\b([ap])\.\s?m\b\.?", r"\1m", t)          # "a.m." / "a. m." -> "am"
    t = re.sub(r"\d+", _digits, t)
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    return [w.strip("'") for w in t.split() if w.strip("'")]


def word_errors(ref: str, hyp: str) -> tuple[int, int]:
    """(word edit distance, words in ref) after normalize_words."""
    r, h = normalize_words(ref), normalize_words(hyp)
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hw in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw))
        prev = cur
    return prev[-1], len(r)


def command_name(cmd) -> str:
    """'none', a fixed command name, or 'target:<Label>' (same form as speech_phrases.csv)."""
    if cmd is None:
        return "none"
    return f"target:{cmd.label}" if cmd.name == "target" else cmd.name


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


# ------------------------------------------------------------------ running
def check_clip(row: dict, dictation, settings) -> dict:
    """Run one clip through audio_in -> Dictation -> commands.parse."""
    from scipy.io import wavfile

    from engine.core import topics
    from engine.core.events import EventBus
    from engine.main import play_wav
    from engine.voice import commands
    from engine.voice.audio_in import AudioIn, load_vad
    from engine.voice.dictation import words_to_text
    from engine.voice.mic import MicSettings

    rate, data = wavfile.read(SYNTH / row["file"])
    audio = data.astype("float32") / 32768.0
    bus = EventBus()
    utterances: list = []
    bus.subscribe(topics.UTTERANCE, utterances.append)
    audio_in = AudioIn(bus, settings, mic_settings=MicSettings(-1, row["file"], "file", rate, 1),
                       vad=load_vad(settings.pause_seconds))
    play_wav(audio_in, audio, rate, threading.Event(), realtime=False)

    texts, ms = [], []
    for utt in utterances:
        t0 = time.perf_counter()
        words = dictation.transcribe(utt)
        ms.append((time.perf_counter() - t0) * 1000)
        if words:
            texts.append(words_to_text(words))
    allow_targets = row["mode"] == "pointer"
    parsed = [command_name(commands.parse(t, allow_targets=allow_targets)) for t in texts]

    expected = row["expected"]
    if row["category"] == "dictation" or expected == "dictation":
        correct = len(texts) >= 1 and all(p == "none" for p in parsed)
    elif expected == "none":
        correct = all(p == "none" for p in parsed)
    else:
        correct = parsed == [expected]
    errors, ref_words = word_errors(row["text"], " ".join(texts))
    return {**row, "heard": " | ".join(texts), "parsed": " | ".join(parsed) or "-",
            "correct": correct, "utterances": len(utterances), "ms": ms,
            "errors": errors, "ref_words": ref_words,
            "false_command": expected == "none" and any(p != "none" for p in parsed)}


def summarize(results: list[dict]) -> list[str]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in results:
        groups[(r["category"], r["variant"])].append(r)
        groups[(r["category"], "all")].append(r)
        groups[("all", r["variant"])].append(r)
        groups[("all", "all")].append(r)
    lines = ["| category | variant | clips | correct | false commands | dictation WER | "
             "whisper ms p50 | p95 | split (2+) | none found |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for (cat, var) in sorted(groups, key=lambda k: (k[0] == "all", k[0], k[1] == "all", k[1])):
        rs = groups[(cat, var)]
        ms = [m for r in rs for m in r["ms"]]
        dic = [r for r in rs if r["category"] in ("dictation",) or r["expected"] == "dictation"]
        wer = (sum(r["errors"] for r in dic) / max(1, sum(r["ref_words"] for r in dic))) if dic else None
        lines.append(f"| {cat} | {var} | {len(rs)} | {sum(r['correct'] for r in rs) / len(rs):.0%} | "
                     f"{sum(r['false_command'] for r in rs)} | {'-' if wer is None else f'{wer:.1%}'} | "
                     f"{percentile(ms, 0.5):.0f} | {percentile(ms, 0.95):.0f} | "
                     f"{sum(r['utterances'] >= 2 for r in rs)} | {sum(r['utterances'] == 0 for r in rs)} |")
    return lines


def table(results: list[dict]) -> list[str]:
    lines = ["| id | category | voice | variant | expected text | heard | parsed | ok | utt | whisper ms |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in results:
        cell = lambda s: str(s).replace("|", "/")  # noqa: E731
        lines.append(f"| {r['id']} | {r['category']} | {r['voice']} | {r['variant']} | {cell(r['text'])} | "
                     f"{cell(r['heard']) or '-'} | {cell(r['parsed'])} | {'yes' if r['correct'] else 'NO'} | "
                     f"{r['utterances']} | {', '.join(f'{m:.0f}' for m in r['ms']) or '-'} |")
    return lines


def main(argv: list[str] | None = None) -> int:
    from engine.config import settings
    from engine.voice.dictation import Dictation

    p = argparse.ArgumentParser(description="Score the voice chain on the synthetic clips.")
    p.add_argument("--variant", help="normal, slow, quiet or noisy")
    p.add_argument("--voice", help="e.g. david or zira")
    p.add_argument("--category", help="command, target, dictation, near_miss or demo")
    p.add_argument("--limit", type=int, default=0, help="only the first N matching clips")
    p.add_argument("--report", type=Path, default=SYNTH / "report.md")
    args = p.parse_args(argv)

    manifest = SYNTH / "manifest.csv"
    if not manifest.exists():
        raise SystemExit(f"No {manifest}. Run .\\scripts\\make_test_speech.ps1 and python -m tests.add_noise.")
    with manifest.open(encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if (not args.variant or r["variant"] == args.variant)
                and (not args.voice or r["voice"] == args.voice)
                and (not args.category or r["category"] == args.category)]
    if args.limit:
        rows = rows[:args.limit]
    header = (f"Voice check: {len(rows)} clips, Whisper {settings.whisper_model} on {settings.whisper_device} "
              f"({settings.whisper_compute_type}), pause {settings.pause_seconds} s, local only")
    print(header)

    dictation = Dictation(settings)
    t0 = time.perf_counter()
    dictation.load()
    print(f"Whisper loaded in {time.perf_counter() - t0:.1f} s")
    results = []
    for i, row in enumerate(rows, 1):
        r = check_clip(row, dictation, settings)
        results.append(r)
        if not r["correct"]:
            print(f"  NO  {r['id']} {r['voice']}/{r['variant']}: said {r['text']!r} heard {r['heard']!r} "
                  f"-> {r['parsed']} (expected {r['expected']}, {r['utterances']} utterances)")
        if i % 50 == 0:
            print(f"  ... {i}/{len(rows)} clips, {sum(x['correct'] for x in results)} correct")

    summary = summarize(results)
    print("\n" + "\n".join(summary))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    # machine-readable copy, for merging runs and before/after comparisons
    args.report.with_suffix(".json").write_text(json.dumps({"header": header, "results": results}, indent=1),
                                                encoding="utf-8")
    args.report.write_text("\n".join([f"# {header}", "", f"Generated {time.strftime('%Y-%m-%d %H:%M')}.", "",
                                      "## Summary", "", *summary, "", "## Every clip", "",
                                      *table(results), ""]), encoding="utf-8")
    print(f"\nFull table: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
