"""Build the manifests: one CSV row per clip (path, text, speaker, group, split, ...).

    python -m engine.voice.finetune.prepare            # everything found in data/voice
    python -m engine.voice.finetune.prepare --tiny     # 40 LibriSpeech clips for the smoke test

Writes data/voice/manifests/all.csv (or tiny.csv), never into the code folders.

Splits come from split.json (speaker IDs only):
  TORGO        by speaker: train / dev (M05) / test (F04, M01)
  LibriSpeech  test-clean = test; dev-clean speakers split in half: train (the typical-speech
               mix) and dev
  Personal     80/20 by sentence (data/voice/personal/ with transcripts.csv: file,text)

TORGO groups, decided against the TRAINING text (TORGO repeats prompts across speakers):
  dys_unseen   3+ words and the sentence is not in the training text   <- primary number
  dys_seen     3+ words and the sentence is also in the training text
  dys_words    1 or 2 words
Dropped TORGO prompts: bracketed instructions ("[say Ah for 5 seconds]"), picture
descriptions (input/images/*.jpg), "xxx", empty files. A word with a hint in brackets,
"lead [as in I will lead you]", keeps just the word.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from engine.voice.finetune import common
from engine.voice.finetune.common import Clip


def clean_torgo_prompt(raw: str) -> str | None:
    """Return Whisper-style text, or None if the prompt is not words to transcribe."""
    text = " ".join(raw.split())
    if not text:
        return None
    if "input/images" in text or re.search(r"\.(jpg|jpeg|png|bmp)\b", text, re.I):
        return None                                   # picture description
    text = re.sub(r"\[[^\]]*\]", " ", text)            # instructions and hints
    text = " ".join(text.split())
    if not text or "[" in text or "]" in text:
        return None
    if re.search(r"\bx{3,}\b", text, re.I):            # "xxx" = unintelligible / no prompt
        return None
    if not common.words(text):
        return None
    return common.whisper_style(text)


def text_key(text: str) -> str:
    """Same sentence = same lowercase words, ignoring punctuation."""
    return " ".join(w.lower() for w in common.words(text))


def _rel(path: Path, data_dir: Path) -> str:
    return path.relative_to(data_dir).as_posix()


def _seconds(path: Path) -> float | None:
    try:
        return common.audio_seconds(path)
    except Exception:
        return None


# ----------------------------------------------------------------------- TORGO
def scan_torgo(data_dir: Path, split: dict, stats: Counter) -> list[Clip]:
    root = data_dir / "torgo"
    if not root.is_dir():
        return []
    where = {spk: name for name in ("train", "dev", "test") for spk in split["torgo"][name]}
    raw: list[tuple[str, str, Path]] = []                # speaker, text, audio
    for prompt in sorted(root.glob("*/*/prompts/*.txt")):
        spk = prompt.parts[-4]
        if spk not in where:
            continue
        stats["torgo prompts"] += 1
        text = clean_torgo_prompt(prompt.read_text(encoding="utf-8", errors="replace"))
        if text is None:
            stats["torgo dropped: not words"] += 1
            continue
        session = prompt.parent.parent
        audio = session / "wav_arrayMic" / f"{prompt.stem}.wav"
        if not audio.exists():
            audio = session / "wav_headMic" / f"{prompt.stem}.wav"
            if audio.exists():
                stats["torgo head-mic fallback"] += 1
        if not audio.exists():
            stats["torgo dropped: no audio"] += 1
            continue
        raw.append((spk, text, audio))

    train_text = {text_key(t) for spk, t, _ in raw if where[spk] == "train"}
    min_words = split["sentence_min_words"]
    clips = []
    for spk, text, audio in raw:
        secs = _seconds(audio)
        if secs is None or not (common.MIN_CLIP_S <= secs <= common.MAX_CLIP_S):
            stats["torgo dropped: unreadable, too short or too long"] += 1
            continue
        n = len(common.words(text))
        if n < min_words:
            group = "dys_words"
        elif text_key(text) in train_text:
            group = "dys_seen"
        else:
            group = "dys_unseen"
        clips.append(Clip(_rel(audio, data_dir), text, spk, group, where[spk], "torgo",
                          round(secs, 2)))
    return clips


# ----------------------------------------------------------------- LibriSpeech
def _libri_genders(root: Path) -> dict[str, str]:
    genders = {}
    f = root / "SPEAKERS.TXT"
    if f.exists():
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith(";") or "|" not in line:
                continue
            parts = [p.strip() for p in line.split("|")]
            genders[parts[0]] = parts[1]
    return genders


def _libri_subset(root: Path, subset: str) -> dict[str, list[tuple[Path, str]]]:
    """speaker -> [(flac, text)]"""
    by_spk: dict[str, list[tuple[Path, str]]] = defaultdict(list)
    for trans in sorted((root / subset).glob("*/*/*.trans.txt")):
        for line in trans.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            utt, _, text = line.partition(" ")
            flac = trans.parent / f"{utt}.flac"
            if flac.exists():
                by_spk[utt.split("-")[0]].append((flac, common.whisper_style(text)))
    return by_spk


def _libri_clips(items, spk, split_name, data_dir, stats) -> list[Clip]:
    out = []
    for flac, text in items:
        secs = _seconds(flac)
        if secs is None or not (common.MIN_CLIP_S <= secs <= common.MAX_CLIP_S):
            stats["librispeech dropped: too long or unreadable"] += 1
            continue
        out.append(Clip(_rel(flac, data_dir), text, spk, "librispeech", split_name,
                        "librispeech", round(secs, 2)))
    return out


def scan_librispeech(data_dir: Path, split: dict, stats: Counter) -> list[Clip]:
    root = data_dir / "librispeech"
    rng = random.Random(split["seed"])
    rules = split["librispeech"]
    clips: list[Clip] = []
    for spk, items in sorted(_libri_subset(root, "test-clean").items()):
        clips += _libri_clips(items, spk, "test", data_dir, stats)

    dev = _libri_subset(root, "dev-clean")
    if dev:
        genders = _libri_genders(root)
        halves = {"train": [], "dev": []}
        for gender in sorted({genders.get(s, "?") for s in dev}):
            same = sorted(s for s in dev if genders.get(s, "?") == gender)
            for i, spk in enumerate(same):
                halves["train" if i % 2 == 0 else "dev"].append(spk)
        for spk in halves["train"]:
            items = list(dev[spk])
            rng.shuffle(items)
            clips += _libri_clips(items[:rules["train_max_clips_per_speaker"]], spk, "train",
                                  data_dir, stats)
        dev_items = [(spk, it) for spk in halves["dev"] for it in dev[spk]]
        rng.shuffle(dev_items)
        for spk, it in dev_items[:rules["dev_max_clips"]]:
            clips += _libri_clips([it], spk, "dev", data_dir, stats)
    return clips


def tiny_librispeech(data_dir: Path, stats: Counter) -> list[Clip]:
    """Smoke test: short test-clean clips from different speakers: 20 train, 10 dev, 10 test."""
    by_spk = _libri_subset(data_dir / "librispeech", "test-clean")
    if not by_spk:
        raise SystemExit("No LibriSpeech test-clean found. Run: python -m engine.voice.finetune.get_data --tiny")
    speakers = sorted(by_spk)
    plan = [("train", speakers[0:2], 10), ("dev", speakers[2:3], 10), ("test", speakers[3:4], 10)]
    clips = []
    for split_name, spks, per_spk in plan:
        for spk in spks:
            short = []
            for flac, text in by_spk[spk]:
                secs = _seconds(flac)
                if secs is not None and 1.0 <= secs <= 6.0:
                    short.append(Clip(_rel(flac, data_dir), text, spk, "librispeech", split_name,
                                      "librispeech", round(secs, 2)))
                if len(short) == per_spk:
                    break
            clips += short
    return clips


# -------------------------------------------------------------------- personal
def scan_personal(data_dir: Path, split: dict, stats: Counter) -> list[Clip]:
    root = data_dir / "personal"
    table = root / "transcripts.csv"
    if not table.exists():
        return []
    rows = []
    with open(table, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            audio = root / r["file"]
            text = (r.get("text") or "").strip()
            if audio.exists() and text:
                rows.append((audio, common.whisper_style(text)))
            else:
                stats["personal dropped: missing file or text"] += 1
    sentences = sorted({text_key(t) for _, t in rows})
    random.Random(split["seed"]).shuffle(sentences)
    n_test = max(1, round(len(sentences) * split["personal"]["test_fraction"])) if sentences else 0
    test_sentences = set(sentences[:n_test])
    clips = []
    for audio, text in rows:
        secs = _seconds(audio)
        if secs is None or not (common.MIN_CLIP_S <= secs <= common.MAX_CLIP_S):
            stats["personal dropped: unreadable, too short or too long"] += 1
            continue
        s = "test" if text_key(text) in test_sentences else "train"
        clips.append(Clip(_rel(audio, data_dir), text, "personal", "personal", s, "personal",
                          round(secs, 2)))
    return clips


# ------------------------------------------------------------------------ main
def summarize(clips: list[Clip], log) -> None:
    table = defaultdict(lambda: [0, 0.0])
    for c in clips:
        key = (c.source, c.split, c.group)
        table[key][0] += 1
        table[key][1] += c.seconds / 60
    rows = [[src, sp, grp, n, f"{mins:.1f}"] for (src, sp, grp), (n, mins) in sorted(table.items())]
    rows.append(["total", "", "", len(clips), f"{sum(c.seconds for c in clips) / 60:.1f}"])
    log("\n" + common.fmt_table(rows, ["source", "split", "group", "clips", "minutes"]))
    spk = defaultdict(lambda: [0, 0.0, ""])
    for c in clips:
        if c.source == "torgo":
            spk[c.speaker][0] += 1
            spk[c.speaker][1] += c.seconds / 60
            spk[c.speaker][2] = c.split
    if spk:
        rows = [[s, sp, n, f"{m:.1f}"] for s, (n, m, sp) in sorted(spk.items())]
        log("\n" + common.fmt_table(rows, ["TORGO speaker", "split", "clips", "minutes"]))


def check_disjoint(clips: list[Clip]) -> None:
    """No speaker (TORGO, LibriSpeech) and no personal sentence in more than one split."""
    seen: dict[tuple[str, str], str] = {}
    for c in clips:
        key = (c.source, text_key(c.text) if c.source == "personal" else c.speaker)
        if seen.setdefault(key, c.split) != c.split:
            raise SystemExit(f"Split leak: {key} is in both {seen[key]} and {c.split}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    common.data_dir_arg(parser)
    parser.add_argument("--tiny", action="store_true", help="40 short LibriSpeech clips (smoke test)")
    args = parser.parse_args(argv)
    data_dir = Path(args.data_dir).resolve()
    log = common.Log()
    split = common.load_split()
    stats: Counter = Counter()

    if args.tiny:
        clips = tiny_librispeech(data_dir, stats)
    else:
        clips = (scan_torgo(data_dir, split, stats) + scan_librispeech(data_dir, split, stats)
                 + scan_personal(data_dir, split, stats))
        if not any(c.source == "torgo" for c in clips):
            log("Warning: no TORGO found in", data_dir / "torgo", "(run get_data on Kaggle)")
    if not clips:
        log("No clips found. Run: python -m engine.voice.finetune.get_data")
        return 1
    check_disjoint(clips)
    out = common.manifest_path(data_dir, args.tiny)
    common.write_manifest(out, clips)
    for k, v in sorted(stats.items()):
        log(f"{k}: {v}")
    summarize(clips, log)
    log(f"Wrote {len(clips)} clips to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
