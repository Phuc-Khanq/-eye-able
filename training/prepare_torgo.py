"""Download TORGO (dysarthric + typical speech) and split it by speaker.

Run on the laptop that will train (needs internet once):
    .\\.venv\\Scripts\\python.exe -m training.prepare_torgo

Writes data/torgo/{train,dev,test} (git ignores data/). Never commit it.

TORGO terms: free for academic, non-profit use; cite Rudzicz et al. (2012).
Don't upload the data or a model trained on it anywhere public.

Why split by speaker: if the same person is in training and test, the score
looks great but says nothing about a new user. Test speakers are never trained on.
Test rows also get `seen_text`: True when the same sentence was read by a training
speaker, so evaluate.py can report results on unseen sentences separately.
"""

from __future__ import annotations

import argparse
import io
import random
import sys
from collections import Counter
from pathlib import Path

from training.torgo_utils import (DEFAULT_TEST_SPEAKERS, MAX_SECONDS, MIN_SECONDS,
                                  is_dysarthric_speaker, is_usable_transcript, normalize,
                                  speaker_of)

ROOT = Path(__file__).resolve().parent.parent
HF_DATASET = "abnerh/TORGO-database"
SAMPLE_RATE = 16_000


def wav_bytes_to_16k(data: bytes):
    """Decode WAV bytes with soundfile (no torchcodec / FFmpeg needed on Windows)."""
    import numpy as np
    import soundfile as sf

    audio, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(SAMPLE_RATE, rate)
        audio = resample_poly(audio, SAMPLE_RATE // g, rate // g).astype(np.float32)
    return audio


def split_rows(examples, test_speakers, dev_fraction: float = 0.05, seed: int = 0):
    """Clean TORGO examples and split them by speaker.

    examples: dicts like the Hugging Face rows, with audio = {"bytes", "path"}
    (audio cast with decode=False), "transcription" and "speech_status".
    Returns (train, dev, test, dropped Counter).
    """
    rows = {"train": [], "test": []}
    dropped = Counter()
    for ex in examples:
        audio = ex["audio"]
        speaker = speaker_of(audio.get("path") or "")
        text = ex.get("transcription") or ""
        if speaker is None:
            dropped["no speaker in file name"] += 1
            continue
        if not is_usable_transcript(text):
            dropped["prompt, not speech"] += 1
            continue
        data = audio.get("bytes")
        if not data and audio.get("path") and Path(audio["path"]).is_file():
            data = Path(audio["path"]).read_bytes()
        try:
            seconds = len(wav_bytes_to_16k(data)) / SAMPLE_RATE
        except Exception:
            dropped["unreadable audio"] += 1
            continue
        if not MIN_SECONDS <= seconds <= MAX_SECONDS:
            dropped["too short or too long"] += 1
            continue
        status = str(ex.get("speech_status") or "").lower()
        dysarthric = status.startswith("dys") if status else is_dysarthric_speaker(speaker)
        row = {"wav": data, "text": text.strip(), "speaker": speaker,
               "dysarthric": dysarthric, "seconds": round(seconds, 2)}
        rows["test" if speaker in test_speakers else "train"].append(row)

    train_texts = {normalize(r["text"]) for r in rows["train"]}
    for r in rows["test"]:
        r["seen_text"] = normalize(r["text"]) in train_texts

    train = rows["train"]
    random.Random(seed).shuffle(train)
    n_dev = max(1, int(len(train) * dev_fraction)) if len(train) > 1 else 0
    dev, train = train[:n_dev], train[n_dev:]
    for r in train + dev:
        r["seen_text"] = True
    return train, dev, rows["test"], dropped


def save_splits(train, dev, test, out: Path) -> None:
    from datasets import Dataset, DatasetDict

    DatasetDict({"train": Dataset.from_list(train), "dev": Dataset.from_list(dev),
                 "test": Dataset.from_list(test)}).save_to_disk(str(out))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/torgo")
    p.add_argument("--test-speakers", default=",".join(DEFAULT_TEST_SPEAKERS))
    p.add_argument("--dev-fraction", type=float, default=0.05,
                   help="share of training rows kept aside to watch the loss")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from datasets import Audio, load_dataset

    print(f"Downloading {HF_DATASET} (first time only)...")
    try:
        ds = load_dataset(HF_DATASET, split="train")
    except Exception as exc:
        print(f"Could not download: {exc}\nIf it says 401/gated: run `hf auth login`, "
              f"accept the terms on huggingface.co/datasets/{HF_DATASET}, and retry.")
        return 1
    ds = ds.cast_column("audio", Audio(decode=False))   # we decode with soundfile ourselves

    test_speakers = {s.strip() for s in args.test_speakers.split(",") if s.strip()}
    train, dev, test, dropped = split_rows(ds, test_speakers, args.dev_fraction, args.seed)
    if not test:
        print(f"None of the test speakers {sorted(test_speakers)} were found. "
              "Pass --test-speakers with codes from the table below.")
    out = ROOT / args.out
    save_splits(train, dev, test, out)

    print("\nRows per speaker (D = dysarthric, T = typical, * = test speaker):")
    counts = Counter(r["speaker"] for r in train + dev + test)
    for spk in sorted(counts):
        tag = "D" if is_dysarthric_speaker(spk) else "T"
        print(f"  {spk:<5} {tag} {'*' if spk in test_speakers else ' '} {counts[spk]:>6}")
    unseen = sum(not r["seen_text"] for r in test)
    print(f"\ntrain {len(train)}  dev {len(dev)}  test {len(test)} "
          f"({unseen} test rows have sentences no training speaker read)")
    if dropped:
        print("dropped: " + ", ".join(f"{k} {v}" for k, v in dropped.items()))
    print(f"saved to {out}")
    return 0 if test else 1


if __name__ == "__main__":
    sys.exit(main())
