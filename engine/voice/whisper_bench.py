"""Benchmark faster-whisper on the demo laptop's GPU.

Records a short clip from a mic into memory (never to disk),
then times a Whisper model on it and reports GPU memory used.

Run from the repo root:
    python -m engine.voice.whisper_bench
    python -m engine.voice.whisper_bench --model base.en --device cpu --compute-type int8 --mic 0
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time

SAMPLE_RATE = 16_000


def gpu_memory_used_mb() -> int | None:
    """Return used GPU memory in MB via nvidia-smi, or None if unavailable."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, encoding="utf-8", check=True,
        ).stdout
    except (subprocess.CalledProcessError, OSError):
        return None
    return int(out.splitlines()[0].strip())


def record_clip(seconds: float, mic: int | None):
    """Record from a mic into memory, as 16 kHz mono (engine/voice/mic.py picks a format that opens)."""
    import numpy as np

    from engine.voice import mic as mic_mod

    print(f"Recording {seconds:.0f} s. Speak a normal sentence now...")
    try:
        audio = mic_mod.record(seconds, preferred=mic)
    except RuntimeError as exc:
        raise SystemExit(str(exc))
    level = float(np.abs(audio).max()) if len(audio) else 0.0
    print(f"Loudest level: {level:.3f} (under 0.01 means the mic heard almost nothing)")
    return audio


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="large-v3-turbo")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--compute-type", default="float16", help="float16 on GPU, int8 on CPU")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--beam-size", type=int, default=1,
                        help="1 = what dictation.py uses; 5 is slower and a bit more accurate")
    parser.add_argument("--mic", type=int, default=None, help="mic number from 'python -m sounddevice'")
    args = parser.parse_args()

    from faster_whisper import WhisperModel

    before_mb = gpu_memory_used_mb()
    t0 = time.perf_counter()
    try:
        model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    except Exception as exc:  # report the CUDA/cuDNN error plainly and stop
        print(f"Could not load {args.model} on {args.device}: {exc}")
        print("Try: python -m engine.voice.whisper_bench --model base.en --device cpu --compute-type int8")
        return 1
    load_s = time.perf_counter() - t0

    audio = record_clip(args.seconds, args.mic)

    times = []
    text = ""
    for i in range(args.runs):
        t0 = time.perf_counter()
        segments, _info = model.transcribe(audio, language="en", beam_size=args.beam_size, word_timestamps=True)
        words = [w for seg in segments for w in seg.words]  # segments is lazy; this runs the model
        times.append(time.perf_counter() - t0)
        if i == 0:
            text = " ".join(f"{w.word.strip()}({w.probability:.2f})" for w in words)

    peak_mb = gpu_memory_used_mb()
    warm = times[1:] or times

    print()
    print(f"Model:        {args.model} on {args.device} ({args.compute_type}), beam size {args.beam_size}")
    print(f"Load time:    {load_s:.1f} s")
    print(f"First run:    {times[0]:.2f} s")
    print(f"Warm average: {sum(warm) / len(warm):.2f} s for a {args.seconds:.0f} s clip (target < 1.5 s on the 5080)")
    if args.device == "cuda" and before_mb is not None and peak_mb is not None:
        print(f"GPU memory:   Whisper uses about {peak_mb - before_mb} MB; UI-Venus 2B needs about 5000 MB more")
    print(f"Heard:        {text or '(nothing)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())