"""Adds a "noisy" copy of every normal synthetic clip: the clip plus pink noise at about 15 dB SNR.

Run after scripts/make_test_speech.ps1:  python -m tests.add_noise
Reads and appends to data/voice/synth/manifest.csv. Synthetic speech only (never a real voice),
written only under data/ (git-ignored).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SYNTH = ROOT / "data" / "voice" / "synth"
SNR_DB = 15.0


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """1/f noise: white noise shaped in the frequency domain, scaled to RMS 1."""
    spectrum = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(spectrum))
    spectrum[1:] /= np.sqrt(f[1:])
    spectrum[0] = 0
    noise = np.fft.irfft(spectrum, n)
    return noise / (np.sqrt(np.mean(noise ** 2)) + 1e-12)


def add_noise(speech: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Mix noise in at snr_db below the speech level (measured on the louder half of the clip)."""
    loud = np.sort(np.abs(speech))[len(speech) // 2:]
    speech_rms = float(np.sqrt(np.mean(loud ** 2))) if len(loud) else 0.0
    noise = pink_noise(len(speech), rng) * speech_rms / (10 ** (snr_db / 20))
    return np.clip(speech + noise, -1.0, 1.0).astype(np.float32)


def main() -> None:
    from scipy.io import wavfile

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--snr", type=float, default=SNR_DB)
    args = p.parse_args()
    manifest = SYNTH / "manifest.csv"
    if not manifest.exists():
        raise SystemExit(f"No {manifest}. Run .\\scripts\\make_test_speech.ps1 first.")
    with manifest.open(encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["variant"] != "noisy"]
    rng = np.random.default_rng(0)
    noisy = []
    for r in rows:
        if r["variant"] != "normal":
            continue
        rate, data = wavfile.read(SYNTH / r["file"])
        speech = data.astype(np.float32) / 32768.0
        out = add_noise(speech, args.snr, rng)
        name = r["file"].replace("__normal.wav", "__noisy.wav")
        wavfile.write(SYNTH / name, rate, (out * 32767).astype(np.int16))
        noisy.append({**r, "variant": "noisy", "file": name})
    with manifest.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows + noisy)
    print(f"Added {len(noisy)} noisy clips at {args.snr:.0f} dB SNR; manifest has {len(rows) + len(noisy)} rows")


if __name__ == "__main__":
    main()
