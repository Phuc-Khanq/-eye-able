"""The mouth gate with REAL speech recognition: a "user" sentence then a "teammate" sentence
(two synthetic voices) through real audio_in (Silero VAD) and real Whisper, with a fake mouth
that moves only while the user speaks. Only the user's words may come out.

Slow (real Whisper), so it isn't a unittest. Run from the repo root after making the clips
(scripts/make_test_speech.ps1):
    python -m tests.mouth_dryrun                     # 10 sentence pairs, David = user, Zira = teammate
    python -m tests.mouth_dryrun --swap              # Zira = user, David = teammate
    python -m tests.mouth_dryrun --gap 2.0           # a long gap: two utterances, the second dropped whole
Exit code 0 only if every run passes.

Pass means: no word from the teammate's sentence (that isn't also in the user's) comes out, and
at least MIN_USER_RECALL of the user's words do.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from engine.core import topics
from engine.voice.voice_check import SYNTH, normalize_words
from tests.helpers import FakeBus
from tests.test_mouth_gate import FakeMouth

CHUNK_S = 0.032          # mic chunk length, like the real stream
LEAD_S = 0.5             # silence before the user starts
MIN_USER_RECALL = 0.8    # share of the user's (normalized) words that must survive


class Clock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


def read(name: str) -> np.ndarray:
    from scipy.io import wavfile

    rate, data = wavfile.read(SYNTH / name)
    assert rate == 16000, name
    return data.astype(np.float32) / 32768.0


def phrases() -> dict[str, str]:
    import csv

    with (SYNTH / "manifest.csv").open(encoding="utf-8", newline="") as f:
        return {r["id"]: r["text"] for r in csv.DictReader(f)}


def run_pair(user_id: str, mate_id: str, user_voice: str, mate_voice: str, gap_s: float,
             dictation, settings) -> dict:
    from engine.voice.audio_in import AudioIn, load_vad
    from engine.voice.dictation import DictationWorker
    from engine.voice.mic import MicSettings

    user = read(f"{user_id}__{user_voice}__normal.wav")
    mate = read(f"{mate_id}__{mate_voice}__normal.wav")
    audio = np.concatenate([np.zeros(int(LEAD_S * 16000), np.float32), user,
                            np.zeros(int(gap_s * 16000), np.float32), mate,
                            np.zeros(int((settings.pause_seconds + 1.0) * 16000), np.float32)])

    clock = Clock()
    t0_ms = clock.t * 1000
    user_ms = (t0_ms + LEAD_S * 1000, t0_ms + (LEAD_S + len(user) / 16000) * 1000)
    mouth = FakeMouth(moving=[user_ms])                  # face always in view; moves only for the user

    bus, log = FakeBus(), []
    audio_in = AudioIn(bus, settings, mic_settings=MicSettings(-1, "synthetic", "file", 16000, 1),
                       vad=load_vad(settings.pause_seconds), clock=clock)
    worker = DictationWorker(bus, dictation, mouth=mouth, settings=settings, clock=clock, log=log.append)
    step = int(CHUNK_S * 16000)
    for i in range(0, len(audio), step):                  # as the mic would deliver it, in real time
        part = audio[i:i + step]
        clock.t += len(part) / 16000
        audio_in.process(part, clock.t * 1000)
    spans, utts = bus.of(topics.SPEECH_SPAN), bus.of(topics.UTTERANCE)
    assert len(spans) == len(utts)
    for span, utt in zip(spans, utts):                   # what the Whisper thread does, in order
        worker.process(utt, span)

    texts = phrases()
    user_words = set(normalize_words(texts[user_id]))
    mate_only = set(normalize_words(texts[mate_id])) - user_words
    out = [w for t in bus.of(topics.TRANSCRIPT) for w in normalize_words(t["text"])]
    leaked = [w for w in out if w in mate_only]
    recall = len(user_words & set(out)) / max(1, len(user_words))
    return {"ok": not leaked and recall >= MIN_USER_RECALL, "out": " | ".join(t["text"] for t in
            bus.of(topics.TRANSCRIPT)), "leaked": leaked, "recall": recall, "utterances": len(utts),
            "user": texts[user_id], "mate": texts[mate_id], "mouth_log": log}


def main(argv: list[str] | None = None) -> int:
    from engine.config import settings
    from engine.voice.dictation import Dictation

    p = argparse.ArgumentParser(description="The mouth gate with real Whisper on two synthetic voices.")
    p.add_argument("--runs", type=int, default=10, help="sentence pairs (up to 15)")
    p.add_argument("--swap", action="store_true", help="Zira is the user, David the teammate")
    p.add_argument("--gap", type=float, default=0.4, help="silence between the two sentences (s)")
    args = p.parse_args(argv)
    user_voice, mate_voice = ("zira", "david") if args.swap else ("david", "zira")

    dictation = Dictation(settings)
    t0 = time.perf_counter()
    dictation.load()
    print(f"Whisper {settings.whisper_model} on {settings.whisper_device} loaded in {time.perf_counter() - t0:.1f} s; "
          f"user {user_voice}, teammate {mate_voice}, gap {args.gap} s, pause {settings.pause_seconds} s")
    passed = 0
    for i in range(1, args.runs + 1):
        user_id = f"dictation_{i:03d}"
        mate_id = f"dictation_{(i + 4) % 15 + 1:03d}"        # a different sentence each time
        r = run_pair(user_id, mate_id, user_voice, mate_voice, args.gap, dictation, settings)
        passed += r["ok"]
        print(f"run {i}: {'PASS' if r['ok'] else 'FAIL'}  user said {r['user']!r}, teammate said {r['mate']!r}")
        print(f"    out: {r['out']!r}  ({r['utterances']} utterance(s), user recall {r['recall']:.0%}"
              + (f", LEAKED {r['leaked']}" if r["leaked"] else "") + ")")
        for line in r["mouth_log"]:
            print(f"    {line}")
    print(f"{passed}/{args.runs} runs passed")
    return 0 if passed == args.runs else 1


if __name__ == "__main__":
    raise SystemExit(main())
