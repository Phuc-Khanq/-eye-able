"""Word error rate (WER) of a Whisper model on the manifests, the way the app decodes.

    python -m engine.voice.finetune.evaluate --model large-v3-turbo               # stock model, test
    python -m engine.voice.finetune.evaluate --model models/whisper-elder-ct2     # fine-tuned
    python -m engine.voice.finetune.evaluate --tiny                               # smoke test (CPU)

Decoding is THE SAME as the app: each clip goes through engine.voice.dictation.Dictation
.transcribe (faster-whisper, language="en", beam_size=1, condition_on_previous_text=False,
same silence and hallucination filters). Text is normalized with Whisper's
EnglishTextNormalizer and scored with jiwer (corpus WER: all errors / all reference words).

Backup backend (--backend hf): Hugging Face transformers with the same greedy decoding,
for when faster-whisper can't use the GPU. Results say which backend ran.

Prints one table (clips, words, WER) per group and per TORGO test speaker, the warm latency
for a 5 s clip, and saves the numbers (no transcripts) as JSON in checkpoints/.
Use --split test only for the stock model and a finished candidate; tune on --split dev.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from engine.voice.finetune import common

LATENCY_RUNS = 5


def _audio_iter(clips, data_dir):
    """Load the next clips on a background thread while Whisper works."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        yield from pool.map(lambda c: common.load_audio(c.audio_path(data_dir)), clips)


def five_second_clip(clips, data_dir):
    import numpy as np

    parts, total = [], 0
    for c in sorted(clips, key=lambda c: -c.seconds)[:10]:
        a = common.load_audio(c.audio_path(data_dir))
        parts.append(a)
        total += len(a)
        if total >= 5 * common.SAMPLE_RATE:
            break
    audio = np.concatenate(parts) if parts else np.zeros(0, np.float32)
    if len(audio) < 5 * common.SAMPLE_RATE:
        audio = np.pad(audio, (0, 5 * common.SAMPLE_RATE - len(audio)))
    return audio[:5 * common.SAMPLE_RATE].astype(np.float32)


class CT2Backend:
    """faster-whisper through the app's own Dictation class."""

    name = "faster-whisper"

    def __init__(self, model: str, device: str, compute_type: str) -> None:
        common.preload_cuda12_libs()
        from faster_whisper import WhisperModel  # heavy: import only when used

        from engine.voice.dictation import Dictation, resolve_model, words_to_text

        self.label = resolve_model(model)
        self._dictation = Dictation(model=WhisperModel(self.label, device=device,
                                                       compute_type=compute_type))
        self._join = words_to_text

    def transcribe_many(self, audios) -> list[str]:
        return [self._join(self._dictation.transcribe(a)) for a in audios]

    def transcribe_one(self, audio) -> str:
        return self._join(self._dictation.transcribe(audio))


class HFBackend:
    """transformers Whisper (+ optional LoRA adapter), same greedy decoding."""

    name = "transformers"

    def __init__(self, base: str, adapter: Path | None, device: str) -> None:
        from engine.voice.finetune import hf

        self.device = hf.pick_device(device)
        self.label = f"{base}" + (f" + {adapter}" if adapter else "")
        self._hf = hf
        self.processor = hf.load_processor(base)
        self.model = hf.load_model(base, adapter, self.device)

    def transcribe_many(self, audios) -> list[str]:
        return self._hf.transcribe(self.model, self.processor, list(audios), self.device)

    def transcribe_one(self, audio) -> str:
        return self.transcribe_many([audio])[0]


def make_backend(args):
    if args.backend == "hf":
        base = args.hf_base or (common.TINY_MODEL if args.tiny else common.BASE_MODEL)
        return HFBackend(base, args.adapter, args.device)
    device = args.device
    if device == "auto":
        try:
            common.preload_cuda12_libs()
            import ctranslate2

            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"
    compute = args.compute_type or ("float16" if device == "cuda" else "int8")
    return CT2Backend(args.model, device, compute)


def score(clips, hyps, split_name: str) -> dict:
    """WER per group and per TORGO speaker."""
    by_group: dict[str, list] = {}
    by_speaker: dict[str, list] = {}
    for c, h in zip(clips, hyps):
        by_group.setdefault(c.group, []).append((c.text, h))
        if c.source == "torgo":
            by_speaker.setdefault(c.speaker, []).append((c.text, h))
    groups = {g: common.wer_table(rows) for g, rows in by_group.items()}
    dys = [(c.text, h) for c, h in zip(clips, hyps) if c.group in common.DYS_GROUPS]
    if dys:
        groups["dys_all"] = common.wer_table(dys)
    speakers = {s: common.wer_table(rows) for s, rows in sorted(by_speaker.items())}
    return {"split": split_name, "groups": groups, "speakers": speakers}


def print_table(result: dict, log) -> None:
    names = dict(common.GROUPS, dys_all="Dysarthric, all")
    rows = []
    for g in list(common.GROUPS) + ["dys_all"]:
        r = result["groups"].get(g)
        if r and r["n"]:
            rows.append([names[g], r["n"], r["words"], r["wer"]])
    for s, r in result["speakers"].items():
        rows.append([f"  TORGO speaker {s}", r["n"], r["words"], r["wer"]])
    log(f"\n{result['model']} ({result['backend']}, {result['device']}), split={result['split']}\n"
        + common.fmt_table(rows, ["group", "clips", "words", "WER %"]))
    if result.get("latency_5s_ms") is not None:
        log(f"Warm latency for a 5 s clip: {result['latency_5s_ms']:.0f} ms "
            f"(average of {LATENCY_RUNS})")


def run(args, log=None) -> dict:
    log = log or common.Log()
    data_dir = Path(args.data_dir)
    clips = [c for c in common.read_manifest(common.manifest_path(data_dir, args.tiny))
             if c.split == args.split]
    if args.groups:
        clips = [c for c in clips if c.group in args.groups]
    if args.limit:
        per: dict[str, int] = {}
        kept = []
        for c in clips:
            if per.get(c.group, 0) < args.limit:
                per[c.group] = per.get(c.group, 0) + 1
                kept.append(c)
        clips = kept
    if not clips:
        raise SystemExit(f"No clips in split '{args.split}'")
    log(f"Evaluating on {len(clips)} {args.split} clips ({sum(c.seconds for c in clips) / 60:.0f} min)")

    backend = make_backend(args)
    device = getattr(backend, "device", None) or (args.device if args.device != "auto" else "")
    if isinstance(backend, CT2Backend):
        device = backend._dictation._model.model.device
    t0 = time.perf_counter()
    hyps: list[str] = []
    batch: list = []
    done = 0
    for audio in _audio_iter(clips, data_dir):
        batch.append(audio)
        if len(batch) == 16:
            hyps += backend.transcribe_many(batch)
            batch = []
            done += 16
            if done % 400 == 0:
                log(f"  {done}/{len(clips)} clips")
    if batch:
        hyps += backend.transcribe_many(batch)
    elapsed = time.perf_counter() - t0

    clip5 = five_second_clip(clips, data_dir)
    backend.transcribe_one(clip5)                                   # warm up
    times = []
    for _ in range(LATENCY_RUNS):
        t1 = time.perf_counter()
        backend.transcribe_one(clip5)
        times.append(time.perf_counter() - t1)

    result = score(clips, hyps, args.split)
    result.update({"model": backend.label, "backend": backend.name, "device": str(device),
                   "latency_5s_ms": round(1000 * sum(times) / len(times), 1),
                   "seconds": round(elapsed, 1), "clips": len(clips),
                   "time": time.strftime("%Y-%m-%d %H:%M:%S")})
    print_table(result, log)
    out = args.out or (Path(args.checkpoints) / f"eval-{Path(str(args.model)).name}-{args.split}"
                       f"-{time.strftime('%Y%m%d-%H%M%S')}.json")
    common.save_json(Path(out), result)
    log(f"Saved {out}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    common.data_dir_arg(parser)
    parser.add_argument("--model", default=None,
                        help="faster-whisper model name or folder (default large-v3-turbo; tiny with --tiny)")
    parser.add_argument("--split", default="test", choices=["test", "dev", "train"])
    parser.add_argument("--device", default="auto", help="auto, cuda or cpu")
    parser.add_argument("--compute-type", default=None, help="default float16 on cuda, int8 on cpu")
    parser.add_argument("--backend", default="ct2", choices=["ct2", "hf"],
                        help="ct2 = faster-whisper like the app (default); hf = transformers backup")
    parser.add_argument("--hf-base", default=None, help="hf backend: base model (default openai/whisper-large-v3-turbo)")
    parser.add_argument("--adapter", type=Path, default=None, help="hf backend: LoRA adapter folder")
    parser.add_argument("--groups", nargs="*", default=None, help="only these groups (e.g. dys_unseen)")
    parser.add_argument("--limit", type=int, default=0, help="at most this many clips per group (0 = all)")
    parser.add_argument("--checkpoints", type=Path, default=common.DEFAULT_CHECKPOINTS)
    parser.add_argument("--out", type=Path, default=None, help="where to save the JSON")
    parser.add_argument("--tiny", action="store_true", help="smoke test: whisper tiny, tiny.csv, CPU")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.model is None:
        args.model = common.TINY_CT2 if args.tiny else common.BASE_CT2
    if args.tiny and args.device == "auto":
        args.device = "cpu"
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
