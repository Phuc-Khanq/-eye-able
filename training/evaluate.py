"""Compare stock Whisper with the adapted model on speakers it never trained on.

Run on the RTX 5080 laptop after finetune_whisper.py:
    .\\.venv\\Scripts\\python.exe -m training.evaluate
    .\\.venv\\Scripts\\python.exe -m training.evaluate --limit 200     # quick look

Both models transcribe through engine/voice/dictation.py, with the same settings the
app uses, so the numbers are what users would get. Prints word error rate (WER, lower is
better) for dysarthric and typical speakers, on all test sentences and on sentences no
training speaker read, then a verdict: SHIP or DON'T SHIP.

The adapted model ships only if dysarthric WER drops by at least 2 points and typical
speech gets no more than 1 point worse (training/torgo_utils.Gate).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from training.torgo_utils import Gate, wer

ROOT = Path(__file__).resolve().parent.parent


def transcribe_all(model_name: str, rows, device: str, compute: str):
    from engine.voice.dictation import Dictation, words_to_text
    from training.prepare_torgo import wav_bytes_to_16k

    settings = SimpleNamespace(whisper_model=model_name, whisper_device=device,
                               whisper_compute_type=compute)
    d = Dictation(settings=settings)
    d.warm_up()
    hyps, t0 = [], time.perf_counter()
    for i, r in enumerate(rows):
        hyps.append(words_to_text(d.transcribe(wav_bytes_to_16k(r["wav"]))))
        if (i + 1) % 200 == 0:
            print(f"  {model_name}: {i + 1}/{len(rows)}")
    ms = (time.perf_counter() - t0) * 1000 / max(1, len(rows))
    return hyps, ms


def score(rows, hyps) -> dict:
    groups = defaultdict(list)
    for r, h in zip(rows, hyps):
        kind = "dysarthric" if r["dysarthric"] else "typical"
        groups[kind].append((r["text"], h))
        if not r["seen_text"]:
            groups[f"{kind}, unseen sentences"].append((r["text"], h))
        groups[f"speaker {r['speaker']}"].append((r["text"], h))
    return {k: {"wer": wer(v), "n": len(v)} for k, v in sorted(groups.items())}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/torgo")
    p.add_argument("--base", default="large-v3-turbo")
    p.add_argument("--adapted", default="models/whisper-turbo-torgo-ct2")
    p.add_argument("--device", default="cuda")
    p.add_argument("--compute-type", default="float16")
    p.add_argument("--limit", type=int, default=0, help="score only the first N test rows")
    args = p.parse_args()

    from datasets import load_from_disk

    rows = load_from_disk(str(ROOT / args.data))["test"]
    if args.limit:
        rows = rows.select(range(min(args.limit, len(rows))))
    rows = list(rows)
    print(f"{len(rows)} test utterances from speakers never trained on")

    results = {}
    for label, name in (("base", args.base), ("adapted", args.adapted)):
        hyps, ms = transcribe_all(name, rows, args.device, args.compute_type)
        results[label] = {"model": name, "ms_per_utterance": round(ms, 1),
                          "groups": score(rows, hyps), "samples": []}
        results[label]["samples"] = [{"ref": r["text"], "hyp": h} for r, h in zip(rows[:8], hyps[:8])]

    base, new = results["base"]["groups"], results["adapted"]["groups"]
    print(f"\n{'group':<34}{'n':>6}{'stock':>9}{'adapted':>9}")
    for g in base:
        print(f"{g:<34}{base[g]['n']:>6}{base[g]['wer']:>9.1%}{new[g]['wer']:>9.1%}")
    print(f"{'ms per utterance':<34}{'':>6}{results['base']['ms_per_utterance']:>9.0f}"
          f"{results['adapted']['ms_per_utterance']:>9.0f}")

    print("\nSamples (reference | stock | adapted):")
    for b, a in zip(results["base"]["samples"], results["adapted"]["samples"]):
        print(f"  {b['ref']!r} | {b['hyp']!r} | {a['hyp']!r}")

    zero = {"wer": 0.0}
    ok, why = Gate().decide(base.get("dysarthric", zero)["wer"], new.get("dysarthric", zero)["wer"],
                            base.get("typical", zero)["wer"], new.get("typical", zero)["wer"])
    results["verdict"] = {"ship": ok, "reason": why}
    print(f"\n{'SHIP' if ok else 'DONT SHIP'}: {why}")
    if ok:
        print('Use it: put {"whisper_model": "%s"} in config.local.json' % args.adapted.replace("\\", "/"))

    report = ROOT / args.adapted / "eval_report.json"
    if report.parent.exists():
        report.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"report: {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
