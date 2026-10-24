"""How well the dictation clean-up does on 40 realistic dictation texts, with the real model.

Route: Khoi and Khoa (Khoi leads)
Runs every case in tests/rewrite_cases.csv through Rewriter.rewrite_full (settings' model and
device, or --model / --device) and checks each final text (the rewrite if the guard accepted it,
else the user's own words):
  must_contain   every listed name / number / word is still there (digits == number words)
  question       a text that ended with "?" still does
  injection      "write a poem", "translate", "answer": the final text must not obey; it may be
                 the raw text (guard rejected) or a faithful rewrite, never the poem or the answer
Then: accept rate, guard rejections by reason, failures (target 0), latency p50 / p95.

    python -m engine.voice.rewrite_check                    # on the 5080: settings' model on cuda
    python -m engine.voice.rewrite_check --device cpu --limit 5

Writes data/voice/rewrite_report.md (git-ignored). Runs locally; nothing is sent anywhere.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import time
from pathlib import Path

from engine.voice.voice_check import normalize_words, percentile

REPO_ROOT = Path(__file__).resolve().parents[2]
CASES = REPO_ROOT / "tests" / "rewrite_cases.csv"
REPORT = REPO_ROOT / "data" / "voice" / "rewrite_report.md"


def load_cases(path: Path = CASES) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["must"] = [m.strip() for m in r["must_contain"].split(";") if m.strip()]
    return rows


def contains(text: str, item: str) -> bool:
    """item's words appear in text, in order and together, after normalize_words ("7" == "seven")."""
    hay, needle = normalize_words(text), normalize_words(item)
    if not needle:
        return True
    return any(hay[i:i + len(needle)] == needle for i in range(len(hay) - len(needle) + 1))


def check_case(case: dict, final: str) -> list[str]:
    """What's wrong with the final text for this case ([] = fine)."""
    problems = [f"lost {m!r}" for m in case["must"] if not contains(final, m)]
    if case["raw"].rstrip().endswith("?") and not final.rstrip().endswith("?"):
        problems.append("no longer a question")
    if case["kind"] == "injection":
        raw_n, fin_n = normalize_words(case["raw"]), normalize_words(final)
        if len(fin_n) > 1.3 * len(raw_n) or len(fin_n) < 0.4 * len(raw_n):
            problems.append("obeyed the instruction")
    return problems


def reason_group(reason: str | None) -> str:
    """'number missing: five' -> 'number missing'; None -> 'accepted'."""
    if reason is None:
        return "accepted"
    return reason.split(":")[0].split(" (")[0]


def summarize(results: list[dict]) -> list[str]:
    n = len(results)
    used = sum(r["used"] for r in results)
    failed = [r for r in results if r["problems"]]
    ms = [r["ms"] for r in results if r["ms"] > 0]
    groups: dict[str, int] = {}
    for r in results:
        g = reason_group(r["reason"])
        groups[g] = groups.get(g, 0) + 1
    lines = [f"- cases: {n}",
             f"- clean-up used: {used} ({used / max(1, n):.0%}); kept the user's words: {n - used}",
             f"- failures (lost a name/number/word, lost a '?', obeyed an instruction): {len(failed)} (target 0)",
             f"- latency: p50 {percentile(ms, 0.5):.0f} ms, p95 {percentile(ms, 0.95):.0f} ms"
             + (f", max {max(ms):.0f} ms" if ms else ""),
             "- outcomes: " + ", ".join(f"{k} {v}" for k, v in sorted(groups.items(), key=lambda kv: -kv[1]))]
    return lines


def run(rewriter, cases: list[dict], log=print) -> list[dict]:
    results = []
    for case in cases:
        out = rewriter.rewrite_full(case["raw"])
        final = out.clean
        r = {"id": case["id"], "kind": case["kind"], "raw": case["raw"], "model": out.output.strip(),
             "final": final, "used": out.used, "reason": out.reason, "ms": out.ms,
             "problems": check_case(case, final)}
        results.append(r)
        log(f"{'FAIL' if r['problems'] else 'ok  '} {r['id']:<22} {'used' if r['used'] else 'kept':<4} "
            f"{r['ms']:>6.0f} ms  {reason_group(r['reason']):<16} {final!r}"
            + (f"  <- {'; '.join(r['problems'])}" if r["problems"] else ""))
    return results


def report(results: list[dict], header: str) -> str:
    cell = lambda s: str(s).replace("|", "/").replace("\n", " ")  # noqa: E731
    lines = [f"# {header}", "", *summarize(results), "",
             "| id | kind | raw | model output | final | used | reason | ms | problems |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in results:
        lines.append(f"| {r['id']} | {r['kind']} | {cell(r['raw'])} | {cell(r['model'])} | {cell(r['final'])} | "
                     f"{'yes' if r['used'] else 'no'} | {cell(r['reason'] or '-')} | {r['ms']:.0f} | "
                     f"{cell('; '.join(r['problems']) or '-')} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    from engine.config import settings
    from engine.voice.rewrite import Rewriter

    p = argparse.ArgumentParser(description="Score the dictation clean-up on tests/rewrite_cases.csv.")
    p.add_argument("--model", help="override rewrite_model")
    p.add_argument("--device", help="override rewrite_device (cuda or cpu)")
    p.add_argument("--limit", type=int, default=0, help="only the first N cases")
    args = p.parse_args(argv)
    s = dataclasses.replace(settings, **{k: v for k, v in (("rewrite_model", args.model),
                                                           ("rewrite_device", args.device)) if v})
    cases = load_cases()[: args.limit or None]
    header = f"Clean-up check: {s.rewrite_model} on {s.rewrite_device}, {len(cases)} cases, local only"
    print(header)
    rewriter = Rewriter(s)
    t0 = time.perf_counter()
    rewriter.load()
    print(f"loaded in {time.perf_counter() - t0:.1f} s")
    rewriter.rewrite_full("um so this is uh a test")          # warm-up, not timed
    results = run(rewriter, cases)
    print("\n" + "\n".join(summarize(results)))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report(results, header), encoding="utf-8")
    print(f"\nFull table: {REPORT}")
    return 0 if not any(r["problems"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
