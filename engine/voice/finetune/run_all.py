"""The whole pipeline in one command, for an unattended Kaggle run. Never asks for input.

    python -m engine.voice.finetune.run_all --platform kaggle --hours 10 --out /kaggle/working
    python -m engine.voice.finetune.run_all --platform kaggle --hours 10 --out /kaggle/working \\
        --resume-from /kaggle/input/<previous-version-output>
    python -m engine.voice.finetune.run_all --tiny --out <some temp folder>      # laptop smoke test
    python -m engine.voice.finetune.run_all --platform local --hours 3 --out downloads\\personal \\
        --search engine\\voice\\finetune\\search_personal.json --export-dir models\\whisper-elder-personal-ct2
                                                      # the demo PC: TORGO + LibriSpeech + your own voice pack

Steps:
  1. get_data --yes (into /kaggle/temp on Kaggle), 2. prepare
  3. baseline: the stock model on TEST (skipped if --resume-from has results.json with one)
  4. train each config in search.json, one after another, choosing on DEV only, until the
     list or the time budget runs out (1 hour is kept for export and the final test)
  5. export the best run, evaluate it on TEST once
  6. write to --out ONLY: RESULTS.md, results.json, best_adapter/ (to resume or re-export),
     whisper-elder-ct2.zip, run_all.log, plus resume_adapter/ if a run was cut short.
     No audio, no transcripts, no full checkpoints.

If faster-whisper can't use the GPU, evaluation falls back to transformers with the same
greedy decoding, and RESULTS.md says so. If a step fails, the log has the error and
RESULTS.md still lists everything that finished.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from engine.voice.finetune import common

KAGGLE_WORK = Path("/kaggle/temp")
RESERVE_HOURS = 1.0           # kept for export + final test evaluation
MIN_RUN_MINUTES = 25          # don't start a config with less time than this

# What "better" means. Relative = fraction of the stock model's WER.
TARGETS = {
    "dys_unseen": {"kind": "relative_drop", "value": 0.20,
                   "text": "at least 20% lower WER than the stock model (PRIMARY)"},
    "dys_seen": {"kind": "relative_drop", "value": 0.20, "text": "at least 20% lower WER"},
    "dys_words": {"kind": "relative_drop", "value": 0.10, "text": "at least 10% lower WER"},
    "librispeech": {"kind": "max_rise", "value": 0.5,
                    "text": "no more than 0.5 WER points worse (doesn't forget typical speech)"},
    "personal": {"kind": "relative_drop", "value": 0.10, "text": "at least 10% lower WER"},
    "latency_5s_ms": {"kind": "max_rise_rel", "value": 0.15,
                      "text": "no more than 15% slower than the stock model"},
}


class Pipeline:
    def __init__(self, args) -> None:
        self.args = args
        self.out = Path(args.out).resolve()
        self.tiny = args.tiny
        if args.platform == "kaggle":
            work = Path(args.work_dir or KAGGLE_WORK)
            self.data_dir = Path(args.data_dir or work / "data" / "voice")
            self.checkpoints = work / "checkpoints"
            self.export_dir = work / "whisper-elder-ct2"
        else:
            self.data_dir = Path(args.data_dir or common.DEFAULT_DATA_DIR)
            self.checkpoints = common.DEFAULT_CHECKPOINTS / ("run_all-tiny" if self.tiny else "run_all")
            from engine.voice.finetune.export import TINY_EXPORT_DIR

            self.export_dir = TINY_EXPORT_DIR if self.tiny else common.DEFAULT_EXPORT_DIR
        if getattr(args, "export_dir", None):          # e.g. models/whisper-elder-personal-ct2: keep the old model
            self.export_dir = Path(args.export_dir)
        for inner in (self.data_dir, self.checkpoints):
            if self.out == inner.resolve() or self.out in inner.resolve().parents:
                raise SystemExit(f"--out {self.out} must not contain {inner} (no data or checkpoints in the output)")
        self.out.mkdir(parents=True, exist_ok=True)
        self.checkpoints.mkdir(parents=True, exist_ok=True)
        self.log = common.Log(self.out / "run_all.log")
        self.t0 = time.monotonic()
        self.deadline = self.t0 + args.hours * 3600
        self.reserve = 0.0 if self.tiny else RESERVE_HOURS * 3600
        self.base_hf = common.TINY_MODEL if self.tiny else common.BASE_MODEL
        self.base_ct2 = common.TINY_CT2 if self.tiny else common.BASE_CT2
        self.ct2_ok: bool | None = None
        self.results: dict = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "platform": args.platform,
                              "tiny": self.tiny, "hours": args.hours, "runs": [], "errors": [],
                              "targets": TARGETS, "baseline": None, "best": None, "best_run": None}

    # ------------------------------------------------------------- helpers
    def minutes_left(self) -> float:
        return (self.deadline - time.monotonic()) / 60

    def step(self, name: str, fn):
        self.log(f"=== {name}")
        try:
            return fn()
        except BaseException as exc:      # SystemExit too: an unattended run must keep going
            if isinstance(exc, KeyboardInterrupt):
                raise
            self.log(f"FAILED: {name}: {exc}")
            self.log(traceback.format_exc())
            self.results["errors"].append(f"{name}: {exc}")
            self.save()
            return None

    def evaluate(self, split: str, ct2_model: str, adapter: Path | None) -> dict:
        """faster-whisper like the app; transformers if faster-whisper can't run here."""
        from engine.voice.finetune import evaluate

        argv = ["--data-dir", str(self.data_dir), "--split", split, "--model", ct2_model,
                "--checkpoints", str(self.checkpoints)]
        if self.tiny:
            argv += ["--tiny", "--device", "cpu"]
        args = evaluate.build_parser().parse_args(argv)
        if self.ct2_ok is not False:
            try:
                r = evaluate.run(args, self.log)
                self.ct2_ok = True
                return r
            except Exception as exc:
                if self.ct2_ok is True:
                    raise
                self.ct2_ok = False
                self.log(f"faster-whisper failed here ({exc}); using the transformers backend")
                self.results["errors"].append(f"faster-whisper unavailable, used transformers: {exc}")
        args.backend = "hf"
        args.hf_base = self.base_hf
        args.adapter = adapter
        return evaluate.run(args, self.log)

    def save(self) -> None:
        self.results["minutes"] = round((time.monotonic() - self.t0) / 60, 1)
        common.save_json(self.out / "results.json", self.results)
        (self.out / "RESULTS.md").write_text(results_markdown(self.results), encoding="utf-8")

    # --------------------------------------------------------------- steps
    def get_data(self) -> None:
        from engine.voice.finetune import get_data

        code = get_data.main(["--yes", "--data-dir", str(self.data_dir)] + (["--tiny"] if self.tiny else []))
        if code:
            raise RuntimeError(f"get_data exited with {code}")

    def prepare(self) -> None:
        from engine.voice.finetune import prepare

        code = prepare.main(["--data-dir", str(self.data_dir)] + (["--tiny"] if self.tiny else []))
        if code:
            raise RuntimeError(f"prepare exited with {code}")

    def load_previous(self) -> None:
        src = self.args.resume_from
        if not src:
            return
        prev_file = next(iter(sorted(Path(src).rglob("results.json"))), None)
        if prev_file is None:
            self.log(f"--resume-from {src}: no results.json found, starting fresh")
            return
        prev = json.loads(prev_file.read_text(encoding="utf-8"))
        root = prev_file.parent
        self.results["baseline"] = prev.get("baseline")
        self.results["resumed_from"] = str(root)
        for run in prev.get("runs", []):
            run = dict(run, from_previous=True)
            if run.get("name") == prev.get("best_run") and (root / "best_adapter").is_dir():
                dest = self.checkpoints / run["name"] / "best"
                shutil.copytree(root / "best_adapter", dest, dirs_exist_ok=True)
                run["best_adapter"] = str(dest)
            self.results["runs"].append(run)
        if (root / "resume_adapter").is_dir():
            self.resume_adapter = root / "resume_adapter"
        self.log(f"Resumed from {root}: baseline {'found' if self.results['baseline'] else 'missing'}, "
                 f"{len(self.results['runs'])} earlier runs")

    def baseline(self) -> None:
        if self.results["baseline"]:
            self.log("Baseline already measured in the previous output: skipped")
            return
        self.results["baseline"] = self.evaluate("test", self.base_ct2, None)
        self.save()

    def configs(self) -> list[dict]:
        if self.tiny:
            return [{"name": "tiny"}]
        return common.load_search(getattr(self.args, "search", None))

    def train_all(self) -> None:
        from engine.voice.finetune.train import TrainConfig, tiny_config, train

        done = {r["name"] for r in self.results["runs"] if r.get("stop_reason") != "time budget"}
        cut_short = {r["name"] for r in self.results["runs"] if r.get("stop_reason") == "time budget"}
        todo = [c for c in self.configs() if c["name"] not in done]
        for i, c in enumerate(todo):
            left = self.minutes_left() - self.reserve / 60
            budget = left / (len(todo) - i)
            if not self.tiny and budget < MIN_RUN_MINUTES:
                self.log(f"Only {left:.0f} min left: not starting {c['name']}")
                break
            cfg = TrainConfig.from_dict(c)
            if self.tiny:
                cfg = tiny_config(cfg)
                cfg.name = c["name"]
            cfg.max_minutes = 0.0 if self.tiny else budget
            resume = None
            if c["name"] in cut_short and getattr(self, "resume_adapter", None):
                resume = self.resume_adapter
                self.results["runs"] = [r for r in self.results["runs"] if r["name"] != c["name"]]
                if (resume / "best_of_run").is_dir():     # its best adapter so far
                    shutil.copytree(resume / "best_of_run", self.checkpoints / c["name"] / "best",
                                    dirs_exist_ok=True)
            self.log(f"--- config {c['name']} ({i + 1}/{len(todo)}), budget {budget:.0f} min")

            def one():
                return train(cfg, self.data_dir, self.checkpoints, resume, self.log)

            r = self.step(f"train {c['name']}", one)
            if r:
                self.results["runs"].append(r)
                self.save()

    def pick_best(self) -> dict | None:
        runs = [r for r in self.results["runs"] if r.get("best_dev") and Path(r.get("best_adapter", "")).is_dir()]
        if not runs:
            return None
        best = min(runs, key=lambda r: r["best_dev"]["score"])
        self.results["best_run"] = best["name"]
        if best["best_dev"]["step"] == 0:
            self.log("Note: no run beat the stock model on dev; exporting it anyway for the record")
        return best

    def export_and_test(self, best: dict) -> None:
        from engine.voice.finetune.export import export

        adapter = Path(best["best_adapter"])
        export(adapter, self.export_dir, self.checkpoints, "float16", self.tiny, self.log)
        self.results["best"] = self.evaluate("test", str(self.export_dir), adapter)
        self.save()

    def write_outputs(self, best: dict | None) -> None:
        if best:
            dest = self.out / "best_adapter"
            shutil.rmtree(dest, ignore_errors=True)
            shutil.copytree(best["best_adapter"], dest)
        cut = [r for r in self.results["runs"] if r.get("stop_reason") == "time budget"
               and not r.get("from_previous")]
        if cut:
            last = self.checkpoints / cut[-1]["name"] / "last"
            if last.is_dir():
                dest = self.out / "resume_adapter"
                shutil.rmtree(dest, ignore_errors=True)
                shutil.copytree(last, dest)
                best_of_run = self.checkpoints / cut[-1]["name"] / "best"
                if best_of_run.is_dir():
                    shutil.copytree(best_of_run, dest / "best_of_run")
        if self.export_dir.is_dir() and (self.export_dir / "model.bin").exists():
            zip_base = self.out / "whisper-elder-ct2"
            shutil.make_archive(str(zip_base), "zip", root_dir=self.export_dir.parent,
                                base_dir=self.export_dir.name)
            self.log(f"Wrote {zip_base}.zip ({(zip_base.with_suffix('.zip')).stat().st_size / 1e6:.0f} MB)")

    def run(self) -> int:
        self.log(f"run_all: platform {self.args.platform}, {self.args.hours} h budget, out {self.out}")
        self.log(f"data {self.data_dir}, checkpoints {self.checkpoints}")
        self.results["environment"] = environment()
        self.save()
        self.load_previous()
        if self.step("get_data", self.get_data) is None and self.results["errors"]:
            self.save()
            return 1
        if self.step("prepare", self.prepare) is None and self.results["errors"]:
            self.save()
            return 1
        self.step("baseline (stock model on test)", self.baseline)
        self.step("training", self.train_all)
        best = self.pick_best()
        if best:
            self.log(f"Best on dev: {best['name']} (score {best['best_dev']['score']} at step "
                     f"{best['best_dev']['step']})")
            self.step("export + final test", lambda: self.export_and_test(best))
        else:
            self.log("No finished run to export")
        self.step("write outputs", lambda: self.write_outputs(best))
        self.results["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.save()
        self.log(f"Done in {(time.monotonic() - self.t0) / 60:.0f} min. See {self.out / 'RESULTS.md'}")
        return 1 if self.results["errors"] and not self.results.get("best") else 0


def environment() -> dict:
    env = {"python": platform.python_version(), "os": platform.platform()}
    for name in ("torch", "transformers", "peft", "ctranslate2", "faster_whisper", "jiwer"):
        try:
            mod = __import__(name)
            env[name] = getattr(mod, "__version__", "?")
        except Exception:
            env[name] = "missing"
    env["gpu"] = common.gpu_name()
    return env


# ----------------------------------------------------------------- RESULTS.md
def _wer(r: dict | None, group: str):
    g = (r or {}).get("groups", {}).get(group)
    return g["wer"] if g and g.get("n") else None


def _check(group: str, base, new) -> str:
    t = TARGETS.get(group)
    if t is None or base is None or new is None:
        return ""
    if t["kind"] == "relative_drop":
        ok = new <= base * (1 - t["value"])
    elif t["kind"] == "max_rise":
        ok = new <= base + t["value"]
    else:
        ok = new <= base * (1 + t["value"])
    return "met" if ok else "not met"


def results_markdown(res: dict) -> str:
    base, best = res.get("baseline"), res.get("best")
    lines = ["# Whisper fine-tune results", ""]
    lines.append(f"Started {res.get('started')}, platform {res.get('platform')}, "
                 f"{res.get('minutes', 0):.0f} min" + (" (TINY smoke test: numbers mean nothing)" if res.get("tiny") else ""))
    env = res.get("environment") or {}
    if env:
        lines.append(f"GPU {env.get('gpu')}, Python {env.get('python')}, torch {env.get('torch')}, "
                     f"transformers {env.get('transformers')}, peft {env.get('peft')}, "
                     f"ctranslate2 {env.get('ctranslate2')}")
    if base:
        lines.append(f"Evaluation backend: {base.get('backend')}"
                     + ("" if base.get("backend") == "faster-whisper" else
                        " (faster-whisper could not use the GPU; same greedy decoding)"))
    lines += ["", "## Test set: stock model vs best fine-tuned model", "",
              "WER % (lower is better). Test speakers were never trained on or used to choose settings.", ""]
    names = dict(common.GROUPS, dys_all="Dysarthric, all")
    rows = []
    for g in list(common.GROUPS) + ["dys_all"]:
        b, n = _wer(base, g), _wer(best, g)
        if b is None and n is None:
            continue
        clips = ((base or best or {}).get("groups", {}).get(g) or {}).get("n")
        target = TARGETS.get(g, {}).get("text", "")
        rows.append([names[g], clips, b, n, target, _check(g, b, n)])
    for spk in sorted(set((base or {}).get("speakers", {})) | set((best or {}).get("speakers", {}))):
        b = ((base or {}).get("speakers", {}).get(spk) or {}).get("wer")
        n = ((best or {}).get("speakers", {}).get(spk) or {}).get("wer")
        clips = (((base or best or {}).get("speakers", {})).get(spk) or {}).get("n")
        rows.append([f"  TORGO test speaker {spk}", clips, b, n, "", ""])
    lb, ln = (base or {}).get("latency_5s_ms"), (best or {}).get("latency_5s_ms")
    if lb is not None or ln is not None:
        rows.append(["Warm latency, 5 s clip (ms)", "", lb, ln, TARGETS["latency_5s_ms"]["text"],
                     _check("latency_5s_ms", lb, ln)])
    if rows:
        lines.append(common.fmt_table(rows, ["group", "clips", "stock", "fine-tuned", "target", "result"]))
    else:
        lines.append("(no test results yet)")
    lines += ["", f"Best run (chosen on DEV only): {res.get('best_run') or '-'}", "",
              "## Every training run (dev set)", ""]
    run_rows = []
    for r in res.get("runs", []):
        c = r.get("config", {})
        bd, b0 = r.get("best_dev") or {}, r.get("baseline_dev") or {}
        run_rows.append([r.get("name"), c.get("lora_r"), c.get("lora_alpha"), c.get("lr"),
                         c.get("typical_mix"), f"{r.get('steps')}/{r.get('max_steps')}",
                         r.get("epochs"), r.get("minutes"),
                         (b0.get("dys") or {}).get("wer"), (bd.get("dys") or {}).get("wer"),
                         (b0.get("typical") or {}).get("wer"), (bd.get("typical") or {}).get("wer"),
                         bd.get("score"), bd.get("step"), r.get("stop_reason")])
    if run_rows:
        lines.append(common.fmt_table(run_rows, [
            "run", "r", "alpha", "lr", "mix", "steps", "epochs", "min", "dev dys stock",
            "dev dys best", "dev typ stock", "dev typ best", "score", "best step", "stopped"]))
        lines += ["", "score = dysarthric dev WER (averaged with unseen-text dev WER when there are "
                  "30+ unseen sentences) + a penalty if typical-speech dev WER got more than 1 point worse."]
    else:
        lines.append("(no finished runs)")
    if res.get("errors"):
        lines += ["", "## Problems", ""] + [f"- {e}" for e in res["errors"]]
    lines += ["", "## Files", "",
              "- `whisper-elder-ct2.zip`: the fine-tuned model for the app. Unzip into `models/` and set "
              "`{\"whisper_model\": \"models/whisper-elder-ct2\"}` in config.local.json.",
              "- `best_adapter/`: the LoRA adapter of the best run (to resume or re-export).",
              "- `resume_adapter/` (only if a run was cut short): continue it with --resume-from.",
              "", "Trained on TORGO: academic, non-profit use only. Never publish or upload the model "
              "or adapter (see engine/voice/finetune/DATA.md).", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--platform", default="local", choices=["local", "kaggle"],
                        help="kaggle: data and checkpoints in /kaggle/temp, never in --out")
    parser.add_argument("--hours", type=float, default=10.0, help="total time budget")
    parser.add_argument("--out", type=Path, required=True, help="where the small results go (/kaggle/working)")
    parser.add_argument("--resume-from", type=Path, default=None,
                        help="a previous run's output folder (e.g. /kaggle/input/<name>)")
    parser.add_argument("--data-dir", type=Path, default=None, help="override the data folder")
    parser.add_argument("--work-dir", type=Path, default=None, help="kaggle: scratch folder (default /kaggle/temp)")
    parser.add_argument("--tiny", action="store_true", help="smoke test: whisper tiny, 40 clips, CPU")
    parser.add_argument("--search", type=Path, default=None,
                        help="train the configs in this file instead of search.json (search_personal.json)")
    parser.add_argument("--export-dir", type=Path, default=None,
                        help="where the converted model goes (default models/whisper-elder-ct2: it would REPLACE "
                             "the current one)")
    args = parser.parse_args(argv)
    return Pipeline(args).run()


if __name__ == "__main__":
    sys.exit(main())
