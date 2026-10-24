"""Turn a trained LoRA adapter into a faster-whisper model the app can load, then test it.

    python -m engine.voice.finetune.export --adapter checkpoints/r32-lr1e-4-mix25/best
    python -m engine.voice.finetune.export --tiny          # smoke test (checkpoints/tiny/best)

Steps:
  1. load the base model (from the adapter's train_config.json) and merge the LoRA into it
  2. save the merged model (float16) to a temporary folder under checkpoints/
  3. ct2-transformers-converter --quantization float16, copying tokenizer.json and
     preprocessor_config.json, into models/whisper-elder-ct2/
  4. run evaluate on the new model (skip with --no-eval)

Use it in the app with config.local.json: {"whisper_model": "models/whisper-elder-ct2"}.
Never commit or upload the result: it was trained on TORGO (see DATA.md).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from engine.voice.finetune import common

COPY_FILES = ("tokenizer.json", "preprocessor_config.json")
TINY_EXPORT_DIR = common.REPO_ROOT / "models" / "whisper-elder-ct2-tiny"


def base_of(adapter: Path, fallback: str) -> str:
    cfg = Path(adapter) / "train_config.json"
    if cfg.exists():
        return json.loads(cfg.read_text(encoding="utf-8")).get("base_model", fallback)
    peft_cfg = Path(adapter) / "adapter_config.json"
    if peft_cfg.exists():
        return json.loads(peft_cfg.read_text(encoding="utf-8")).get("base_model_name_or_path") or fallback
    return fallback


def merge(adapter: Path, base: str, merged_dir: Path, log) -> None:
    import torch
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration

    from engine.voice.finetune import hf

    log(f"Merging {adapter} into {base}")
    model = WhisperForConditionalGeneration.from_pretrained(base, dtype=torch.float32)
    model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    model = model.to(torch.float16)
    if merged_dir.exists():
        shutil.rmtree(merged_dir)
    model.save_pretrained(str(merged_dir), safe_serialization=True)
    processor = hf.load_processor(base)
    processor.save_pretrained(str(merged_dir))
    for name in COPY_FILES:                        # make sure both files are there to copy
        if not (merged_dir / name).exists():
            from huggingface_hub import hf_hub_download

            shutil.copy(hf_hub_download(base, name), merged_dir / name)


def convert(merged_dir: Path, out_dir: Path, quantization: str, log) -> None:
    exe = shutil.which("ct2-transformers-converter")
    if exe is None:
        scripts = Path(sys.executable).parent
        for candidate in (scripts / "ct2-transformers-converter.exe", scripts / "ct2-transformers-converter"):
            if candidate.exists():
                exe = str(candidate)
                break
    cmd = ([exe] if exe else [sys.executable, "-m", "ctranslate2.converters.transformers"]) + [
        "--model", str(merged_dir), "--output_dir", str(out_dir), "--quantization", quantization,
        "--copy_files", *COPY_FILES, "--force"]
    log("Converting:", " ".join(Path(cmd[0]).name if i == 0 else c for i, c in enumerate(cmd)))
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)
    missing = [n for n in ("model.bin", *COPY_FILES) if not (out_dir / n).exists()]
    if missing:
        raise SystemExit(f"Conversion finished but {missing} are missing in {out_dir}")
    size = sum(f.stat().st_size for f in out_dir.iterdir()) / 1e6
    log(f"Wrote {out_dir} ({size:.0f} MB)")


def export(adapter: Path, out_dir: Path, checkpoints: Path, quantization: str = "float16",
           tiny: bool = False, log=None) -> Path:
    log = log or common.Log()
    adapter = Path(adapter)
    if not (adapter / "adapter_config.json").exists():
        raise SystemExit(f"No LoRA adapter in {adapter}")
    base = base_of(adapter, common.TINY_MODEL if tiny else common.BASE_MODEL)
    merged_dir = Path(checkpoints) / "_merged_tmp"
    try:
        merge(adapter, base, merged_dir, log)
        convert(merged_dir, Path(out_dir), quantization, log)
    finally:
        shutil.rmtree(merged_dir, ignore_errors=True)   # 1.6 GB we don't need any more
    return Path(out_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    common.data_dir_arg(parser)
    parser.add_argument("--adapter", type=Path, default=None,
                        help="LoRA adapter folder (e.g. checkpoints/<run>/best)")
    parser.add_argument("--out", type=Path, default=None,
                        help="output folder (default models/whisper-elder-ct2; tiny: models/whisper-elder-ct2-tiny)")
    parser.add_argument("--quantization", default="float16")
    parser.add_argument("--checkpoints", type=Path, default=common.DEFAULT_CHECKPOINTS)
    parser.add_argument("--no-eval", action="store_true", help="don't run evaluate afterwards")
    parser.add_argument("--split", default="test", choices=["test", "dev"],
                        help="split for the evaluation afterwards")
    parser.add_argument("--tiny", action="store_true", help="smoke test: checkpoints/tiny/best, CPU")
    args = parser.parse_args(argv)

    adapter = args.adapter or (args.checkpoints / "tiny" / "best" if args.tiny else None)
    if adapter is None:
        parser.error("--adapter is required (e.g. checkpoints/<run>/best)")
    out = args.out or (TINY_EXPORT_DIR if args.tiny else common.DEFAULT_EXPORT_DIR)
    log = common.Log()
    export(adapter, out, args.checkpoints, args.quantization, args.tiny, log)
    if not args.no_eval:
        from engine.voice.finetune import evaluate

        eval_args = evaluate.build_parser().parse_args(
            ["--data-dir", str(args.data_dir), "--model", str(out), "--split", args.split,
             "--checkpoints", str(args.checkpoints)] + (["--tiny", "--device", "cpu"] if args.tiny else []))
        evaluate.run(eval_args, log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
