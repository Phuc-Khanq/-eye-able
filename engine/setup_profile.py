"""Picks what this PC can run and gets it ready: called by scripts/setup.ps1 (Install EYEable.bat).

Route: Khang
    python -m engine.setup_profile --vram 16303 --write     # choose a profile, write config.local.json if missing
    python -m engine.setup_profile --download               # download the models config.local.json asks for
    python -m engine.setup_profile --show                   # what this PC will run, and why

Profiles (GPU memory in MB as nvidia-smi reports it; the needs were measured on the demo PC, Oct 24 night:
Whisper large-v3-turbo about 2.5 GB, UI-Venus 1.5 2B about 6 GB, Qwen2.5-3B clean-up about 5 GB, peak 15.1 GB):
  full   16 GB card (RTX 4080/5080/4090/5090, 4060 Ti 16 GB): everything, as in the demo
  gpu12  12 GB card: everything but the dictation clean-up and refine (your own words are typed)
  gpu8   8 GB card: no look-and-say (UI-Venus); the smaller Qwen2.5-1.5B does the clean-up
  cpu    no NVIDIA card, or under 8 GB: head pointer, blink click, commands and dictation with Whisper small
         on the CPU (slower and less accurate), no look-and-say, no clean-up, no live preview
An existing config.local.json is never overwritten: the person's own settings win.
Nothing here runs at app time; it only writes config.local.json and fills the Hugging Face cache.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "config.local.json"

# GPU memory a profile needs (MB, from the demo PC's [gpu] lines plus the desktop's own ~1.5 GB)
NEEDS_MB = {"full": 15000, "gpu12": 11000, "gpu8": 7500}

PROFILES: dict[str, dict] = {
    "full": {},                                                   # the defaults in engine/config.py
    "gpu12": {"rewrite_dictation": False},
    "gpu8": {"use_ui_venus": False, "rewrite_model": "Qwen/Qwen2.5-1.5B-Instruct"},
    "cpu": {"whisper_device": "cpu", "whisper_model": "small", "whisper_compute_type": "int8",
            "use_ui_venus": False, "rewrite_dictation": False, "live_preview": False},
}

WHAT_YOU_GET = {
    "full": "everything: head pointer, blink click, voice commands, dictation with live preview and clean-up, "
            "look-and-say (\"click Send\")",
    "gpu12": "head pointer, blink click, voice commands, dictation with live preview, look-and-say; "
             "no clean-up (your own words are typed)",
    "gpu8": "head pointer, blink click, voice commands, dictation with live preview and clean-up (smaller model); "
            "no look-and-say",
    "cpu": "head pointer, blink click, voice commands, dictation (Whisper small on the CPU: a second or two per "
           "sentence, less accurate); no live preview, no clean-up, no look-and-say",
}


def pick(vram_mb: int | None) -> str:
    """The best profile for this much GPU memory (None or 0: no NVIDIA GPU)."""
    if not vram_mb:
        return "cpu"
    for name in ("full", "gpu12", "gpu8"):
        if vram_mb >= NEEDS_MB[name]:
            return name
    return "cpu"


def nvidia_vram_mb(run=subprocess.run) -> int | None:
    """Total memory of the first NVIDIA GPU in MB, or None (no card, no driver)."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                  capture_output=True, text=True, timeout=10).stdout
        return int(out.strip().splitlines()[0].strip())
    except Exception:
        return None


def write_config(profile: str, path: Path = CONFIG) -> bool:
    """Write config.local.json for `profile` unless one exists. Returns whether it wrote. UTF-8 without a BOM
    (config.py reads it with json.loads; a BOM, which PowerShell's Set-Content adds, would break it)."""
    if path.exists():
        return False
    path.write_text(json.dumps(PROFILES[profile]) + "\n", encoding="utf-8")
    return True


def wanted_models(settings) -> list[tuple[str, str]]:
    """[(kind, name)] the app will load with these settings: kind "whisper" or "hf"."""
    out = []
    model = str(settings.whisper_model)
    if not (REPO_ROOT / model).is_dir() and not Path(model).is_dir():   # a local folder (a fine-tune) needs nothing
        out.append(("whisper", model))
    if getattr(settings, "use_ui_venus", True):
        out.append(("hf", str(settings.ui_venus_model)))
    if getattr(settings, "rewrite_dictation", True):
        out.append(("hf", str(settings.rewrite_model)))
    return out


def download(settings, log=print) -> None:
    """Fill the Hugging Face cache now, so the first launch doesn't wait 10 minutes for downloads."""
    for kind, name in wanted_models(settings):
        log(f"Downloading {name} (once; later runs load it from disk) ...")
        if kind == "whisper":
            from faster_whisper.utils import download_model

            download_model(name)
        else:
            from huggingface_hub import snapshot_download

            snapshot_download(name)
        log(f"  {name}: ready")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Pick what this PC can run and get it ready")
    p.add_argument("--vram", type=int, default=None, help="GPU memory in MB (default: ask nvidia-smi)")
    p.add_argument("--profile", choices=["auto", *PROFILES], default="auto")
    p.add_argument("--write", action="store_true", help="write config.local.json for the profile if missing")
    p.add_argument("--download", action="store_true", help="download the models the settings ask for")
    p.add_argument("--show", action="store_true", help="print the profile and what it runs")
    args = p.parse_args(argv)

    vram = args.vram if args.vram is not None else nvidia_vram_mb()
    profile = pick(vram) if args.profile == "auto" else args.profile
    if args.show or args.write:
        gpu = f"{vram} MB NVIDIA GPU" if vram else "no NVIDIA GPU"
        print(f"This PC: {gpu} -> profile '{profile}': {WHAT_YOU_GET[profile]}")
    if args.write:
        if write_config(profile):
            print(f"Wrote config.local.json: {json.dumps(PROFILES[profile])}")
        else:
            print(f"Kept your config.local.json: {CONFIG.read_text(encoding='utf-8').strip()}")
    if args.download:
        from engine.config import load

        download(load())
    return 0


if __name__ == "__main__":
    sys.exit(main())
