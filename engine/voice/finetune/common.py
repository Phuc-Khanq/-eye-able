"""Shared helpers for the fine-tuning scripts: folders, manifests, text, audio, logging.

Works on Windows (Python 3.11) and on Kaggle's Linux (Python 3.13). Heavy libraries
(torch, transformers, peft, faster_whisper) are only imported inside functions.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")   # Windows cache note, harmless
HERE = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "voice"
DEFAULT_CHECKPOINTS = REPO_ROOT / "checkpoints"
DEFAULT_EXPORT_DIR = REPO_ROOT / "models" / "whisper-elder-ct2"
SPLIT_FILE = HERE / "split.json"
SEARCH_FILE = HERE / "search.json"

BASE_MODEL = "openai/whisper-large-v3-turbo"   # Hugging Face weights we fine-tune
BASE_CT2 = "large-v3-turbo"                     # the same model for faster-whisper
TINY_MODEL = "openai/whisper-tiny"              # --tiny smoke runs
TINY_CT2 = "tiny"

SAMPLE_RATE = 16_000
MIN_CLIP_S = 0.3
MAX_CLIP_S = 30.0          # Whisper hears at most 30 s

MANIFEST_FIELDS = ["path", "text", "speaker", "group", "split", "source", "seconds"]

# Groups in the manifests (the "group" column) and the row names in every results table
GROUPS = {
    "dys_unseen": "Dysarthric, unseen-text sentences (PRIMARY)",
    "dys_seen": "Dysarthric, seen-text sentences",
    "dys_words": "Dysarthric, single words (1-2 words)",
    "librispeech": "LibriSpeech (typical speech)",
    "personal": "Personal voice pack",
}
DYS_GROUPS = ("dys_unseen", "dys_seen", "dys_words")


# ---------------------------------------------------------------------- paths
def data_dir_arg(parser) -> None:
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help=f"where the datasets live (default: <repo>/data/voice)")


def manifests_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "manifests"


def manifest_path(data_dir: Path, tiny: bool = False) -> Path:
    return manifests_dir(data_dir) / ("tiny.csv" if tiny else "all.csv")


def is_kaggle() -> bool:
    return Path("/kaggle").is_dir() and sys.platform.startswith("linux")


# ------------------------------------------------------------------ manifests
@dataclass
class Clip:
    path: str        # relative to data_dir, with forward slashes
    text: str        # Whisper-style text (capital letter, full stop)
    speaker: str
    group: str
    split: str       # train | dev | test
    source: str      # torgo | librispeech | personal
    seconds: float

    def audio_path(self, data_dir: Path) -> Path:
        return Path(data_dir) / self.path


def write_manifest(path: Path, clips: list[Clip]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for c in clips:
            w.writerow({k: getattr(c, k) for k in MANIFEST_FIELDS})


def read_manifest(path: Path) -> list[Clip]:
    if not Path(path).exists():
        raise SystemExit(f"No manifest at {path}. Run: python -m engine.voice.finetune.prepare")
    with open(path, encoding="utf-8", newline="") as f:
        return [Clip(r["path"], r["text"], r["speaker"], r["group"], r["split"], r["source"],
                     float(r["seconds"])) for r in csv.DictReader(f)]


def load_split() -> dict:
    return json.loads(SPLIT_FILE.read_text(encoding="utf-8"))


def load_search(path=None) -> list[dict]:
    """The configs to train, from search.json or another file with the same shape (run_all --search)."""
    return json.loads(Path(path or SEARCH_FILE).read_text(encoding="utf-8"))["configs"]


# ------------------------------------------------------------------------ text
def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text)


def whisper_style(text: str) -> str:
    """Write a transcript the way Whisper does, so training keeps capitals and full stops.

    LibriSpeech is ALL CAPS without punctuation and TORGO is mostly lowercase; training on
    that would teach the model to drop the formatting dictation needs. WER is computed on
    normalized text, so this never changes the scores.
    """
    text = " ".join(text.replace("’", "'").split()).strip(" \"")
    if not text:
        return ""
    if text.isupper() or text.islower():
        text = text.lower()
        text = re.sub(r"\bi\b", "I", text)
        text = re.sub(r"\bi'(m|ll|ve|d)\b", lambda m: "I'" + m.group(1), text)
    text = text[0].upper() + text[1:]
    if text[-1] not in ".?!":
        text += "."
    return text


_normalizer = None
# British -> American spellings ("counselled" -> "counseled") used by Whisper's normalizer.
# A copy of normalizer.json from the openai/whisper-* models on Hugging Face (MIT license),
# kept here so every evaluation scores with exactly the same rules, offline. Without it,
# a correct "counseled" would count as an error, and stock vs fine-tuned would not compare.
SPELLING_FILE = HERE / "english_spelling.json"
SPELLING_ENTRIES = 1740


def normalize(text: str) -> str:
    """Whisper's EnglishTextNormalizer (the one used in the Whisper paper for WER)."""
    global _normalizer
    if _normalizer is None:
        from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

        spelling = json.loads(SPELLING_FILE.read_text(encoding="utf-8"))
        if len(spelling) != SPELLING_ENTRIES:
            raise RuntimeError(f"{SPELLING_FILE} has {len(spelling)} entries, expected "
                               f"{SPELLING_ENTRIES}: WER would not be comparable")
        _normalizer = EnglishTextNormalizer(spelling)
    return _normalizer(text)


def wer_table(rows: list[tuple[str, str]]) -> dict:
    """rows = [(reference, hypothesis)] -> {"wer", "n", "words", errors...}. Corpus-level WER."""
    import jiwer

    refs, hyps = [], []
    for ref, hyp in rows:
        r = normalize(ref)
        if not r.strip():
            continue
        refs.append(r)
        hyps.append(normalize(hyp))
    if not refs:
        return {"n": 0, "wer": None, "words": 0}
    out = jiwer.process_words(refs, hyps)
    n_words = out.hits + out.substitutions + out.deletions
    return {"n": len(refs), "words": n_words, "wer": round(100.0 * out.wer, 2),
            "sub": out.substitutions, "del": out.deletions, "ins": out.insertions}


# ----------------------------------------------------------------------- audio
def load_audio(path: Path):
    """Any WAV/FLAC -> 16 kHz mono float32 (same resampling as the live mic path)."""
    import soundfile as sf

    from engine.voice.mic import to_16k

    audio, rate = sf.read(str(path), dtype="float32", always_2d=False)
    return to_16k(audio, rate)


def audio_seconds(path: Path) -> float:
    import soundfile as sf

    info = sf.info(str(path))
    return info.frames / float(info.samplerate)


# ------------------------------------------------------------------------ CUDA
def preload_cuda12_libs() -> None:
    """faster-whisper's CTranslate2 needs CUDA 12 libraries (cuBLAS 12, cuDNN 9).

    Kaggle's PyTorch is built for CUDA 13, so requirements-train.txt installs the CUDA 12
    pip packages, and here we load them before CTranslate2 looks for them. No-op on
    Windows and when the packages aren't installed.
    """
    if not sys.platform.startswith("linux"):
        return
    import ctypes
    import site

    roots = [Path(p) for p in site.getsitepackages() + [site.getusersitepackages()]]
    names = ["cublas/lib/libcublasLt.so.12", "cublas/lib/libcublas.so.12",
             "cudnn/lib/libcudnn.so.9"]
    for name in names:
        for root in roots:
            lib = root / "nvidia" / name
            if lib.exists():
                try:
                    ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
                break
    # cuDNN loads its sub-libraries itself: let it find them
    for root in roots:
        cudnn = root / "nvidia" / "cudnn" / "lib"
        if cudnn.is_dir():
            os.environ["LD_LIBRARY_PATH"] = f"{cudnn}:{os.environ.get('LD_LIBRARY_PATH', '')}"
            for lib in sorted(cudnn.glob("libcudnn_*.so.9")):
                try:
                    ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
            break


def gpu_name() -> str:
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# --------------------------------------------------------------------- logging
class Log:
    """Print and also append to a log file (never audio, never transcripts)."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.t0 = time.monotonic()
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, *parts) -> None:
        mins = (time.monotonic() - self.t0) / 60
        line = f"[{mins:6.1f} min] " + " ".join(str(p) for p in parts)
        print(line, flush=True)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def fmt_table(rows: list[list], header: list[str]) -> str:
    """Plain Markdown table that also reads well in a terminal."""
    cells = [header] + [["" if c is None else str(c) for c in r] for r in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(header))]
    line = lambda r: "| " + " | ".join(c.ljust(w) for c, w in zip(r, widths)) + " |"
    return "\n".join([line(cells[0]), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
                     + [line(r) for r in cells[1:]])
