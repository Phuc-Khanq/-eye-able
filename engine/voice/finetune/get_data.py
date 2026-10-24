"""Download the datasets into data/voice/ (meant for Kaggle; --tiny is fine on a laptop).

    python -m engine.voice.finetune.get_data            # TORGO + LibriSpeech dev/test (about 4.3 GB)
    python -m engine.voice.finetune.get_data --tiny     # only LibriSpeech test-clean (347 MB)
    python -m engine.voice.finetune.get_data --yes --data-dir /kaggle/temp/data/voice

What it keeps:
  TORGO      dysarthric speakers F01 F03 F04 M01-M05 only: the wav_arrayMic WAVs and the
             prompts/ text (wav_headMic only where an utterance has no array-mic file).
             The archives are deleted after extraction.
  LibriSpeech dev-clean and test-clean, FLAC as is, plus transcripts.

Shows the download sizes and free disk space, then asks before starting (--yes skips the
question). Safe to rerun: finished parts are skipped, broken downloads resume.
See DATA.md for the licenses (TORGO is academic, non-profit use only).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import time
import urllib.request
from pathlib import Path, PurePosixPath

from engine.voice.finetune import common

TORGO_SPEAKERS = {"F01", "F03", "F04", "M01", "M02", "M03", "M04", "M05"}
TORGO_URL = "https://www.cs.toronto.edu/~complingweb/data/TORGO/{}.tar.bz2"
LIBRI_MIRRORS = ("https://www.openslr.org/resources/12/{}.tar.gz",
                 "https://openslr.elda.org/resources/12/{}.tar.gz")

# name -> (urls, archive file, roughly how much it takes once extracted)
PARTS = {
    "torgo-F": ([TORGO_URL.format("F")], "F.tar.bz2", 0.7e9),
    "torgo-M": ([TORGO_URL.format("M")], "M.tar.bz2", 1.4e9),
    "librispeech-dev-clean": ([m.format("dev-clean") for m in LIBRI_MIRRORS], "dev-clean.tar.gz", 0.35e9),
    "librispeech-test-clean": ([m.format("test-clean") for m in LIBRI_MIRRORS], "test-clean.tar.gz", 0.36e9),
}
TINY_PARTS = ["librispeech-test-clean"]
CHUNK = 1 << 20
USER_AGENT = "diseyeble-get-data/1.0"


def gb(n: float) -> str:
    return f"{n / 1e9:.2f} GB"


def remote_size(urls: list[str]) -> int | None:
    for url in urls:
        try:
            req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=30) as r:
                return int(r.headers.get("Content-Length", 0)) or None
        except Exception:
            continue
    return None


def download(urls: list[str], dest: Path, log) -> None:
    """Download with resume (HTTP Range) and a few retries, trying each mirror."""
    part = dest.with_suffix(dest.suffix + ".part")
    last_error = None
    for url in urls:
        for attempt in range(5):
            have = part.stat().st_size if part.exists() else 0
            headers = {"User-Agent": USER_AGENT}
            if have:
                headers["Range"] = f"bytes={have}-"
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=60) as r:
                    if have and r.status != 206:      # server ignored Range: start over
                        have = 0
                    total = have + int(r.headers.get("Content-Length", 0))
                    mode = "ab" if have else "wb"
                    done, next_report, t0 = have, have, time.monotonic()
                    with open(part, mode) as f:
                        while True:
                            buf = r.read(CHUNK)
                            if not buf:
                                break
                            f.write(buf)
                            done += len(buf)
                            if total and done >= next_report:
                                speed = (done - have) / max(1e-3, time.monotonic() - t0) / 1e6
                                log(f"  {dest.name}: {100 * done / total:5.1f}% of {gb(total)} "
                                    f"({speed:.1f} MB/s)")
                                next_report = done + total // 10
                if total and part.stat().st_size < total:
                    raise IOError("connection closed early")
                part.replace(dest)
                return
            except Exception as exc:
                last_error = exc
                log(f"  {dest.name}: {exc}; retrying ({attempt + 1}/5)")
                time.sleep(min(60, 5 * (attempt + 1)))
    raise SystemExit(f"Could not download {dest.name}: {last_error}")


def _safe_target(root: Path, member_name: str) -> Path | None:
    rel = PurePosixPath(member_name)
    if rel.is_absolute() or ".." in rel.parts:
        return None
    return root.joinpath(*rel.parts)


def extract_torgo(archive: Path, out_root: Path, log) -> int:
    """Keep only <spk>/<session>/wav_arrayMic/*.wav, wav_headMic/*.wav and prompts/*.txt."""
    kept = 0
    with tarfile.open(archive, mode="r|bz2") as tar:          # streaming: one pass
        for m in tar:
            if not m.isfile():
                continue
            parts = PurePosixPath(m.name).parts
            if len(parts) < 4:
                continue
            # archive layout: F/F01/Session1/wav_arrayMic/0001.wav
            spk = next((p for p in parts if p in TORGO_SPEAKERS), None)
            folder, name = parts[-2], parts[-1]
            wanted = ((folder in ("wav_arrayMic", "wav_headMic") and name.lower().endswith(".wav"))
                      or (folder == "prompts" and name.lower().endswith(".txt")))
            if spk is None or not wanted:
                continue
            i = parts.index(spk)
            target = _safe_target(out_root, "/".join(parts[i:]))
            if target is None:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(m)
            if src is None:
                continue
            with open(target, "wb") as f:
                shutil.copyfileobj(src, f, CHUNK)
            kept += 1
    # head mic only where the array mic is missing
    removed = 0
    for head in out_root.glob("*/*/wav_headMic/*.wav"):
        array = head.parent.parent / "wav_arrayMic" / head.name
        if array.exists():
            head.unlink()
            removed += 1
    for d in out_root.glob("*/*/wav_headMic"):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
    log(f"  kept {kept - removed} files ({removed} head-mic duplicates removed)")
    return kept - removed


def extract_librispeech(archive: Path, out_root: Path, log) -> int:
    """Keep FLAC files, *.trans.txt and SPEAKERS.TXT under data/voice/librispeech/."""
    kept = 0
    with tarfile.open(archive, mode="r|gz") as tar:
        for m in tar:
            if not m.isfile():
                continue
            name = m.name
            if not (name.endswith(".flac") or name.endswith(".trans.txt")
                    or name.endswith("SPEAKERS.TXT")):
                continue
            rel = PurePosixPath(name)
            if rel.parts and rel.parts[0] == "LibriSpeech":
                rel = PurePosixPath(*rel.parts[1:])
            target = _safe_target(out_root, str(rel))
            if target is None:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(m)
            if src is None:
                continue
            with open(target, "wb") as f:
                shutil.copyfileobj(src, f, CHUNK)
            kept += 1
    log(f"  kept {kept} files")
    return kept


def free_bytes(path: Path) -> int:
    p = Path(path)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    common.data_dir_arg(parser)
    parser.add_argument("--tiny", action="store_true", help="only LibriSpeech test-clean (smoke test)")
    parser.add_argument("--yes", action="store_true", help="don't ask, just download")
    parser.add_argument("--keep-archives", action="store_true", help="don't delete the .tar files")
    args = parser.parse_args(argv)
    data_dir = Path(args.data_dir)
    log = common.Log()

    names = TINY_PARTS if args.tiny else list(PARTS)
    todo = [n for n in names if not (data_dir / f".done-{n}").exists()]
    if not todo:
        log(f"Everything is already in {data_dir}")
        return 0

    log(f"Download into {data_dir}:")
    biggest, kept = 0.0, 0.0
    for n in todo:
        urls, _archive, extracted = PARTS[n]
        size = remote_size(urls)
        log(f"  {n:<24} download {gb(size) if size else '?':>8}, keeps about {gb(extracted)}")
        biggest = max(biggest, size or 3e9)
        kept += extracted
    need = biggest + kept          # one archive on disk at a time, deleted after extraction
    free = free_bytes(data_dir)
    log(f"Free disk space: {gb(free)} (needs about {gb(need)} at the peak)")
    if free < need:
        log("Not enough free disk space.")
        return 1
    if not args.yes:
        try:
            answer = input("Start the download? [y/N] ").strip().lower()
        except EOFError:
            answer = ""
        if answer not in ("y", "yes"):
            log("Nothing downloaded.")
            return 1

    downloads = data_dir / "_downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    for n in todo:
        urls, archive_name, _ = PARTS[n]
        archive = downloads / archive_name
        log(f"{n}: downloading")
        if not archive.exists():
            download(urls, archive, log)
        log(f"{n}: extracting")
        if n.startswith("torgo"):
            extract_torgo(archive, data_dir / "torgo", log)
        else:
            extract_librispeech(archive, data_dir / "librispeech", log)
        if not args.keep_archives:
            archive.unlink()
        (data_dir / f".done-{n}").write_text("ok\n", encoding="utf-8")
    if downloads.exists() and not any(downloads.iterdir()):
        downloads.rmdir()
    log("Done. Next: python -m engine.voice.finetune.prepare" + (" --tiny" if args.tiny else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
