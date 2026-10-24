""""Delete my voice data": removes personal recordings and vocabulary.

Route: Khoi and Khoa (Khoa leads)
Publishes: status
Subscribes: a delete request from the settings panel (topics.DELETE_VOICE_DATA, sent by the
            mode manager after the user confirms)
APIs: pathlib

What it deletes: data/voice/personal/ (the recordings made with
`python -m engine.voice.finetune.record_pack`, plus their transcripts.csv).
Live microphone audio is never saved, so there is nothing else to delete.

    python -m engine.voice.voice_data      # delete from the command line (asks first)
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from engine.core import messages, topics

REPO_ROOT = Path(__file__).resolve().parents[2]
PERSONAL_DIR = REPO_ROOT / "data" / "voice" / "personal"


def personal_dirs(data_dir: Path | None = None) -> list[Path]:
    return [Path(data_dir) / "personal"] if data_dir else [PERSONAL_DIR]


def delete_all(data_dir: Path | None = None) -> list[Path]:
    """Delete every personal voice folder. Returns the folders that were removed."""
    removed = []
    for folder in personal_dirs(data_dir):
        if folder.exists():
            shutil.rmtree(folder)
            removed.append(folder)
    return removed


class VoiceData:
    """Deletes the personal voice data when the mode manager asks (after a confirm)."""

    def __init__(self, bus, data_dir: Path | None = None) -> None:
        self.bus = bus
        self.data_dir = data_dir
        bus.subscribe(topics.DELETE_VOICE_DATA, self.on_delete)

    def on_delete(self, _payload) -> None:
        try:
            removed = delete_all(self.data_dir)
        except OSError as exc:
            self._status("error", f"Could not delete your voice data: {exc}")
            return
        self._status("info", "Deleted your voice recordings" if removed
                     else "There were no voice recordings to delete")

    def _status(self, level: str, text: str) -> None:
        self.bus.publish(topics.TO_UI, messages.make(messages.STATUS, level=level, text=text))


if __name__ == "__main__":
    folders = [f for f in personal_dirs() if f.exists()]
    if not folders:
        print("No personal voice data found.")
        sys.exit(0)
    for f in folders:
        n = sum(1 for p in f.rglob("*") if p.is_file())
        print(f"{f}: {n} files")
    try:
        answer = input("Delete all of it? [y/N] ").strip().lower()
    except EOFError:
        answer = ""
    if answer in ("y", "yes"):
        delete_all()
        print("Deleted.")
    else:
        print("Nothing deleted.")
