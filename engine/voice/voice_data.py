""""Delete my voice data": removes personal recordings and vocabulary.

Route: Khoi and Khoa (Khoa leads)
Publishes: status
Subscribes: topics.DELETE_VOICE_DATA (the mode manager sends it only after the user confirms)
APIs: pathlib

Everything personal to one user's voice lives under VOICE_DIR (git-ignored):
    user_data/voice/commands/     personal_commands.py: the user's 3 to 5 samples per command
    user_data/voice/vocabulary.txt  names and words for Whisper's hotwords
Anything that saves voice data must save it there, so this one delete covers it.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from engine.core import messages, topics

ROOT = Path(__file__).resolve().parent.parent.parent
VOICE_DIR = ROOT / "user_data" / "voice"


def delete_all(voice_dir: Path = VOICE_DIR) -> int:
    """Delete everything inside voice_dir (the folder itself stays). Returns files removed."""
    voice_dir = Path(voice_dir).resolve()
    if voice_dir == voice_dir.parent or voice_dir in (ROOT, *ROOT.parents):
        raise ValueError(f"Refusing to delete {voice_dir}: not a voice-data folder")
    if not voice_dir.is_dir():
        return 0
    removed = 0
    for child in voice_dir.iterdir():
        if child.is_dir() and not child.is_symlink():
            removed += sum(1 for p in child.rglob("*") if p.is_file())
            shutil.rmtree(child)
        else:
            child.unlink()
            removed += 1
    return removed


class VoiceData:
    """Deletes the files and forgets the vocabulary already loaded into Whisper."""

    def __init__(self, bus, dictation=None, voice_dir: Path = VOICE_DIR) -> None:
        self.bus = bus
        self.dictation = dictation
        self.voice_dir = voice_dir
        bus.subscribe(topics.DELETE_VOICE_DATA, self.on_delete)

    def on_delete(self, _msg) -> None:
        try:
            removed = delete_all(self.voice_dir)
        except Exception as exc:
            self._status("error", f"Couldn't delete your voice data: {exc}")
            return
        if self.dictation is not None:
            self.dictation.hotwords = None
        self._status("info", "Your voice data is deleted." if removed
                     else "There was no saved voice data to delete.")

    def _status(self, level: str, text: str) -> None:
        self.bus.publish(topics.TO_UI, messages.make(messages.STATUS, level=level, text=text))
