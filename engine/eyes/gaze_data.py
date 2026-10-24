"""Saved gaze sessions: the numbers from every calibration and check, so the model trains on all of them.

Route: Khang and Hung (Khang leads)
APIs: JSON files in calibration/sessions/<profile>/ (git ignores calibration/). Numbers only: eye and
      head features per frame and where the dot was. No images, no video.

A profile is one person on one setup (same screen, same camera position). Data from a different screen
size is kept but skipped for training, because the same eye movement covers a different part of the screen.
Dot roles:
  calib     a calibration dot
  recenter  a dot used to re-center a session (the model's shift for that sitting)
  check     a check dot the model hadn't seen
  sweep     part of the optional head sweep (training only)
"""
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "calibration" / "sessions"
VERSION = 1


@dataclass
class DotData:
    target: tuple[float, float]   # where the dot was, as fractions of the screen (0..1)
    rows: np.ndarray              # (n, 6) feature rows collected while looking at it
    role: str                     # calib | recenter | check | sweep


@dataclass
class Session:
    profile: str
    screen: tuple[int, int]
    camera: str = ""
    iris_mode: str = "center"
    kind: str = "calibration"     # calibration | quick
    created: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    dots: list[DotData] = field(default_factory=list)
    path: Path | None = None

    def add(self, target_px: tuple[float, float], rows, role: str) -> None:
        rows = np.asarray(rows, dtype=np.float64).reshape(-1, 6)
        if len(rows):
            self.dots.append(DotData((target_px[0] / self.screen[0], target_px[1] / self.screen[1]), rows, role))


def safe_name(profile: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", profile.strip()) or "default"


def save_session(s: Session) -> Path:
    """Writes the session (again, to the same file, if it was saved before)."""
    if s.path is None:
        folder = DATA_DIR / safe_name(s.profile)
        folder.mkdir(parents=True, exist_ok=True)
        stem = time.strftime("%Y%m%d-%H%M%S") + f"-{s.kind}"
        s.path, n = folder / f"{stem}.json", 2
        while s.path.exists():  # never overwrite another session
            s.path, n = folder / f"{stem}-{n}.json", n + 1
    data = {"version": VERSION, "profile": s.profile, "created": s.created, "screen": list(s.screen),
            "camera": s.camera, "iris_mode": s.iris_mode, "kind": s.kind,
            "dots": [{"target": [round(d.target[0], 5), round(d.target[1], 5)], "role": d.role,
                      "rows": np.round(d.rows, 6).tolist()} for d in s.dots]}
    s.path.write_text(json.dumps(data), encoding="utf-8")
    return s.path


def load_sessions(profile: str, screen: tuple[int, int] | None = None) -> tuple[list[Session], list[str]]:
    """All saved sessions of a profile, oldest first, plus notes about any that were skipped."""
    folder = DATA_DIR / safe_name(profile)
    sessions, notes = [], []
    for path in sorted(folder.glob("*.json")) if folder.exists() else []:
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            s = Session(profile=d["profile"], screen=tuple(d["screen"]), camera=d.get("camera", ""),
                        iris_mode=d.get("iris_mode", "center"), kind=d.get("kind", "calibration"),
                        created=d.get("created", ""), path=path,
                        dots=[DotData(tuple(x["target"]), np.array(x["rows"], dtype=np.float64).reshape(-1, 6), x["role"])
                              for x in d["dots"]])
        except Exception as e:
            notes.append(f"skipped {path.name}: unreadable ({type(e).__name__})")
            continue
        if screen is not None and tuple(s.screen) != tuple(screen):
            notes.append(f"skipped {path.name}: screen {s.screen[0]}x{s.screen[1]}, not {screen[0]}x{screen[1]}")
            continue
        sessions.append(s)
    return sessions, notes


def list_profiles() -> list[str]:
    return sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir()) if DATA_DIR.exists() else []
