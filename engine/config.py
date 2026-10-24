"""Settings with their defaults. The console's settings panel changes these at runtime.

Each laptop can override any setting in config.local.json at the repo root
(git ignores it). Example for a laptop without an NVIDIA GPU:
    {"whisper_device": "cpu", "whisper_model": "small", "use_ui_venus": false}
"""
import json
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass
class Settings:
    # Camera
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    fps: int = 30
    face_model_path: str = "models/face_landmarker.task"
    blink_threshold: float = 0.5

    # Pointing
    dwell_ms: int = 800
    sensitivity: float = 1.0

    # UI-Venus
    use_ui_venus: bool = True
    ui_venus_model: str = "inclusionAI/UI-Venus-1.5-2B"
    ui_venus_device: str = "cuda"
    crop_width: int = 1280
    crop_height: int = 720

    # Voice
    sample_rate: int = 16000
    pause_seconds: float = 1.5  # user setting, 1 to 4
    whisper_model: str = "large-v3-turbo"
    whisper_device: str = "cuda"

    # Engine <-> UI
    ws_host: str = "127.0.0.1"
    ws_port: int = 8765


def load() -> Settings:
    s = Settings()
    local = Path(__file__).resolve().parent.parent / "config.local.json"
    if local.exists():
        known = {f.name for f in fields(Settings)}
        for key, value in json.loads(local.read_text(encoding="utf-8")).items():
            if key in known:
                setattr(s, key, value)
            else:
                print(f"config.local.json: unknown setting '{key}' ignored")
    return s


settings = load()
