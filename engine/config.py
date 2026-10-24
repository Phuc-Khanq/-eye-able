"""Settings with their defaults. The console's settings panel changes these at runtime."""
from dataclasses import dataclass


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
    ui_venus_model: str = "inclusionAI/UI-Venus-1.5-2B"
    crop_width: int = 1280
    crop_height: int = 720

    # Voice
    sample_rate: int = 16000
    pause_seconds: float = 1.5  # user setting, 1 to 4
    whisper_model: str = "large-v3-turbo"

    # Engine <-> UI
    ws_host: str = "127.0.0.1"
    ws_port: int = 8765


settings = Settings()
