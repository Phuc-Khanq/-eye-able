"""Runs MediaPipe FaceLandmarker on each frame: 478 landmarks, 52 blendshapes, head matrix.

Route: Khang and Hung (Khang leads)
Publishes: FaceFrame
Subscribes: camera frames
APIs: MediaPipe 1.1.1 Tasks FaceLandmarker in VIDEO mode (detect_for_video), on the CPU.
      Checked against the installed package: mp.tasks.vision.FaceLandmarker, mp.tasks.BaseOptions,
      mp.tasks.vision.RunningMode, mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb).
"""
from pathlib import Path

import cv2
import numpy as np

from engine.config import settings
from engine.core.types import FaceFrame

REPO_ROOT = Path(__file__).resolve().parents[2]


class FaceMesh:
    def __init__(self, model_path: str | None = None) -> None:
        import mediapipe as mp  # imported here so the engine starts fast

        self._mp = mp
        path = Path(model_path or settings.face_model_path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"Face model not found at {path}. Run Install EYEable.bat (or scripts\\install.ps1) to download it.")

        vision = mp.tasks.vision
        options = vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts = -1

    def process(self, frame_bgr: np.ndarray, t_ms: int) -> FaceFrame | None:
        """Return a FaceFrame, or None when no face is found. Pass the unmirrored camera frame."""
        ts = max(int(t_ms), self._last_ts + 1)  # VIDEO mode needs strictly increasing timestamps
        self._last_ts = ts
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, ts)
        if not result.face_landmarks:
            return None

        landmarks = np.array([(p.x, p.y, p.z) for p in result.face_landmarks[0]], dtype=np.float32)
        blendshapes = (
            {c.category_name: float(c.score) for c in result.face_blendshapes[0]}
            if result.face_blendshapes else {}
        )
        matrix = (
            np.asarray(result.facial_transformation_matrixes[0], dtype=np.float32)
            if result.facial_transformation_matrixes else np.eye(4, dtype=np.float32)
        )
        h, w = frame_bgr.shape[:2]
        return FaceFrame(t_ms=ts, width=w, height=h, landmarks=landmarks,
                         blendshapes=blendshapes, head_matrix=matrix)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
