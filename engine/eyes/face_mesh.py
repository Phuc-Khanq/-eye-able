"""Runs MediaPipe FaceLandmarker on each frame: 478 landmarks, 52 blendshapes, head matrix.

Route: Khang and Hung (Khang leads)
Publishes: FaceFrame
Subscribes: camera frames
APIs: MediaPipe Tasks FaceLandmarker, VIDEO mode, detect_for_video; options: output_face_blendshapes, output_facial_transformation_matrixes, num_faces=1. Check import paths against mediapipe 1.1.1.
"""

class FaceMesh:
    def __init__(self, model_path: str) -> None:
        raise NotImplementedError

    def process(self, frame_bgr, t_ms: int):
        """Return a FaceFrame, or None when no face is found. Don't mirror the frame first."""
        raise NotImplementedError
