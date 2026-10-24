"""Opens the webcam once and shares the newest frame. Frames stay in memory, never saved.

Route: Khang and Hung (Khang leads)
Publishes: raw frames to face_mesh
Subscribes: nothing
APIs: OpenCV VideoCapture(index, cv2.CAP_DSHOW), MJPG fourcc, 1280x720 at 30 fps
"""

class Camera:
    def start(self) -> None:
        raise NotImplementedError

    def latest(self):
        """Newest frame (BGR numpy array) and its timestamp in ms."""
        raise NotImplementedError
