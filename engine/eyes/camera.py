"""Opens the webcam once and shares the newest frame. Frames stay in memory, never saved.

Route: Khang and Hung (Khang leads)
Publishes: raw frames to face_mesh
Subscribes: nothing
APIs: OpenCV VideoCapture with DirectShow (or MSMF), MJPG, 1280x720 at 30 fps by default

Usage:
    cam = Camera()
    cam.start()
    frame_id = 0
    while True:
        frame, t_ms, frame_id = cam.wait_next(frame_id)
        ...
    cam.stop()
"""
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from engine.config import settings

log = logging.getLogger(__name__)

BACKENDS = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY}


@dataclass
class CameraMode:
    index: int
    backend: str
    fourcc: str
    width: int   # what the camera actually delivers, read from a real frame
    height: int
    fps_reported: float


def fourcc_to_str(value: float) -> str:
    code = int(value)
    text = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))
    if not all(32 < ord(ch) < 127 for ch in text):
        return "?"  # the driver didn't report a readable format
    return text


def open_capture(index: int, backend: str, fourcc: str, width: int, height: int, fps: int):
    """Open a camera and ask for a mode. Returns (cap, mode), or (None, None) if it won't open."""
    cap = cv2.VideoCapture(index, BACKENDS.get(backend, cv2.CAP_ANY))
    if not cap.isOpened():
        cap.release()
        return None, None
    # Order matters on Windows: format first, then size, then frame rate.
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keeps latency low where the backend supports it

    # Some webcams fail the first few reads while they start up, so give them half a second.
    frame = None
    for _ in range(10):
        ok, frame = cap.read()
        if ok and frame is not None:
            break
        frame = None
        time.sleep(0.05)
    if frame is None:
        cap.release()
        return None, None
    mode = CameraMode(
        index=index,
        backend=backend,
        fourcc=fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC)),
        width=frame.shape[1],
        height=frame.shape[0],
        fps_reported=cap.get(cv2.CAP_PROP_FPS),
    )
    return cap, mode


class Camera:
    """Reads frames on a background thread and keeps only the newest one."""

    def __init__(self, index=None, backend=None, fourcc=None, width=None, height=None, fps=None):
        self.index = settings.camera_index if index is None else index
        self.backend = backend or settings.camera_backend
        self.fourcc = settings.camera_fourcc if fourcc is None else fourcc
        self.width = width or settings.frame_width
        self.height = height or settings.frame_height
        self.fps = fps or settings.fps

        self.mode: CameraMode | None = None
        self._cap = None
        self._thread = None
        self._running = False
        self._cond = threading.Condition()
        self._frame: np.ndarray | None = None
        self._t_ms = 0
        self._frame_id = 0
        self._times = deque(maxlen=60)

    def start(self) -> CameraMode:
        self._cap, self.mode = open_capture(
            self.index, self.backend, self.fourcc, self.width, self.height, self.fps
        )
        if self._cap is None:
            raise RuntimeError(
                f"Couldn't open camera {self.index} with the '{self.backend}' backend. "
                "Close other apps that use the camera (Zoom, Teams, OBS, the Windows Camera app), "
                "or run `python -m tests.camera_probe` to find a working index and mode."
            )
        if (self.mode.width, self.mode.height) != (self.width, self.height):
            log.warning(
                "Asked for %dx%d but the camera gives %dx%d",
                self.width, self.height, self.mode.width, self.mode.height,
            )
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()
        return self.mode

    def _loop(self) -> None:
        while self._running:
            ok, frame = self._cap.read()  # blocks until the camera delivers the next frame
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            now = time.monotonic()
            with self._cond:
                self._frame = frame
                self._t_ms = int(now * 1000)
                self._frame_id += 1
                self._times.append(now)
                self._cond.notify_all()

    def latest(self):
        """Newest frame (BGR, unmirrored), its time in ms, and its id. The frame is None before the first one."""
        with self._cond:
            return self._frame, self._t_ms, self._frame_id

    def wait_next(self, last_id: int, timeout: float = 1.0):
        """Block until a frame newer than last_id arrives, then return it like latest()."""
        with self._cond:
            self._cond.wait_for(lambda: self._frame_id != last_id or not self._running, timeout)
            return self._frame, self._t_ms, self._frame_id

    @property
    def measured_fps(self) -> float:
        """Frames per second actually arriving, over the last 60 frames."""
        with self._cond:
            if len(self._times) < 2:
                return 0.0
            span = self._times[-1] - self._times[0]
            return (len(self._times) - 1) / span if span > 0 else 0.0

    def stop(self) -> None:
        self._running = False
        with self._cond:
            self._cond.notify_all()
        if self._thread:
            self._thread.join(timeout=2)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
