"""Sends the webcam picture to the demo page, so the page can show your face next to its simulated desktop.

Route: Khang and Hung
Publishes: the UI message debug_frame {"jpeg": base64, "w", "h"}, about preview_fps times a second
Subscribes: nothing (the eyes service calls offer() with each camera frame); wakes only while engine/core/page_mode.py
            says the demo page is open
APIs: OpenCV for the resize and JPEG, base64

Privacy: a frame goes only to the page, over the WebSocket on 127.0.0.1 (and only to pages from allowed origins,
settings.ws_allowed_origins). Nothing is written to disk. When no page is showing, nothing is encoded or sent.
The encoding happens on its own thread, so the camera thread (and the cursor) never wait for it.
"""
from __future__ import annotations

import base64
import threading
import time

from engine.core import messages, topics


def encode_frame(frame, width: int, quality: int, mirror: bool) -> tuple[str, int, int]:
    """(base64 JPEG, w, h) of a BGR camera frame, shrunk to `width` px wide and mirrored like a mirror."""
    import cv2

    h, w = frame.shape[:2]
    if width and w > width:
        frame = cv2.resize(frame, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)
        h, w = frame.shape[:2]
    if mirror:
        frame = cv2.flip(frame, 1)
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("could not encode the camera frame")
    return base64.b64encode(buf.tobytes()).decode("ascii"), int(w), int(h)


class PreviewStream:
    def __init__(self, bus, settings, page, clock=time.monotonic, encoder=encode_frame) -> None:
        self.bus, self.page, self.clock, self.encoder = bus, page, clock, encoder
        self.interval = 1.0 / max(1, int(getattr(settings, "preview_fps", 12)))
        self.width = int(getattr(settings, "preview_width", 480))
        self.quality = int(getattr(settings, "preview_quality", 65))
        self.mirror = bool(getattr(settings, "preview_mirror", True))
        self._frame = None
        self._last = 0.0
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def offer(self, frame) -> None:
        """Camera thread: remember the newest frame. Cheap; does nothing unless the page is showing."""
        if frame is None or not self.page.active:
            return
        now = self.clock()
        if now - self._last < self.interval:
            return
        self._last = now
        self._frame = frame
        self._wake.set()

    def send_latest(self) -> bool:
        """Encode and send the newest offered frame (the worker thread calls this). True if one went out."""
        frame, self._frame = self._frame, None
        if frame is None:
            return False
        jpeg, w, h = self.encoder(frame, self.width, self.quality, self.mirror)
        self.bus.publish(topics.TO_UI, messages.make(messages.DEBUG_FRAME, jpeg=jpeg, w=w, h=h))
        return True

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="preview", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(1.0)
            self._wake.clear()
            if self._stop.is_set():
                break
            try:
                self.send_latest()
            except Exception as exc:    # a bad frame must never take the page, or the engine, down
                print(f"Preview: dropped a frame: {type(exc).__name__}: {exc}")
