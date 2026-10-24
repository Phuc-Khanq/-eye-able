"""Was the user's mouth moving? Lets the voice route keep only words the user said.

Route: Khang and Hung (Khang leads)
Publishes: nothing; the voice route asks it
Subscribes: topics.MOUTH (MouthSample: camera frame time and MediaPipe's jawOpen, every frame with a face)
APIs: plain Python. Times are time.monotonic() in milliseconds, the camera's clock; audio_in stamps
      utterances on the same clock. Nothing is saved; the last 30 s live in memory.

How "moving" is decided, per frame:
  talking   jawOpen swings by at least MOVE_RANGE within +-WINDOW_MS (the jaw goes up and down as you talk)
  or open   jawOpen is OPEN_ABOVE_REST over the resting level (a long open vowel)
The resting level is a low percentile of the last few seconds, so a mouth that rests slightly open still works.

Use:
    mouth.moving_fraction(start_ms, end_ms)  -> 0..1, or None when the face was seen in under half the frames
    (None means "can't tell": accept the speech rather than drop it)
Tune live with: python -m tests.mouth_check
"""
from __future__ import annotations

import bisect
import threading
from collections import deque

from engine.core import topics
from engine.core.types import MouthSample

KEEP_MS = 30_000          # how much history to keep
WINDOW_MS = 150           # half-width of the window the swing is measured over
MOVE_RANGE = 0.06         # jawOpen max - min inside the window that counts as talking
OPEN_ABOVE_REST = 0.15    # or this far above the resting level
REST_MS = 5_000           # the resting level comes from the last 5 s...
REST_PERCENTILE = 0.2     # ...as this low percentile
FRAME_MS = 33             # about 30 fps, until the real frame rate is known
MIN_FACE_SHARE = 0.5      # below this share of expected frames, moving_fraction says None


class MouthTracker:
    def __init__(self, bus=None) -> None:
        self._lock = threading.Lock()
        self._t: deque[int] = deque()
        self._open: deque[float] = deque()
        if bus is not None:
            bus.subscribe(topics.MOUTH, self.add)

    def add(self, s: MouthSample) -> None:
        """Called on the camera thread for every frame with a face."""
        with self._lock:
            if self._t and s.t_ms < self._t[-1]:
                return   # out of order: ignore
            self._t.append(int(s.t_ms))
            self._open.append(float(s.open))
            while self._t and self._t[-1] - self._t[0] > KEEP_MS:
                self._t.popleft()
                self._open.popleft()

    def _rest(self, t: list[int], o: list[float], at_ms: int) -> float:
        i0 = bisect.bisect_left(t, at_ms - REST_MS)
        i1 = bisect.bisect_right(t, at_ms)
        vals = sorted(o[i0:i1]) or [0.0]
        return vals[int(REST_PERCENTILE * (len(vals) - 1))]

    def moving_flags(self, start_ms: int, end_ms: int) -> list[tuple[int, bool]]:
        """(frame time, moving) for every frame with a face between start_ms and end_ms."""
        with self._lock:
            t, o = list(self._t), list(self._open)
        out = []
        i0, i1 = bisect.bisect_left(t, start_ms), bisect.bisect_right(t, end_ms)
        for i in range(i0, i1):
            w0, w1 = bisect.bisect_left(t, t[i] - WINDOW_MS), bisect.bisect_right(t, t[i] + WINDOW_MS)
            win = o[w0:w1]
            swing = max(win) - min(win)
            rest = self._rest(t, o, t[i])
            out.append((t[i], swing >= MOVE_RANGE or o[i] >= rest + OPEN_ABOVE_REST))
        return out

    def frame_ms(self) -> int:
        """Typical time between frames lately (a dim room can drop a webcam to 8 fps)."""
        with self._lock:
            t = list(self._t)[-61:]
        gaps = sorted(b - a for a, b in zip(t, t[1:]) if 0 < b - a < 500)
        return gaps[len(gaps) // 2] if gaps else FRAME_MS

    def moving_fraction(self, start_ms: float, end_ms: float) -> float | None:
        """Share of the span's face frames with a moving mouth, or None if the face was mostly not seen."""
        start_ms, end_ms = int(start_ms), int(end_ms)
        if end_ms <= start_ms:
            end_ms = start_ms + FRAME_MS
        flags = self.moving_flags(start_ms, end_ms)
        expected = max(1, (end_ms - start_ms) // self.frame_ms())
        if len(flags) < MIN_FACE_SHARE * expected:
            return None
        return sum(m for _, m in flags) / len(flags)
