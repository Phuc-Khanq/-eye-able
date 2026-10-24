"""Moves the real cursor smoothly: a dead zone against wobble, then a glide at 120 updates a second.

Route: Khang and Hung (Khang leads)
Publishes: OS cursor moves
Subscribes: cursor targets from the pointer (head, gaze or hybrid), snaps from UI-Venus
APIs: a background thread; os_input.move (pynput) does the moving. Nothing is saved.

Why: the camera gives a new head pose 30 times a second. Moving the cursor only then makes it hop
in visible steps (a 600 px/s move is 20 px per hop), and the head's natural sway makes it shimmer
when it should rest. So the pointer only sets a target: StableCursor ignores wobble under still_px,
and this thread glides the real cursor toward the target 120 times a second (time constant glide_ms,
about 40 ms of extra lag). snap() jumps straight there, for UI-Venus answers and clicks.
"""
import math
import threading
import time
from collections import deque

from engine.eyes.smoothing import StableCursor


class CursorDriver:
    def __init__(self, move_fn, rate_hz: float = 120.0, glide_ms: float = 40.0, still_px: float = 8.0) -> None:
        self.move_fn = move_fn
        self.period = 1.0 / rate_hz
        self.glide_s = max(glide_ms, 0.0) / 1000.0
        self.stable = StableCursor(still_px)
        self._lock = threading.Lock()
        self._target: tuple[float, float] | None = None
        self._pos: tuple[float, float] | None = None
        self._sent: tuple[int, int] | None = None
        self._trail: deque[tuple[float, float, float]] = deque()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.enabled = True        # False = leave the real cursor alone (paused)

    # -- from the camera thread ------------------------------------------------------------------
    def set_target(self, x: float, y: float) -> None:
        with self._lock:
            self._target = self.stable(x, y)

    def snap(self, x: float, y: float) -> None:
        """Jump now (UI-Venus answer, click). The dead zone restarts here."""
        with self._lock:
            self.stable.reset(x, y)
            self._target = self._pos = (x, y)
        self._send(x, y)

    def click(self, x: int, y: int, click_fn) -> None:
        """Snap to (x, y) and click there; the glide can't pull the cursor away mid-click."""
        with self._lock:
            self.stable.reset(x, y)
            self._target = self._pos = (x, y)
            click_fn(x, y)
            self._sent = (int(round(x)), int(round(y)))

    @property
    def position(self) -> tuple[float, float] | None:
        return self._pos

    def wobble_px(self) -> float:
        """How much the real cursor moved around its average over the last second (RMS, px)."""
        pts = list(self._trail)
        if len(pts) < 10:
            return 0.0
        mx = sum(p[1] for p in pts) / len(pts)
        my = sum(p[2] for p in pts) / len(pts)
        return math.sqrt(sum((p[1] - mx) ** 2 + (p[2] - my) ** 2 for p in pts) / len(pts))

    # -- the glide thread -----------------------------------------------------------------------
    def start(self) -> "CursorDriver":
        self._thread = threading.Thread(target=self._run, name="cursor-glide", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def step(self, dt: float) -> tuple[float, float] | None:
        """One glide step (the thread calls this; tests call it directly). Returns the new position."""
        with self._lock:
            if self._target is None:
                return None
            tx, ty = self._target
            if self._pos is None or self.glide_s <= 0:
                self._pos = (tx, ty)
            else:
                a = 1.0 - math.exp(-dt / self.glide_s)
                px, py = self._pos
                self._pos = (px + (tx - px) * a, py + (ty - py) * a)
            return self._pos

    def _send(self, x: float, y: float) -> None:
        if not self.enabled:
            return
        xi, yi = int(round(x)), int(round(y))
        if (xi, yi) != self._sent:
            self.move_fn(xi, yi)
            self._sent = (xi, yi)

    def _run(self) -> None:
        last = time.perf_counter()
        while not self._stop.is_set():
            time.sleep(self.period)   # Python 3.11+ on Windows sleeps with a high-resolution timer
            now = time.perf_counter()
            pos = self.step(now - last)
            last = now
            if pos is None:
                continue
            self._trail.append((now, pos[0], pos[1]))
            while self._trail and now - self._trail[0][0] > 1.0:
                self._trail.popleft()
            self._send(*pos)
