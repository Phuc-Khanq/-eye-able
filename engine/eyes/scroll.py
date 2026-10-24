"""Scrolling with the head, smooth like a trackpad.

Route: Khang and Hung (Khang leads)
Publishes: nothing itself; EyesService drives it and tells the UI
APIs: HeadScroll (head pitch -> scroll speed), SmoothScroller (speed -> small, evenly spaced wheel steps)

How it feels: in scroll mode the cursor stays where it is (the wheel goes to the window under it) and the
head works like a joystick. The first 0.4 s sets "straight ahead" (the median head pitch, so saying
"scroll" while nodding doesn't matter). Then chin down scrolls down, chin up scrolls up: nothing inside the
dead zone (2 degrees), a gentle crawl just outside it for reading, faster the further you tilt, a cap at
25 notches a second (about 2500 px in a browser) at about 10 degrees from straight. Speed is not position, so you hold a
small tilt to keep scrolling and come back to straight to stop.

What makes it smooth (SmoothScroller):
  - the speed eases toward what the head asks for (0.15 s to speed up, 0.3 s to slow down), so it never
    jumps, and when you leave the mode or lose the face it coasts to a stop like a trackpad after a swipe
  - the wheel gets a step about 60 times a second, each a fraction of a notch (Windows takes 1/120 of a
    notch as the smallest step), not 30 camera frames of 3-notch jumps
  - "scroll up" / "scroll down" by voice glide the same way: a short eased push of 5 notches in 0.7 s

Speeds are in notches per second; positive = scroll up (content moves down), like os_input.scroll.
"""
from __future__ import annotations

import math
import statistics
import threading
import time

from engine.eyes.smoothing import OneEuroFilter

# --- head -> speed
DEAD_DEG = 2.0           # tilt this far from straight ahead does nothing
GAIN = 1.5               # notches per second one degree past the dead zone
POWER = 1.3              # >1: a gentle crawl for reading, then faster
MAX_RATE = 25.0          # notches per second at most
SETTLE_MS = 400          # "straight ahead" = the median pitch of the first 0.4 s
MAX_STEP_DEG = 10.0      # a bigger jump between two frames is a tracking glitch (skipped once)
ANGLE_MIN_CUTOFF = 1.0   # One Euro on the pitch, same as the head pointer
ANGLE_BETA = 0.3

# --- speed -> wheel
TAU_UP_S = 0.15          # speeding up
TAU_DOWN_S = 0.30        # slowing down, braking, coasting after the mode ends
UNITS_PER_NOTCH = 120    # Windows WHEEL_DELTA
MIN_UNITS = 10           # don't send steps smaller than this (apps that ignore tiny deltas)
GLIDE_NOTCHES = 5.0      # one "scroll down" by voice
GLIDE_S = 0.7
RATE_HZ = 60


class HeadScroll:
    """Head pitch (degrees, positive = chin down) -> scroll speed in notches per second."""

    def __init__(self) -> None:
        self._f = OneEuroFilter(ANGLE_MIN_CUTOFF, ANGLE_BETA)
        self.start()

    def start(self) -> None:
        """A new scroll session: the next frames set 'straight ahead'."""
        self._f.reset()
        self._t0: int | None = None
        self._samples: list[float] = []
        self._raw: float | None = None
        self._wild = 0
        self.anchor: float | None = None
        self.offset = 0.0     # degrees from straight ahead (chin down positive), for the UI
        self.rate = 0.0

    @property
    def settled(self) -> bool:
        return self.anchor is not None

    def update(self, t_ms: int, pitch: float) -> float:
        """One camera frame. Returns the wanted speed (positive = up, negative = down)."""
        if self._raw is not None and abs(pitch - self._raw) > MAX_STEP_DEG:
            self._wild += 1
            if self._wild < 2:
                return self.rate                       # one wild frame: keep the last speed
        self._wild = 0
        self._raw = pitch
        p = self._f(t_ms / 1000.0, pitch)
        if self._t0 is None:
            self._t0 = t_ms
        if self.anchor is None:
            self._samples.append(p)
            if t_ms - self._t0 < SETTLE_MS:
                return 0.0
            self.anchor = statistics.median(self._samples)
        self.offset = p - self.anchor
        beyond = max(0.0, abs(self.offset) - DEAD_DEG)
        mag = min(MAX_RATE, GAIN * beyond ** POWER)
        self.rate = -math.copysign(mag, self.offset) if mag > 0 else 0.0
        return self.rate


class SmoothScroller:
    """Eases the wanted speed and turns it into small, evenly spaced wheel steps on its own thread.

    scroll_fn(units): units are 1/120 of a notch, positive = up. step(dt) does one tick (the thread calls
    it 60 times a second; tests call it directly).
    """

    def __init__(self, scroll_fn, rate_hz: int = RATE_HZ, clock=time.monotonic, on_error=None) -> None:
        self._scroll = scroll_fn
        self._period = 1.0 / rate_hz
        self._clock = clock
        self._on_error = on_error
        self._lock = threading.Lock()
        self._target = 0.0
        self.rate = 0.0                       # the eased speed, notches per second
        self._glides: list[list[float]] = []  # [age_s, notches, secs]
        self._acc = 0.0                       # wheel units not sent yet
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -------------------------------------------------------------- inputs (any thread)
    def set_rate(self, notches_per_s: float) -> None:
        """What the head asks for now (positive = up). 0 lets it coast to a stop."""
        with self._lock:
            self._target = float(notches_per_s)

    def glide(self, direction: str, notches: float = GLIDE_NOTCHES, secs: float = GLIDE_S) -> None:
        """A short eased push, e.g. "scroll down" by voice."""
        with self._lock:
            self._glides.append([0.0, notches if direction == "up" else -notches, secs])

    @property
    def busy(self) -> bool:
        with self._lock:
            return abs(self.rate) > 0.02 or abs(self._target) > 0 or bool(self._glides)

    # -------------------------------------------------------------- one tick
    def step(self, dt: float) -> int:
        """Advance dt seconds; sends and returns the wheel units for this tick (0 if none)."""
        dt = min(max(dt, 0.0), 0.1)
        with self._lock:
            target = self._target
            braking = target * self.rate < 0 or abs(target) < abs(self.rate)
            tau = TAU_DOWN_S if braking else TAU_UP_S
            self.rate += (target - self.rate) * (1.0 - math.exp(-dt / tau))
            if target == 0.0 and abs(self.rate) < 0.02:
                self.rate = 0.0
            notches = self.rate * dt
            alive = []
            for g in self._glides:
                age, total, secs = g
                a, b = min(age, secs), min(age + dt, secs)
                # a raised-cosine speed profile: starts and ends at 0, adds up to `total` notches
                notches += total / 2.0 * (math.cos(math.pi * a / secs) - math.cos(math.pi * b / secs))
                g[0] = age + dt
                if g[0] < secs:
                    alive.append(g)
            self._glides = alive
            self._acc += notches * UNITS_PER_NOTCH
            if self.rate == 0.0 and target == 0.0 and not alive and abs(self._acc) < MIN_UNITS:
                self._acc = 0.0                  # idle: drop the leftover so it doesn't trickle out later
            units = 0
            if abs(self._acc) >= MIN_UNITS:
                units = int(self._acc)
                self._acc -= units
        if units:
            self._scroll(units)
        return units

    # -------------------------------------------------------------- thread
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="scroll", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            self._target, self.rate, self._glides, self._acc = 0.0, 0.0, [], 0.0
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def _run(self) -> None:
        last = self._clock()
        warned = 0.0
        while not self._stop.is_set():
            time.sleep(self._period if self.busy else 0.05)
            now = self._clock()
            try:
                self.step(now - last)
            except Exception as exc:   # a failed wheel call must not stop the thread
                if self._on_error is not None and now - warned > 10:
                    warned = now
                    self._on_error(exc)
            last = now
