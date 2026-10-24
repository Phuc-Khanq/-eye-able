"""One Euro filter for gaze, and dwell detection.

Route: Khang and Hung (Khang leads)
Publishes: GazePoint (gaze), dwell
Subscribes: raw gaze predictions
APIs: own code. One Euro filter: Casiez, Roussel and Vogel, CHI 2012.

The One Euro filter smooths hard when the signal is still (removes jitter) and lightly when it
moves fast (keeps lag low). Two knobs:
  min_cutoff  lower = steadier at rest, but more lag on slow moves (Hz)
  beta        higher = less lag on fast moves (eye jumps)
"""
import math


def _alpha(dt: float, cutoff: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 2.0, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self) -> None:
        self._t = None
        self._x = 0.0
        self._dx = 0.0

    def __call__(self, t_s: float, value: float) -> float:
        if self._t is None or not math.isfinite(self._x):
            self._t, self._x, self._dx = t_s, value, 0.0
            return value
        dt = t_s - self._t
        if dt <= 0:
            return self._x
        a_d = _alpha(dt, self.d_cutoff)
        dx = (value - self._x) / dt
        self._dx = a_d * dx + (1 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        a = _alpha(dt, cutoff)
        self._x = a * value + (1 - a) * self._x
        self._t = t_s
        return self._x


class PointSmoother:
    """Two One Euro filters, for x and y."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 2.0) -> None:
        self.fx = OneEuroFilter(min_cutoff, beta)
        self.fy = OneEuroFilter(min_cutoff, beta)

    def __call__(self, t_s: float, x: float, y: float) -> tuple[float, float]:
        return self.fx(t_s, x), self.fy(t_s, y)

    def reset(self) -> None:
        self.fx.reset()
        self.fy.reset()


class DwellDetector:
    """Progress 0 to 1 while the point stays inside a circle; 1 means the dwell finished."""

    def __init__(self, radius_px: float = 60.0, dwell_ms: int = 800) -> None:
        self.radius_px = radius_px
        self.dwell_ms = dwell_ms
        self._anchor = None
        self._start_ms = 0

    def update(self, t_ms: int, x: float, y: float) -> float:
        if self._anchor is None or math.hypot(x - self._anchor[0], y - self._anchor[1]) > self.radius_px:
            self._anchor = (x, y)
            self._start_ms = t_ms
            return 0.0
        return min(1.0, (t_ms - self._start_ms) / self.dwell_ms)

    def reset(self) -> None:
        self._anchor = None
