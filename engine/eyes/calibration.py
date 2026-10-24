"""Gaze model: eye and head features -> screen point, trained on every saved session of a profile.

Route: Khang and Hung (Khang leads)
Publishes: calib_target, calib_result, the fitted gaze model
Subscribes: EyeFeatures, HeadPose, calibrate messages
APIs: NumPy (closed-form ridge regression). Model saved as JSON in calibration/ (numbers only, git-ignored).
      Training data lives in calibration/sessions/ (engine/eyes/gaze_data.py).

How it works
  Features per frame: h, v, lid, blend_v (eye_features) and yaw, pitch (head_pose, zeroed at the
  middle dot). Each dot becomes one row: the median over about a second of looking at it.
  One shared mapping for all sessions, plus one shift per session. Sitting a bit differently moves
  the whole gaze map; the shift absorbs that, so the shared mapping learns from all the data.
  In use: load the trained model, look at a few dots (recenter) to find today's shift, done.
  Term sets are picked by error on data the fit didn't see:
    one session    leave-one-dot-out
    more sessions  leave-one-session-out: train on the others, re-center on the held-out session's
                   middle dot (or its re-center dots), measure its other dots. That's real use.
  x: "small" (h, yaw, h^2, h*v) or "full" (all squares and products of h, v, yaw, pitch)
  y: "eye", "eye+mediapipe", "mediapipe" or "full" (squares and products of h, v, lid, pitch, blend_v)
  Live inputs are held to the trained range plus one spread (CLIP_SD), so an unusual head turn can't
  be multiplied into a huge jump.
"""
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from engine.core.types import EyeFeatures, HeadPose

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = REPO_ROOT / "calibration" / "gaze_model.json"
FEATURES = ("h", "v", "lid", "yaw", "pitch", "blend_v")
IDX = {name: i for i, name in enumerate(FEATURES)}
ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
MIN_DOTS = 6
VERSION = 4
CLIP_SD = 1.0
EVAL_ROLES = ("calib", "check")


def grid_targets(screen_w: int, screen_h: int, points: int = 25, margin: float = 0.1) -> list[tuple[float, float]]:
    """Dots on an n x n grid (9 -> 3x3, 16 -> 4x4, 25 -> 5x5) in screen pixels. The middle dot comes
    first (head pose is zeroed there), then the rest row by row in a snake so the eyes move little."""
    n = max(2, int(round(math.sqrt(points))))
    xs = np.linspace(margin, 1 - margin, n) * screen_w
    ys = np.linspace(margin, 1 - margin, n) * screen_h
    dots = []
    for r, y in enumerate(ys):
        row = [(float(x), float(y)) for x in xs]
        dots.extend(row if r % 2 == 0 else row[::-1])
    center = (screen_w / 2, screen_h / 2)
    if n % 2 == 1:
        dots.remove(min(dots, key=lambda d: math.hypot(d[0] - center[0], d[1] - center[1])))
    return [center] + dots


def feature_row(eyes: EyeFeatures, head: HeadPose | None) -> np.ndarray:
    yaw = head.yaw if head is not None else 0.0
    pitch = head.pitch if head is not None else 0.0
    return np.array([eyes.h, eyes.v, eyes.lid, yaw, pitch, eyes.blend_v], dtype=np.float64)


def _cols(z, *names):
    return [z[:, IDX[n]] for n in names]


def _quad(z, names):
    c = _cols(z, *names)
    return np.column_stack(c + [a * a for a in c] + [c[i] * c[j] for i in range(len(c)) for j in range(i + 1, len(c))])


def _x_small(z):
    h, v, yaw = _cols(z, "h", "v", "yaw")
    return np.column_stack([h, yaw, h * h, h * v])


def _y_eye(z):
    h, v, lid, pitch = _cols(z, "h", "v", "lid", "pitch")
    return np.column_stack([v, lid, pitch, v * v, h * v])


def _y_both(z):
    h, v, lid, pitch, bv = _cols(z, "h", "v", "lid", "pitch", "blend_v")
    return np.column_stack([v, lid, pitch, bv, v * v, h * v])


def _y_mediapipe(z):
    h, lid, pitch, bv = _cols(z, "h", "lid", "pitch", "blend_v")
    return np.column_stack([bv, lid, pitch, bv * bv, h * bv])


X_TERMS = {"small": _x_small, "full": lambda z: _quad(z, ("h", "v", "yaw", "pitch"))}
Y_TERMS = {"eye": _y_eye, "eye+mediapipe": _y_both, "mediapipe": _y_mediapipe,
           "full": lambda z: _quad(z, ("h", "v", "lid", "pitch", "blend_v"))}


def _fit_w(X: np.ndarray, y: np.ndarray, g: np.ndarray, alpha: float) -> tuple[np.ndarray, dict]:
    """Ridge with one free shift per session (g): returns weights and each session's shift."""
    Xd, yd = X.copy(), y.copy()
    for k in np.unique(g):
        m = g == k
        Xd[m] -= X[m].mean(axis=0)
        yd[m] -= y[m].mean()
    sd = Xd.std(axis=0)
    sd[sd < 1e-9] = 1.0
    Z = Xd / sd
    w = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ yd) / sd
    shifts = {int(k): float(np.mean(y[g == k] - X[g == k] @ w)) for k in np.unique(g)}
    return w, shifts


@dataclass
class DotResult:
    target: tuple[float, float]     # screen pixels
    predicted: tuple[float, float]  # prediction from a fit that never saw this dot
    error_px: float


@dataclass
class CalibrationReport:
    mode: str                 # "dot" (one session, leave-one-dot-out) or "session" (leave-one-session-out)
    mean_px: float
    median_px: float
    max_px: float
    x_px: float
    y_px: float
    sessions: int
    dots_total: int
    x_terms: str
    y_terms: str
    no_recenter_px: float = float("nan")   # session mode: same test without the re-center
    single_px: float = float("nan")        # session mode: this session alone, leave-one-dot-out
    y_compare: dict[str, float] = field(default_factory=dict)
    dots: list[DotResult] = field(default_factory=list)  # the current session's dots, for drawing

    def summary(self) -> str:
        if self.mode == "dot":
            return (f"leave-one-dot-out error: mean {self.mean_px:.0f} px, median {self.median_px:.0f} px, "
                    f"worst {self.max_px:.0f} px (side to side {self.x_px:.0f}, up/down {self.y_px:.0f}; "
                    f"{self.dots_total} dots, 1 session)")
        return (f"trained on {self.sessions} sessions ({self.dots_total} dots). New-session test: mean {self.mean_px:.0f} px, "
                f"median {self.median_px:.0f}, worst {self.max_px:.0f} (side {self.x_px:.0f}, up/down {self.y_px:.0f})")

    def compare(self) -> str:
        if self.mode == "session":
            return (f"vs. this session alone: {self.single_px:.0f} px.  Without re-center: {self.no_recenter_px:.0f} px.  "
                    f"Inputs: x {self.x_terms}, y {self.y_terms}")
        parts = [f"{n} {e:.0f} px" + (" <- used" if n == self.y_terms else "") for n, e in self.y_compare.items()]
        return "Up/down error by input: " + ",  ".join(parts)


class GazeModel:
    """Maps a feature row (h, v, lid, yaw, pitch, blend_v) to a screen point in physical pixels."""

    def __init__(self, screen_w: int, screen_h: int) -> None:
        self.screen_w, self.screen_h = int(screen_w), int(screen_h)
        self.center = np.zeros(len(FEATURES))
        self.scale = np.ones(len(FEATURES))
        self.lo = np.full(len(FEATURES), -np.inf)
        self.hi = np.full(len(FEATURES), np.inf)
        self.x_terms, self.y_terms = "small", "eye"
        self.wx: np.ndarray | None = None
        self.wy: np.ndarray | None = None
        self.bias = (0.0, 0.0)          # today's shift (fractions of the screen)
        self.default_bias = (0.0, 0.0)  # average shift over the training sessions
        self.sessions = 0
        self.report: CalibrationReport | None = None

    # -- training ------------------------------------------------------------------------------
    def _normalize(self, F: np.ndarray) -> np.ndarray:
        return (F - self.center) / self.scale

    def fit_sessions(self, sessions, current: int | None = None, y_inputs: str = "auto") -> CalibrationReport:
        """sessions: gaze_data.Session list. current: index of the session being calibrated now
        (its own shift becomes today's shift). y_inputs: "auto" or one name from Y_TERMS."""
        if y_inputs != "auto" and y_inputs not in Y_TERMS:
            raise ValueError(f"y_inputs must be 'auto' or one of {list(Y_TERMS)}")
        M, T, G, R = [], [], [], []
        for k, s in enumerate(sessions):
            for d in s.dots:
                if len(d.rows):
                    M.append(np.median(d.rows, axis=0))
                    T.append(d.target)
                    G.append(k)
                    R.append(d.role)
        M, T, G, R = np.array(M).reshape(-1, len(FEATURES)), np.array(T).reshape(-1, 2), np.array(G), np.array(R)
        if len(M) < MIN_DOTS:
            raise ValueError(f"Need at least {MIN_DOTS} dots with data, got {len(M)}")
        cur = len(sessions) - 1 if current is None else current
        self.sessions = len(sessions)

        self.center = M.mean(axis=0)
        spread = M.std(axis=0)
        self.scale = np.where(spread < 1e-6, 1.0, spread)
        Z = self._normalize(M)
        self.lo, self.hi = Z.min(axis=0) - CLIP_SD, Z.max(axis=0) + CLIP_SD

        # Which test to run: leave-one-session-out needs at least 2 sessions with dots to measure.
        testable = [k for k in range(len(sessions)) if np.any((G == k) & np.isin(R, EVAL_ROLES))]
        mode = "session" if len(sessions) >= 2 and len(testable) >= 2 else "dot"

        def anchors(k):
            """Dots that re-center session k: its re-center dots, else its dot nearest the middle."""
            idx = np.where((G == k) & (R == "recenter"))[0]
            if len(idx):
                return idx
            own = np.where(G == k)[0]
            return own[[int(np.argmin(np.hypot(T[own, 0] - 0.5, T[own, 1] - 0.5)))]]

        def evaluate(terms, axis, alpha):
            """Predictions (fractions) for every tested dot, and the matching no-re-center predictions."""
            X, y = terms(Z), T[:, axis]
            preds, plain, idx_all = [], [], []
            if mode == "dot":
                own = np.where(G == cur)[0]
                for i in own:
                    keep = own[own != i]
                    w, sh = _fit_w(X[keep], y[keep], G[keep], alpha)
                    preds.append(X[i] @ w + sh[cur])
                    plain.append(preds[-1])
                    idx_all.append(i)
            else:
                for k in testable:
                    train = G != k
                    w, sh = _fit_w(X[train], y[train], G[train], alpha)
                    a = anchors(k)
                    shift = float(np.mean(y[a] - X[a] @ w))
                    avg = float(np.mean(list(sh.values())))
                    for i in np.where((G == k) & np.isin(R, EVAL_ROLES))[0]:
                        if i in a:
                            continue
                        preds.append(X[i] @ w + shift)
                        plain.append(X[i] @ w + avg)
                        idx_all.append(i)
            return np.array(preds), np.array(plain), np.array(idx_all, dtype=int)

        def best(term_sets, axis, size):
            results = {}
            for name, terms in term_sets.items():
                for alpha in ALPHAS:
                    p, plain, idx = evaluate(terms, axis, alpha)
                    if len(idx) == 0:
                        continue
                    err = float(np.mean(np.abs(p - T[idx, axis])) * size)
                    if name not in results or err < results[name][1]:
                        results[name] = (alpha, err, p, plain, idx)
            return results

        rx = best(X_TERMS, 0, self.screen_w)
        y_sets = Y_TERMS if y_inputs == "auto" else {y_inputs: Y_TERMS[y_inputs]}
        ry = best(y_sets, 1, self.screen_h)
        if not rx or not ry:
            raise ValueError("Not enough dots to test the model")
        self.x_terms = min(rx, key=lambda n: rx[n][1])
        self.y_terms = min(ry, key=lambda n: ry[n][1])
        ax, _, px, plain_x, idx = rx[self.x_terms]
        ay, _, py, plain_y, _ = ry[self.y_terms]

        err = np.hypot((px - T[idx, 0]) * self.screen_w, (py - T[idx, 1]) * self.screen_h)
        err_plain = np.hypot((plain_x - T[idx, 0]) * self.screen_w, (plain_y - T[idx, 1]) * self.screen_h)
        on_cur = G[idx] == cur
        report = CalibrationReport(
            mode=mode, mean_px=float(err.mean()), median_px=float(np.median(err)), max_px=float(err.max()),
            x_px=float(np.mean(np.abs(px - T[idx, 0])) * self.screen_w),
            y_px=float(np.mean(np.abs(py - T[idx, 1])) * self.screen_h),
            sessions=len(sessions), dots_total=len(M), x_terms=self.x_terms, y_terms=self.y_terms,
            no_recenter_px=float(err_plain.mean()) if mode == "session" else float("nan"),
            y_compare={n: r[1] for n, r in ry.items()},
            dots=[DotResult((float(T[i, 0] * self.screen_w), float(T[i, 1] * self.screen_h)),
                            (float(a * self.screen_w), float(b * self.screen_h)), float(e))
                  for i, a, b, e in zip(idx[on_cur], px[on_cur], py[on_cur], err[on_cur])])
        if mode == "session":  # what this session alone would have given, for comparison
            solo = GazeModel(self.screen_w, self.screen_h)
            try:
                report.single_px = solo.fit_sessions([sessions[cur]], 0, y_inputs).mean_px
            except ValueError:
                pass

        self.wx, sx = _fit_w(X_TERMS[self.x_terms](Z), T[:, 0], G, ax)
        self.wy, sy = _fit_w(Y_TERMS[self.y_terms](Z), T[:, 1], G, ay)
        self.default_bias = (float(np.mean(list(sx.values()))), float(np.mean(list(sy.values()))))
        self.bias = (sx.get(cur, self.default_bias[0]), sy.get(cur, self.default_bias[1]))
        self.report = report
        return report

    def fit(self, dots: list[tuple[tuple[float, float], np.ndarray]], y_inputs: str = "auto") -> CalibrationReport:
        """One session from (target in screen pixels, feature rows). Kept for tests and simple use."""
        from engine.eyes.gaze_data import Session
        s = Session(profile="adhoc", screen=(self.screen_w, self.screen_h))
        for t, rows in dots:
            s.add(t, rows, "calib")
        return self.fit_sessions([s], 0, y_inputs)

    # -- using it ------------------------------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self.wx is not None

    def _raw(self, row: np.ndarray) -> tuple[float, float]:
        z = np.clip(self._normalize(np.asarray(row, dtype=np.float64)[None, :]), self.lo, self.hi)
        return float(X_TERMS[self.x_terms](z)[0] @ self.wx), float(Y_TERMS[self.y_terms](z)[0] @ self.wy)

    def predict_row(self, row: np.ndarray, bias: tuple[float, float] | None = None) -> tuple[float, float]:
        """Screen point in physical pixels, allowed 5% past each edge so the edges stay reachable."""
        bx, by = self.bias if bias is None else bias
        rx, ry = self._raw(row)
        nx, ny = min(max(rx + bx, -0.05), 1.05), min(max(ry + by, -0.05), 1.05)
        return nx * self.screen_w, ny * self.screen_h

    def predict(self, eyes: EyeFeatures, head: HeadPose | None) -> tuple[float, float]:
        return self.predict_row(feature_row(eyes, head))

    def recenter(self, dots: list[tuple[tuple[float, float], np.ndarray]]) -> tuple[float, float]:
        """Sets today's shift from a few dots: (target in screen pixels, feature rows while looking at it).
        Returns how far the gaze had drifted, in pixels."""
        rx, ry = [], []
        for (tx, ty), rows in dots:
            rows = np.asarray(rows, dtype=np.float64)
            if len(rows):
                px, py = self._raw(np.median(rows, axis=0))
                rx.append(tx / self.screen_w - px)
                ry.append(ty / self.screen_h - py)
        if not rx:
            return (0.0, 0.0)
        old = self.bias
        self.bias = (float(np.mean(rx)), float(np.mean(ry)))
        return ((old[0] - self.bias[0]) * self.screen_w, (old[1] - self.bias[1]) * self.screen_h)

    # -- saving --------------------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"version": VERSION, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "features": list(FEATURES),
                "screen": [self.screen_w, self.screen_h], "center": self.center.tolist(), "scale": self.scale.tolist(),
                "lo": self.lo.tolist(), "hi": self.hi.tolist(), "x_terms": self.x_terms, "y_terms": self.y_terms,
                "wx": self.wx.tolist(), "wy": self.wy.tolist(), "bias": list(self.bias),
                "default_bias": list(self.default_bias), "sessions": self.sessions,
                "test_mean_px": self.report.mean_px if self.report else None}

    def save(self, path: Path | str | None = None) -> Path:
        path = Path(path) if path is not None else DEFAULT_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | str | None = None) -> "GazeModel":
        """Loads a trained model. Its shift is the average one; call recenter() for today's."""
        path = Path(path) if path is not None else DEFAULT_PATH
        d = json.loads(path.read_text(encoding="utf-8"))
        if d.get("version") != VERSION or d.get("features") != list(FEATURES):
            raise ValueError(f"{path} was made by an older version. Recalibrate: python -m tests.gaze_dot")
        m = cls(*d["screen"])
        m.center, m.scale = np.array(d["center"]), np.array(d["scale"])
        m.lo, m.hi = np.array(d["lo"]), np.array(d["hi"])
        m.x_terms, m.y_terms = d["x_terms"], d["y_terms"]
        m.wx, m.wy = np.array(d["wx"]), np.array(d["wy"])
        m.default_bias = tuple(d["default_bias"])
        m.bias = m.default_bias
        m.sessions = d.get("sessions", 0)
        return m
