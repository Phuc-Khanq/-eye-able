"""Gaze calibration: features at each dot -> a small regression to screen x, y.

Route: Khang and Hung (Khang leads)
Publishes: calib_target, calib_result, the fitted gaze model
Subscribes: EyeFeatures, HeadPose, calibrate messages
APIs: NumPy (closed-form ridge regression). Saved as JSON in calibration/ (numbers only, no images;
      git ignores the folder).

How it works
  Features per frame: h, v, lid and blend_v (eye_features) and yaw, pitch (head_pose, relative to
  the pose at the center dot). blend_v is MediaPipe's own up/down eye score (eyeLookUp/Down
  blendshapes). For each dot we keep the median of each feature over about a second.
  x is predicted from h, yaw, h^2 and h*v. Up and down is the weak axis on a webcam, so y tries three
  input sets and keeps the one with the lowest leave-one-dot-out error:
    "eye"          v, lid, pitch (+ v^2, h*v)              our own iris measurement
    "eye+mediapipe" the same plus blend_v                  both together
    "mediapipe"    blend_v, lid, pitch (+ blend_v^2, h*blend_v)
  Live inputs are held to the calibration range plus one spread on each side (CLIP_SD). The head barely
  moves during calibration, so without this a small head turn later gets multiplied into a big jump.
  Few terms on purpose: 25 dots can't support a big model. Ridge strength is also picked by
  leave-one-dot-out. That error is the honest number: how far off a spot is when the model never saw it.
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
ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
MIN_DOTS = 6
VERSION = 3
CLIP_SD = 1.0  # live inputs may go this many spreads past the calibration range, no further


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


def _terms_x(z: np.ndarray) -> np.ndarray:
    h, v, _lid, yaw, _pitch, _bv = z.T
    return np.column_stack([h, yaw, h * h, h * v])


def _y_eye(z: np.ndarray) -> np.ndarray:
    h, v, lid, _yaw, pitch, _bv = z.T
    return np.column_stack([v, lid, pitch, v * v, h * v])


def _y_both(z: np.ndarray) -> np.ndarray:
    h, v, lid, _yaw, pitch, bv = z.T
    return np.column_stack([v, lid, pitch, bv, v * v, h * v])


def _y_mediapipe(z: np.ndarray) -> np.ndarray:
    h, _v, lid, _yaw, pitch, bv = z.T
    return np.column_stack([bv, lid, pitch, bv * bv, h * bv])


Y_INPUTS = {"eye": _y_eye, "eye+mediapipe": _y_both, "mediapipe": _y_mediapipe}


def _ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> dict:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-9] = 1.0
    Z = (X - mu) / sd
    ym = float(y.mean())
    w = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y - ym))
    return {"mu": mu, "sd": sd, "w": w, "ym": ym}


def _ridge_predict(p: dict, X: np.ndarray) -> np.ndarray:
    return ((X - p["mu"]) / p["sd"]) @ p["w"] + p["ym"]


def _loo(terms, Z: np.ndarray, t: np.ndarray, alpha: float, Ze: np.ndarray, te: np.ndarray) -> np.ndarray:
    """Leave-one-dot-out predictions for one axis. The head-sweep rows (Ze, te) are always trained on."""
    X, Xe = terms(Z), terms(Ze)
    preds = np.zeros(len(Z))
    for i in range(len(Z)):
        keep = np.arange(len(Z)) != i
        fit = _ridge_fit(np.vstack([X[keep], Xe]), np.concatenate([t[keep], te]), alpha)
        preds[i] = _ridge_predict(fit, X[i:i + 1])[0]
    return preds


def _best_alpha(terms, Z: np.ndarray, t: np.ndarray, size_px: int, Ze: np.ndarray, te: np.ndarray):
    """(alpha, leave-one-dot-out predictions, mean error in pixels) for the best ridge strength."""
    best = None
    for alpha in ALPHAS:
        preds = _loo(terms, Z, t, alpha, Ze, te)
        err = float(np.mean(np.abs(preds - t)) * size_px)
        if best is None or err < best[2]:
            best = (alpha, preds, err)
    return best


@dataclass
class DotResult:
    target: tuple[float, float]
    predicted: tuple[float, float]  # leave-one-dot-out prediction
    error_px: float
    samples: int


@dataclass
class CalibrationReport:
    loo_mean_px: float
    loo_median_px: float
    loo_max_px: float
    loo_x_px: float          # mean side-to-side error
    loo_y_px: float          # mean up-and-down error
    train_mean_px: float
    y_inputs: str
    sweep_n: int = 0         # head-sweep chunks used
    y_compare: dict[str, float] = field(default_factory=dict)  # y error per input set
    dots: list[DotResult] = field(default_factory=list)

    def summary(self) -> str:
        return (f"leave-one-dot-out error: mean {self.loo_mean_px:.0f} px, median {self.loo_median_px:.0f} px, "
                f"worst {self.loo_max_px:.0f} px (side to side {self.loo_x_px:.0f}, up/down {self.loo_y_px:.0f}; "
                f"{len(self.dots)} dots" + (", head sweep" if self.sweep_n else "") + ")")

    def compare(self) -> str:
        parts = [f"{name} {err:.0f} px" + (" <- used" if name == self.y_inputs else "")
                 for name, err in self.y_compare.items()]
        return "Up/down error by input: " + ",  ".join(parts)


class GazeModel:
    """Maps a feature row (h, v, lid, yaw, pitch, blend_v) to a screen point in physical pixels."""

    def __init__(self, screen_w: int, screen_h: int) -> None:
        self.screen_w = int(screen_w)
        self.screen_h = int(screen_h)
        self.center = np.zeros(len(FEATURES))
        self.scale = np.ones(len(FEATURES))
        self.px: dict | None = None
        self.py: dict | None = None
        self.alpha_x = self.alpha_y = 1.0
        self.y_inputs = "eye"
        self.lo = np.full(len(FEATURES), -np.inf)  # allowed range of normalized inputs
        self.hi = np.full(len(FEATURES), np.inf)
        self.sweep_n = 0
        self.report: CalibrationReport | None = None

    # -- fitting -------------------------------------------------------------------------------
    def _normalize(self, F: np.ndarray) -> np.ndarray:
        return (F - self.center) / self.scale

    def fit(self, dots: list[tuple[tuple[float, float], np.ndarray]], y_inputs: str = "auto",
            sweep: list[tuple[tuple[float, float], np.ndarray]] | None = None) -> CalibrationReport:
        """dots: (target in screen pixels, feature rows (n, 6) collected while looking at it).
        y_inputs: "auto" picks the best of Y_INPUTS by leave-one-dot-out, or name one to force it.
        sweep: same shape, from looking at one dot while turning and nodding a little (short chunks).
        It teaches the model how head movement shifts the eye numbers; still dots alone can't."""
        if y_inputs != "auto" and y_inputs not in Y_INPUTS:
            raise ValueError(f"y_inputs must be 'auto' or one of {list(Y_INPUTS)}")
        dots = [(t, np.asarray(rows, dtype=np.float64)) for t, rows in dots if len(rows) > 0]
        if len(dots) < MIN_DOTS:
            raise ValueError(f"Need at least {MIN_DOTS} dots with data, got {len(dots)}")
        M = np.array([np.median(rows, axis=0) for _, rows in dots])  # one robust row per dot
        T = np.array([(t[0] / self.screen_w, t[1] / self.screen_h) for t, _ in dots])
        sweep = [(t, np.asarray(r, dtype=np.float64)) for t, r in (sweep or []) if len(r) > 0]
        Me = np.array([np.median(r, axis=0) for _, r in sweep]).reshape(-1, M.shape[1])
        Te = np.array([(t[0] / self.screen_w, t[1] / self.screen_h) for t, _ in sweep]).reshape(-1, 2)
        self.sweep_n = len(sweep)

        everything = np.vstack([M, Me])
        self.center = everything.mean(axis=0)
        spread = everything.std(axis=0)
        self.scale = np.where(spread < 1e-6, 1.0, spread)
        Z, Ze = self._normalize(M), self._normalize(Me)
        allz = np.vstack([Z, Ze])
        self.lo, self.hi = allz.min(axis=0) - CLIP_SD, allz.max(axis=0) + CLIP_SD

        self.alpha_x, loo_x, err_x = _best_alpha(_terms_x, Z, T[:, 0], self.screen_w, Ze, Te[:, 0])
        candidates = Y_INPUTS if y_inputs == "auto" else {y_inputs: Y_INPUTS[y_inputs]}
        y_results = {name: _best_alpha(terms, Z, T[:, 1], self.screen_h, Ze, Te[:, 1])
                     for name, terms in candidates.items()}
        self.y_inputs = min(y_results, key=lambda name: y_results[name][2])
        self.alpha_y, loo_y, err_y = y_results[self.y_inputs]

        terms_y = Y_INPUTS[self.y_inputs]
        Za, Ta = np.vstack([Z, Ze]), np.vstack([T, Te])
        self.px = _ridge_fit(_terms_x(Za), Ta[:, 0], self.alpha_x)
        self.py = _ridge_fit(terms_y(Za), Ta[:, 1], self.alpha_y)

        loo_err = np.hypot((loo_x - T[:, 0]) * self.screen_w, (loo_y - T[:, 1]) * self.screen_h)
        train_err = np.hypot((_ridge_predict(self.px, _terms_x(Z)) - T[:, 0]) * self.screen_w,
                             (_ridge_predict(self.py, terms_y(Z)) - T[:, 1]) * self.screen_h)
        self.report = CalibrationReport(
            loo_mean_px=float(loo_err.mean()),
            loo_median_px=float(np.median(loo_err)),
            loo_max_px=float(loo_err.max()),
            loo_x_px=err_x,
            loo_y_px=err_y,
            train_mean_px=float(train_err.mean()),
            y_inputs=self.y_inputs,
            sweep_n=self.sweep_n,
            y_compare={name: r[2] for name, r in y_results.items()},
            dots=[DotResult(target=(t[0], t[1]),
                            predicted=(float(px * self.screen_w), float(py * self.screen_h)),
                            error_px=float(e), samples=len(rows))
                  for (t, rows), px, py, e in zip(dots, loo_x, loo_y, loo_err)],
        )
        return self.report

    # -- using it ------------------------------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self.px is not None

    def predict_row(self, row: np.ndarray) -> tuple[float, float]:
        """Screen point in physical pixels, allowed 5% past each edge so the edges stay reachable."""
        z = np.clip(self._normalize(np.asarray(row, dtype=np.float64)[None, :]), self.lo, self.hi)
        nx = float(_ridge_predict(self.px, _terms_x(z))[0])
        ny = float(_ridge_predict(self.py, Y_INPUTS[self.y_inputs](z))[0])
        nx, ny = min(max(nx, -0.05), 1.05), min(max(ny, -0.05), 1.05)
        return nx * self.screen_w, ny * self.screen_h

    def predict(self, eyes: EyeFeatures, head: HeadPose | None) -> tuple[float, float]:
        return self.predict_row(feature_row(eyes, head))

    # -- saving --------------------------------------------------------------------------------
    def to_dict(self) -> dict:
        def params(p):
            return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in p.items()}
        return {"version": VERSION, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "features": list(FEATURES),
                "screen": [self.screen_w, self.screen_h], "center": self.center.tolist(),
                "scale": self.scale.tolist(), "lo": self.lo.tolist(), "hi": self.hi.tolist(), "y_inputs": self.y_inputs,
                "alpha_x": self.alpha_x, "alpha_y": self.alpha_y,
                "x": params(self.px), "y": params(self.py),
                "loo_mean_px": self.report.loo_mean_px if self.report else None}

    def save(self, path: Path | str | None = None) -> Path:
        path = Path(path) if path is not None else DEFAULT_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | str | None = None) -> "GazeModel":
        path = Path(path) if path is not None else DEFAULT_PATH
        d = json.loads(path.read_text(encoding="utf-8"))
        if d.get("version") != VERSION or d.get("features") != list(FEATURES):
            raise ValueError(f"{path} was made by an older version. Recalibrate: python -m tests.gaze_dot")
        m = cls(*d["screen"])
        m.center, m.scale = np.array(d["center"]), np.array(d["scale"])
        m.lo, m.hi = np.array(d["lo"]), np.array(d["hi"])
        m.y_inputs, m.alpha_x, m.alpha_y = d["y_inputs"], d["alpha_x"], d["alpha_y"]

        def params(p):
            return {"mu": np.array(p["mu"]), "sd": np.array(p["sd"]), "w": np.array(p["w"]), "ym": p["ym"]}
        m.px, m.py = params(d["x"]), params(d["y"])
        return m
