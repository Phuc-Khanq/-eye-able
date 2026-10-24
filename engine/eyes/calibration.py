"""Gaze calibration: features at each dot -> a small regression to screen x, y.

Route: Khang and Hung (Khang leads)
Publishes: calib_target, calib_result, the fitted gaze model
Subscribes: EyeFeatures, HeadPose, calibrate messages
APIs: NumPy (closed-form ridge regression). Saved as JSON in calibration/ (numbers only, no images;
      git ignores the folder).

How it works
  Features per frame: h, v, lid (eye_features) and yaw, pitch (head_pose, relative to the pose at
  the center dot). For each dot we keep the median of each feature over about a second.
  x is predicted from h, yaw, h^2 and h*v; y from v, lid, pitch, v^2 and h*v. Few terms on purpose:
  9 dots can't support a big model. Ridge regularization picks its strength by leave-one-dot-out.
  The leave-one-dot-out error is the honest number: how far off a spot is when the model never saw it.
"""
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from engine.core.types import EyeFeatures, HeadPose

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = REPO_ROOT / "calibration" / "gaze_model.json"
FEATURES = ("h", "v", "lid", "yaw", "pitch")
ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
MIN_DOTS = 6


def grid_targets(screen_w: int, screen_h: int, points: int = 9, margin: float = 0.1) -> list[tuple[float, float]]:
    """Dots on an n x n grid (9 -> 3x3, 16 -> 4x4) in screen pixels. The middle dot comes first
    (head pose is zeroed there), then the rest row by row in a snake so the eyes move little."""
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
    return np.array([eyes.h, eyes.v, eyes.lid, yaw, pitch], dtype=np.float64)


def _terms_x(z: np.ndarray) -> np.ndarray:
    h, v, _lid, yaw, _pitch = z.T
    return np.column_stack([h, yaw, h * h, h * v])


def _terms_y(z: np.ndarray) -> np.ndarray:
    h, v, lid, _yaw, pitch = z.T
    return np.column_stack([v, lid, pitch, v * v, h * v])


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


@dataclass
class DotResult:
    target: tuple[float, float]
    predicted: tuple[float, float]  # leave-one-dot-out prediction
    error_px: float
    samples: int


@dataclass
class CalibrationReport:
    alpha: float
    loo_mean_px: float
    loo_median_px: float
    loo_max_px: float
    train_mean_px: float
    dots: list[DotResult]

    def summary(self) -> str:
        return (f"leave-one-dot-out error: mean {self.loo_mean_px:.0f} px, median {self.loo_median_px:.0f} px, "
                f"worst {self.loo_max_px:.0f} px (fit error {self.train_mean_px:.0f} px, alpha {self.alpha:g}, "
                f"{len(self.dots)} dots)")


class GazeModel:
    """Maps a feature row (h, v, lid, yaw, pitch) to a screen point in physical pixels."""

    def __init__(self, screen_w: int, screen_h: int) -> None:
        self.screen_w = int(screen_w)
        self.screen_h = int(screen_h)
        self.center = np.zeros(len(FEATURES))
        self.scale = np.ones(len(FEATURES))
        self.px: dict | None = None
        self.py: dict | None = None
        self.alpha = 1.0
        self.report: CalibrationReport | None = None

    # -- fitting -------------------------------------------------------------------------------
    def _normalize(self, F: np.ndarray) -> np.ndarray:
        return (F - self.center) / self.scale

    def _fit_on(self, Z: np.ndarray, T: np.ndarray, alpha: float) -> tuple[dict, dict]:
        return _ridge_fit(_terms_x(Z), T[:, 0], alpha), _ridge_fit(_terms_y(Z), T[:, 1], alpha)

    def _loo(self, Z: np.ndarray, T: np.ndarray, alpha: float) -> np.ndarray:
        preds = np.zeros_like(T)
        for i in range(len(Z)):
            keep = np.arange(len(Z)) != i
            px, py = self._fit_on(Z[keep], T[keep], alpha)
            preds[i, 0] = _ridge_predict(px, _terms_x(Z[i:i + 1]))[0]
            preds[i, 1] = _ridge_predict(py, _terms_y(Z[i:i + 1]))[0]
        return preds

    def _errors_px(self, preds: np.ndarray, T: np.ndarray) -> np.ndarray:
        return np.hypot((preds[:, 0] - T[:, 0]) * self.screen_w, (preds[:, 1] - T[:, 1]) * self.screen_h)

    def fit(self, dots: list[tuple[tuple[float, float], np.ndarray]]) -> CalibrationReport:
        """dots: (target in screen pixels, feature rows (n, 5) collected while looking at it)."""
        dots = [(t, np.asarray(rows, dtype=np.float64)) for t, rows in dots if len(rows) > 0]
        if len(dots) < MIN_DOTS:
            raise ValueError(f"Need at least {MIN_DOTS} dots with data, got {len(dots)}")
        M = np.array([np.median(rows, axis=0) for _, rows in dots])  # one robust row per dot
        T = np.array([(t[0] / self.screen_w, t[1] / self.screen_h) for t, _ in dots])

        self.center = M.mean(axis=0)
        spread = M.std(axis=0)
        self.scale = np.where(spread < 1e-6, 1.0, spread)
        Z = self._normalize(M)

        best = None
        for alpha in ALPHAS:
            err = self._errors_px(self._loo(Z, T, alpha), T)
            if best is None or err.mean() < best[1].mean():
                best = (alpha, err)
        self.alpha, loo_err = best
        loo_pred = self._loo(Z, T, self.alpha)
        self.px, self.py = self._fit_on(Z, T, self.alpha)
        train = np.column_stack([_ridge_predict(self.px, _terms_x(Z)), _ridge_predict(self.py, _terms_y(Z))])
        train_err = self._errors_px(train, T)

        self.report = CalibrationReport(
            alpha=self.alpha,
            loo_mean_px=float(loo_err.mean()),
            loo_median_px=float(np.median(loo_err)),
            loo_max_px=float(loo_err.max()),
            train_mean_px=float(train_err.mean()),
            dots=[DotResult(target=(t[0], t[1]),
                            predicted=(float(p[0] * self.screen_w), float(p[1] * self.screen_h)),
                            error_px=float(e), samples=len(rows))
                  for (t, rows), p, e in zip(dots, loo_pred, loo_err)],
        )
        return self.report

    # -- using it ------------------------------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self.px is not None

    def predict_row(self, row: np.ndarray) -> tuple[float, float]:
        """Screen point in physical pixels, allowed 5% past each edge so the edges stay reachable."""
        z = self._normalize(np.asarray(row, dtype=np.float64)[None, :])
        nx = float(_ridge_predict(self.px, _terms_x(z))[0])
        ny = float(_ridge_predict(self.py, _terms_y(z))[0])
        nx, ny = min(max(nx, -0.05), 1.05), min(max(ny, -0.05), 1.05)
        return nx * self.screen_w, ny * self.screen_h

    def predict(self, eyes: EyeFeatures, head: HeadPose | None) -> tuple[float, float]:
        return self.predict_row(feature_row(eyes, head))

    # -- saving --------------------------------------------------------------------------------
    def to_dict(self) -> dict:
        def params(p):
            return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in p.items()}
        return {"version": 1, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "features": list(FEATURES),
                "screen": [self.screen_w, self.screen_h], "center": self.center.tolist(),
                "scale": self.scale.tolist(), "alpha": self.alpha,
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
        m = cls(*d["screen"])
        m.center, m.scale, m.alpha = np.array(d["center"]), np.array(d["scale"]), d["alpha"]

        def params(p):
            return {"mu": np.array(p["mu"]), "sd": np.array(p["sd"]), "w": np.array(p["w"]), "ym": p["ym"]}
        m.px, m.py = params(d["x"]), params(d["y"])
        return m
