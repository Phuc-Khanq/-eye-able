"""Head-angle cursor: gaze jumps, head fine-tunes.

Route: Khang and Hung (Hung leads)
Publishes: GazePoint
Subscribes: HeadPose, GazePoint (smoothed gaze)
APIs: NumPy

Gaze is fast but jittery; the head is slow but steady. So:
  - When the gaze point lands more than JUMP_PX from the cursor, the cursor jumps there. For the
    next SETTLE_MS it keeps following gaze, so a smoothed gaze that is still sliding toward the
    target doesn't leave the cursor short. Then the current head pose becomes the anchor.
  - Between jumps, small head turns move the cursor from that spot, FINE_PX_PER_DEG per degree.
  - With no gaze at all (not calibrated, or gaze switched off), the head alone drives the cursor:
    the neutral pose is the screen center, and FULL_YAW_DEG / FULL_PITCH_DEG reach the edges.
Sensitivity from the settings scales head motion in both modes. Head angles pass through a light
One Euro filter first (the filter itself is Khang's, in smoothing.py), because 0.4 degrees of
tracking noise would otherwise shake the cursor by 8 px at FINE_PX_PER_DEG. Gaze smoothing stays
in the caller.
"""
import numpy as np

from engine.config import settings
from engine.core.types import GazePoint, HeadPose
from engine.eyes.smoothing import OneEuroFilter

JUMP_PX = 150            # gaze this far from the cursor moves it in one jump
SETTLE_MS = 300          # after a jump, follow gaze this long before head fine-tuning takes over
FINE_PX_PER_DEG = 20.0   # head fine-tune speed after a jump
FULL_YAW_DEG = 15.0      # head-only mode: turn that reaches a side edge
FULL_PITCH_DEG = 10.0    # head-only mode: tilt up or down that reaches the top or bottom edge
FILTER_MIN_CUTOFF = 0.8  # Hz; lower = steadier at rest, more lag
FILTER_BETA = 0.05       # higher = less lag during fast head motion


class HeadPointer:
    def __init__(self, screen_w: int, screen_h: int, jump_px: float = JUMP_PX,
                 fine_px_per_deg: float = FINE_PX_PER_DEG, settle_ms: int = SETTLE_MS) -> None:
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.jump_px = jump_px
        self.settle_ms = settle_ms
        self.jump_ms: int | None = None
        self.fine_px_per_deg = fine_px_per_deg
        self.anchor_xy: tuple[float, float] | None = None
        self.anchor_pose: HeadPose | None = None
        self.cursor: tuple[float, float] = ((screen_w - 1) / 2, (screen_h - 1) / 2)
        self._yaw_filter = OneEuroFilter(FILTER_MIN_CUTOFF, FILTER_BETA)
        self._pitch_filter = OneEuroFilter(FILTER_MIN_CUTOFF, FILTER_BETA)

    def _clamp(self, x: float, y: float) -> tuple[float, float]:
        return float(np.clip(x, 0, self.screen_w - 1)), float(np.clip(y, 0, self.screen_h - 1))

    def jump_to(self, x: float, y: float, pose: HeadPose) -> None:
        """Put the cursor at (x, y) and fine-tune from the current head pose."""
        self.anchor_xy = self._clamp(x, y)
        self.anchor_pose = pose
        self.cursor = self.anchor_xy

    def clear_anchor(self) -> None:
        """Back to head-only mode."""
        self.anchor_xy = None
        self.anchor_pose = None
        self.jump_ms = None
        self._yaw_filter.reset()
        self._pitch_filter.reset()

    def to_cursor(self, pose: HeadPose) -> tuple[float, float]:
        sens = max(settings.sensitivity, 0.1)
        if self.anchor_xy is None:
            nx = np.clip(pose.yaw * sens / FULL_YAW_DEG, -1.0, 1.0)      # -1 left edge, 1 right edge
            ny = np.clip(pose.pitch * sens / FULL_PITCH_DEG, -1.0, 1.0)  # chin down moves the cursor down
            self.cursor = ((nx + 1.0) / 2.0 * (self.screen_w - 1), (ny + 1.0) / 2.0 * (self.screen_h - 1))
        else:
            k = self.fine_px_per_deg * sens
            ax, ay = self.anchor_xy
            self.cursor = self._clamp(ax + (pose.yaw - self.anchor_pose.yaw) * k,
                                      ay + (pose.pitch - self.anchor_pose.pitch) * k)
        return self.cursor

    def update(self, pose: HeadPose, gaze: GazePoint | None = None) -> GazePoint:
        """Feed every frame. Pass the smoothed gaze when there is one; returns the cursor point."""
        t_s = pose.t_ms / 1000.0
        pose = HeadPose(t_ms=pose.t_ms, yaw=self._yaw_filter(t_s, pose.yaw),
                        pitch=self._pitch_filter(t_s, pose.pitch), roll=pose.roll)
        if gaze is not None and np.isfinite(gaze.x) and np.isfinite(gaze.y):
            cx, cy = self.cursor
            settling = self.jump_ms is not None and pose.t_ms - self.jump_ms < self.settle_ms
            if self.anchor_xy is None or np.hypot(gaze.x - cx, gaze.y - cy) > self.jump_px:
                self.jump_to(gaze.x, gaze.y, pose)
                self.jump_ms = pose.t_ms
            elif settling:
                self.jump_to(gaze.x, gaze.y, pose)
        x, y = self.to_cursor(pose)
        return GazePoint(t_ms=pose.t_ms, x=x, y=y, confidence=gaze.confidence if gaze is not None else 1.0)


def to_cursor(pose: HeadPose) -> tuple[float, float]:
    """One-off head-only helper on the primary screen. Keep a HeadPointer in the live loop."""
    from engine.eyes.os_input import screen_size

    return HeadPointer(*screen_size()).to_cursor(pose)
