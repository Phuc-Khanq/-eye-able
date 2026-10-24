"""Head, hybrid or gaze pointing: one frame of eye features and head pose -> where the cursor should be.

Route: Khang and Hung (Khang leads)
Publishes: GazePoint (the cursor point, screen px) to the eyes service
Subscribes: EyeFeatures, HeadPose
APIs: HeadPointer (Hung's head cursor), PointSmoother (One Euro), GazeModel (calibration)

Modes:
  head    head angles move the cursor: about 15 degrees reaches a side edge, 10 degrees the top or bottom.
          The default. Steady and predictable; the eyes aren't used. Webcam gaze lands 100-200 px off on
          the demo PC, so UI-Venus does the precise part either way.
  hybrid  gaze jumps the cursor when it lands more than jump_px away; small head turns fine-tune it.
  gaze    gaze only (jump_px 0).
Blinks: the head pose is held through blink frames, because a blink can nudge MediaPipe's head angles.
"""
from engine.core.types import EyeFeatures, GazePoint, HeadPose
from engine.eyes.head_pointer import HeadPointer
from engine.eyes.smoothing import PointSmoother

MODES = ("head", "hybrid", "gaze")
JUMP_PX = 250.0   # hybrid: gaze this far from the cursor moves it; below that the head fine-tunes


class Pointer:
    def __init__(self, screen_w: int, screen_h: int, mode: str = "head", model=None,
                 jump_px: float = JUMP_PX) -> None:
        self.W, self.H = screen_w, screen_h
        self.model = model                      # GazeModel, or None (head only)
        self.jump_px = jump_px
        self.head = HeadPointer(screen_w, screen_h, jump_px=jump_px)
        self.smoother = PointSmoother(min_cutoff=1.0, beta=0.5)  # in screen fractions, like gaze_dot
        self.gaze_shift = (0.0, 0.0)            # px, set by the one-look calibration
        self._held: HeadPose | None = None
        self.mode = "head"
        if not self.set_mode(mode):
            raise ValueError(f"pointer '{mode}' needs a gaze model; calibrate first (python -m tests.gaze_dot)")

    def set_mode(self, mode: str) -> bool:
        """False (and no change) when the mode needs a gaze model and there is none."""
        if mode not in MODES:
            raise ValueError(f"pointer mode must be one of {MODES}")
        if mode != "head" and self.model is None:
            return False
        self.mode = mode
        self.head.jump_px = 0.0 if mode == "gaze" else self.jump_px
        self.reset()
        return True

    def reset(self) -> None:
        """After calibration or a mode change: no stale anchor or smoothing."""
        self.head.clear_anchor()
        self.smoother.reset()

    def raw_gaze(self, feats: EyeFeatures | None, pose: HeadPose) -> tuple[float, float] | None:
        """Where the gaze model says you look, before smoothing and before the calibration shift."""
        if self.model is None or feats is None or not feats.usable:
            return None
        return self.model.predict(feats, pose)

    def update(self, t_ms: int, feats: EyeFeatures | None, pose: HeadPose) -> GazePoint:
        usable = feats is None or feats.usable
        if usable or self._held is None:
            self._held = pose
        point_pose = pose if usable else HeadPose(pose.t_ms, self._held.yaw, self._held.pitch, self._held.roll)
        gaze = None
        if self.mode != "head":
            g = self.raw_gaze(feats, pose)
            if g is not None:
                gx, gy = g[0] + self.gaze_shift[0], g[1] + self.gaze_shift[1]
                sx, sy = self.smoother(t_ms / 1000.0, gx / self.W, gy / self.H)
                gaze = GazePoint(t_ms=t_ms, x=sx * self.W, y=sy * self.H, confidence=1.0)
        return self.head.update(point_pose, gaze)
