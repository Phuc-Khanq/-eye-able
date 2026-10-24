"""Head, hybrid or gaze pointing: one frame of eye features and head pose -> where the cursor should be.

Route: Khang and Hung (Khang leads)
Publishes: GazePoint (the cursor point, screen px) to the eyes service
Subscribes: EyeFeatures, HeadPose
APIs: HeadPointer (Hung's head cursor), PointSmoother (One Euro), GazeModel (calibration)

Modes:
  head      the default: the head moves the cursor like a mouse with acceleration. Turning the head moves
            the cursor by gain(speed) pixels per degree: a quick turn throws it far (a quick 10-degree turn
            moves about 400 px), a slow turn moves it precisely (10 px per degree). There is
            no fixed center, so leaning or shifting in the chair doesn't drift it: like lifting a mouse,
            a slow turn back re-centers the head without moving the cursor much.
  head-abs  head angles map to fixed spots: about 15 degrees reaches a side edge, 10 the top or bottom.
            Predictable, but reaching edges is tiring and the center drifts as posture changes.
  hybrid  gaze jumps the cursor when it lands more than jump_px away; small head turns fine-tune it.
  gaze    gaze only (jump_px 0).
Blinks: the head pose is held through blink frames (a blink can nudge MediaPipe's head angles), but only for
  BLINK_MAX_MS: looking down lowers the eyelids and reads as a long 'blink', and holding through that made the
  cursor freeze and then jump. In head mode the movement restarts from the current angle after a hold, so the
  angle change during the hold is dropped instead of applied in one frame.
"""
import math

from engine.config import settings
from engine.core.types import EyeFeatures, GazePoint, HeadPose
from engine.eyes.head_pointer import HeadPointer
from engine.eyes.smoothing import OneEuroFilter, PointSmoother

MODES = ("head", "head-abs", "hybrid", "gaze")
JUMP_PX = 250.0   # hybrid: gaze this far from the cursor moves it; below that the head fine-tunes

# head (relative) mode. Speeds are of the head, in degrees per second.
GAIN_SLOW = 10.0        # px per degree when turning slowly: precise
GAIN_FAST = 50.0        # px per degree when turning fast: a quick 10-degree turn moves about 400 px
SPEED_SLOW = 8.0        # at or under this speed the gain is GAIN_SLOW (most aiming happens here)...
SPEED_FAST = 60.0       # ...at or over this one it's GAIN_FAST, smoothly in between (gentle acceleration)
MAX_STEP_PX = 90.0      # the cursor moves at most this far per camera frame (about 2700 px/s): no single-frame jumps
BLINK_MAX_MS = 300      # a 'blink' longer than this is the eyelids dropping as you look down: stop holding
GLITCH_FRAMES = 2       # this many wild frames in a row is a real fast turn: restart from the new angle
VERTICAL_BOOST = 1.4    # people tilt up and down less easily than they turn
MAX_STEP_DEG = 10.0     # a bigger raw jump between two frames is a tracking glitch, not a movement
SPEED_ATTACK = 0.5      # how fast the speed estimate rises (0-1 per frame)...
SPEED_RELEASE = 0.2     # ...and falls; slower falling keeps the gain steady through a movement
ANGLE_MIN_CUTOFF = 1.0  # One Euro on the head angles: steadiness at rest (Hz)...
ANGLE_BETA = 0.3        # ...and how quickly it stops lagging when the head moves

# Gaze (hybrid and gaze modes), One Euro in screen fractions. Tuned on 195 real calibration dots (9 sittings):
# wobble while looking at one spot 8.1 -> 5.8 px, and a jump to a new spot is followed in 0.07 s, not 0.13 s.
GAZE_MIN_CUTOFF = 0.3
GAZE_BETA = 4.0

# The cursor's dead zone (CursorDriver) per mode: wobble smaller than this doesn't move it.
# The head mode needs almost none (slow moves are only about 12 px per degree, and a bigger dead zone
# makes slow moves sticky); the absolute head mode and gaze do.
STILL_PX = {"head": 2.0, "head-abs": 8.0, "hybrid": 20.0, "gaze": 20.0}


class HeadRelPointer:
    """Head as a mouse with acceleration (the 'head' mode)."""

    def __init__(self, screen_w: int, screen_h: int) -> None:
        self.W, self.H = screen_w, screen_h
        self._fy = OneEuroFilter(ANGLE_MIN_CUTOFF, ANGLE_BETA)
        self._fp = OneEuroFilter(ANGLE_MIN_CUTOFF, ANGLE_BETA)
        self._last: tuple[int, float, float] | None = None
        self._raw: tuple[float, float] | None = None
        self._wild = 0
        self.speed = 0.0
        self.center()

    def center(self) -> None:
        """Cursor to the screen center; the next frame starts a new relative movement."""
        self.cursor = ((self.W - 1) / 2, (self.H - 1) / 2)
        self.rebase()

    def rebase(self) -> None:
        """Keep the cursor where it is; the next frame is the new starting angle (no movement from it)."""
        self._last = None
        self._raw = None
        self._wild = 0
        self.speed = 0.0
        self._fy.reset()
        self._fp.reset()

    @staticmethod
    def gain(speed: float) -> float:
        k = min(max((speed - SPEED_SLOW) / (SPEED_FAST - SPEED_SLOW), 0.0), 1.0)
        k = k * k * (3 - 2 * k)   # smoothstep: no sudden jump in gain
        return GAIN_SLOW + (GAIN_FAST - GAIN_SLOW) * k

    def update(self, pose: HeadPose) -> GazePoint:
        if self._raw is not None and math.hypot(pose.yaw - self._raw[0], pose.pitch - self._raw[1]) > MAX_STEP_DEG:
            self._wild += 1
            if self._wild < GLITCH_FRAMES:
                # one wild frame: skip it (the next frame is compared with the last good one)
                return GazePoint(t_ms=pose.t_ms, x=self.cursor[0], y=self.cursor[1], confidence=1.0)
            self.rebase()   # still far away: the head really is there now; carry on from it without a jump
        self._wild = 0
        self._raw = (pose.yaw, pose.pitch)
        t = pose.t_ms / 1000.0
        yaw, pitch = self._fy(t, pose.yaw), self._fp(t, pose.pitch)
        if self._last is not None:
            dt = max((pose.t_ms - self._last[0]) / 1000.0, 1e-3)
            dyaw, dpitch = yaw - self._last[1], pitch - self._last[2]
            if dt < 0.5:
                inst = math.hypot(dyaw, dpitch) / dt
                rate = SPEED_ATTACK if inst > self.speed else SPEED_RELEASE
                self.speed += rate * (inst - self.speed)
                g = self.gain(self.speed) * max(settings.sensitivity, 0.1)
                mx, my = dyaw * g, dpitch * g * VERTICAL_BOOST   # chin down = cursor down
                step = math.hypot(mx, my)
                if step > MAX_STEP_PX:
                    mx, my = mx * MAX_STEP_PX / step, my * MAX_STEP_PX / step
                x, y = self.cursor[0] + mx, self.cursor[1] + my
                self.cursor = (min(max(x, 0.0), self.W - 1.0), min(max(y, 0.0), self.H - 1.0))
        self._last = (pose.t_ms, yaw, pitch)
        return GazePoint(t_ms=pose.t_ms, x=self.cursor[0], y=self.cursor[1], confidence=1.0)


class Pointer:
    def __init__(self, screen_w: int, screen_h: int, mode: str = "head", model=None,
                 jump_px: float = JUMP_PX) -> None:
        self.W, self.H = screen_w, screen_h
        self.model = model                      # GazeModel, or None (head only)
        self.jump_px = jump_px
        self.head = HeadPointer(screen_w, screen_h, jump_px=jump_px)
        self.rel = HeadRelPointer(screen_w, screen_h)
        self.smoother = PointSmoother(min_cutoff=GAZE_MIN_CUTOFF, beta=GAZE_BETA)  # in screen fractions
        self.gaze_shift = (0.0, 0.0)            # px, set by the one-look calibration
        self._held: HeadPose | None = None
        self._hold_since: int | None = None     # t_ms the eyes stopped being usable
        self._was_holding = False
        self.mode = "head"
        if not self.set_mode(mode):
            raise ValueError(f"pointer '{mode}' needs a gaze model; calibrate first (python -m tests.gaze_dot)")

    def set_mode(self, mode: str) -> bool:
        """False (and no change) when the mode needs a gaze model and there is none."""
        if mode not in MODES:
            raise ValueError(f"pointer mode must be one of {MODES}")
        if mode in ("hybrid", "gaze") and self.model is None:
            return False
        self.mode = mode
        self.head.jump_px = 0.0 if mode == "gaze" else self.jump_px
        self.reset()
        return True

    @property
    def still_px(self) -> float:
        """The dead zone the cursor should use in this mode (CursorDriver.stable.still_px)."""
        return STILL_PX[self.mode]

    def reset(self) -> None:
        """After calibration or a mode change: no stale anchor or smoothing; the head mode re-centers."""
        self.head.clear_anchor()
        self.smoother.reset()
        self.rel.center()
        self._hold_since = None
        self._was_holding = False

    def place(self, x: float, y: float) -> None:
        """Head mode: continue from (x, y), e.g. the button UI-Venus just found. Other modes: nothing."""
        if self.mode == "head":
            self.rel.cursor = (float(x), float(y))
            self.rel.rebase()

    def raw_gaze(self, feats: EyeFeatures | None, pose: HeadPose) -> tuple[float, float] | None:
        """Where the gaze model says you look, before smoothing and before the calibration shift."""
        if self.model is None or feats is None or not feats.usable:
            return None
        return self.model.predict(feats, pose)

    def update(self, t_ms: int, feats: EyeFeatures | None, pose: HeadPose) -> GazePoint:
        usable = feats is None or feats.usable
        if not usable and self._hold_since is None:
            self._hold_since = t_ms
        holding = (not usable and self._held is not None and t_ms - self._hold_since <= BLINK_MAX_MS)
        if not holding:
            if self._was_holding and self.mode == "head":
                self.rel.rebase()     # restart from the current angle: the change during the hold isn't applied
            if usable:
                self._hold_since = None
            self._held = pose
        self._was_holding = holding
        if self.mode == "head":
            if holding:
                return GazePoint(t_ms=t_ms, x=self.rel.cursor[0], y=self.rel.cursor[1], confidence=1.0)
            return self.rel.update(pose)
        point_pose = pose if not holding else HeadPose(pose.t_ms, self._held.yaw, self._held.pitch, self._held.roll)
        gaze = None
        if self.mode in ("hybrid", "gaze"):
            g = self.raw_gaze(feats, pose)
            if g is not None:
                gx, gy = g[0] + self.gaze_shift[0], g[1] + self.gaze_shift[1]
                sx, sy = self.smoother(t_ms / 1000.0, gx / self.W, gy / self.H)
                gaze = GazePoint(t_ms=t_ms, x=sx * self.W, y=sy * self.H, confidence=1.0)
        return self.head.update(point_pose, gaze)
