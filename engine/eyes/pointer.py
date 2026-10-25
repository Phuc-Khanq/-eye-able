"""Head, hybrid or gaze pointing: one frame of eye features and head pose -> where the cursor should be.

Route: Khang and Hung (Khang leads)
Publishes: GazePoint (the cursor point, screen px) to the eyes service
Subscribes: EyeFeatures, HeadPose
APIs: HeadPointer (Hung's head cursor), PointSmoother (One Euro), GazeModel (calibration)

Modes:
  head-nose the head mode, plus the nose as a second reading of the head's turn: in the outer part of a turn it carries
            the cursor on to the edge that turn points at (see NOSE_* below). For a cursor that stops short of an edge.
  head-eyes the default when there is a gaze calibration: the head mode below, plus the eyes. While the head turns
            toward where you look, the cursor is also pulled toward that spot, so a smaller head turn gets there;
            the pull stops at the edge of the gaze's error ellipse around that spot, and the last stretch is the
            head's own fine control. Resting the head, glancing or reading never moves the cursor (see EYE_* below).
            Webcam gaze can't be the precise part: on Khang's calibration it lands about 170 px from where he looks
            (median; about 100 px sideways, 135 up or down), while the head pointer is steady to a pixel or two.
  head      the head moves the cursor like a mouse with acceleration (no eyes). Turning the head moves
            the cursor by gain(speed) pixels per degree: a quick turn throws it far (a quick 10-degree turn
            moves about 400 px), a slow turn moves it precisely (10 px per degree). There is
            no fixed center, so leaning or shifting in the chair doesn't drift it: like lifting a mouse,
            a slow turn back re-centers the head without moving the cursor much.
  head-push the head mode, but only a turn AWAY from your resting head angle (home) moves the cursor. Turning back
            toward home leaves the cursor where it is, like lifting a mouse to put it back in the middle of the pad:
            push right, come back to the middle, push right again to go further. To go the other way, turn past
            home to that side. Home is the angle the head had when the pointer started or "recenter" was said, and
            it follows a resting head very slowly (HOME_FOLLOW_S) so a change of posture doesn't build up.
            (Oct 25, Khang: "push the cursor left or right, recenter, push it where you want".)
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
from collections import deque
from dataclasses import dataclass

from engine.config import settings
from engine.core.types import EyeFeatures, GazePoint, HeadPose
from engine.eyes.head_pointer import HeadPointer
from engine.eyes.smoothing import OneEuroFilter, PointSmoother

MODES = ("head", "head-eyes", "head-nose", "head-push", "head-abs", "hybrid", "gaze")
REL_MODES = ("head", "head-eyes", "head-nose", "head-push")      # the relative "mouse" pointer is in charge
NEEDS_GAZE = ("head-eyes", "hybrid", "gaze")
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
# The per-frame limits above (MAX_STEP_PX, MAX_STEP_DEG, SPEED_ATTACK/RELEASE) are for a camera at 25 fps or
# more. A slower frame (a dropped one, or Hung's webcam at 8 fps in dim light) gets them scaled by how much
# longer it was, up to MAX_FRAME_SCALE: before, a 60 deg/s turn at 8 fps lost two thirds of its distance and
# anything faster was taken for a glitch, so the cursor didn't move at all. At 25+ fps nothing changes.
FRAME_S = 0.04
MAX_FRAME_SCALE = 4.0
# Edge throws: a step over MAX_STEP_PX that would have gone past a screen edge isn't cut any more; the rest
# carries over to the next frames (still at most MAX_STEP_PX each) until the cursor is at that edge. Before,
# throwing the head faster moved the cursor LESS: a 25-degree turn reached the left edge at 60 deg/s but
# stopped halfway at 150 deg/s. Throws that end on the screen are cut as before, so the middle feels the same.
# Turning back drops what's left. (No assist for slow moves: icons 20-60 px from an edge must stay reachable.)
MAX_CARRY_PX = 1500.0
# A comfortable turn (tests.pointer_feel edge test, Hung's laptop, 32 fps, 2026-10-24): the head moved over about
# 38 degrees of yaw and 23 of pitch (5th..95th percentile), so about 19 degrees to a side and 11 up or down from
# the middle. All four edges were reached in 0.3-2.8 s. Replaying that log through the old code missed the right edge.
REVERSE_DEG = 0.3       # a head move against the throw bigger than this per frame (9 deg/s; not tracking noise)
# Rest gate (Oct 24 night, "it jitters more than it used to"): a head at rest still wobbles (MediaPipe pitch about
# 0.16 degrees frame to frame, yaw about 0.07), and in this relative mode every wobble moved the cursor: it crept about
# 2 px a second and hopped twice a second. Each axis now moves only once its filtered angle has changed more than
# the low value over the last REST_WINDOW_MS, fully above the high one (smoothly in between). Replaying Khang's two
# pointer_feel logs (3,600 frames): resting creep 2.3 -> 0 px/s, hops 6% -> 0% of resting frames, every move the
# same length, slow aiming about 8% shorter. Per axis, so a sideways move no longer carries the pitch wobble.
REST_WINDOW_MS = 150
REST_YAW_DEG = (0.08, 0.20)
REST_PITCH_DEG = (0.12, 0.30)   # pitch wobbles about twice as much as yaw

# Gaze (hybrid and gaze modes), One Euro in screen fractions. Tuned on 195 real calibration dots (9 sittings):
# wobble while looking at one spot 8.1 -> 5.8 px, and a jump to a new spot is followed in 0.07 s, not 0.13 s.
GAZE_MIN_CUTOFF = 0.3
GAZE_BETA = 4.0

# The cursor's dead zone (CursorDriver) per mode: wobble smaller than this doesn't move it.
# The head mode needs almost none (slow moves are only about 12 px per degree, and a bigger dead zone
# makes slow moves sticky); the absolute head mode and gaze do.
STILL_PX = {"head": 2.0, "head-eyes": 2.0, "head-nose": 2.0, "head-push": 2.0, "head-abs": 8.0, "hybrid": 20.0,
            "gaze": 20.0}

# head-push: only a turn away from home moves the cursor (see the docstring).
HOME_FOLLOW_S = 20.0    # while the head rests, home drifts toward its angle with this time constant: slow enough that
                        # holding a push for a few seconds doesn't make the way back count, quick enough that a new
                        # sitting posture becomes the new home within a minute ("recenter" sets it at once)

# head-eyes: how the eyes pull (Oct 24 night, Khang: "head as the macro control, the iris as an add-on")
EYE_RADIUS_PX = (110.0, 140.0)  # the gaze's error ellipse (x, y) around where the model says you look: no pull inside
                                # it (Khang's calibration, leave-one-session-out: 98 px off sideways, 135 up or down)
EYE_SPEED_DEG = (4.0, 12.0)     # head speed, deg/s: no pull at or under the low one (resting, fine aiming), full pull
                                # from the high one, smoothly in between: the head says "go", the eyes say where
EYE_ALIGN = (0.3, 0.8)          # cosine of the angle between the head's move and the way to the gaze spot: no pull
                                # under the low one (the head goes elsewhere), full pull over the high one
EYE_PULL_RATE = 5.0             # per second: while it pulls, the gap beyond the ellipse closes like 1 - exp(-5 t)
EYE_MAX_STEP_PX = 60.0          # at most this much pull per camera frame (25+ fps; scaled for slower frames)


# head-nose (Oct 24 night, Khang: "after a while it doesn't reach the far left or right"): the head pointer is
# relative, so errors pile up and the cursor can stop short of an edge. The nose (engine/eyes/nose.py) gives the head's
# turn a second time. In the outer part of a turn it carries the cursor to the screen spot that turn points at, and
# only forward (never back toward the middle), so the middle of the screen feels exactly like "head".
NOSE_EDGE_DEG = (19.0, 11.0)    # a turn that reaches a side edge (yaw), the top or bottom (pitch): the comfortable
                                # range measured in tests.pointer_feel (Hung's laptop, Oct 24)
NOSE_ZONE = 0.7                 # the assist starts at this share of that turn, full at the whole of it
NOSE_PULL_RATE = 5.0            # per second, like the eye pull: the gap to the edge closes like 1 - exp(-5 t)
NOSE_MAX_STEP_PX = 45.0         # at most this much assist per camera frame (25+ fps; scaled for slower frames)


def _away(before: float, after: float) -> float:
    """head-push, one axis, angles measured from home: the part of a move from `before` to `after` that goes away
    from home. A move back toward home counts as nothing; a move that crosses home counts only from home on."""
    if before * after < 0:
        return after                     # crossed home: the stretch past it is a push the other way
    return after - before if abs(after) > abs(before) else 0.0


def _ramp(x: float, lo: float, hi: float) -> float:
    """0 at or under lo, 1 at or over hi, smooth (smoothstep) in between."""
    k = min(max((x - lo) / (hi - lo), 0.0), 1.0)
    return k * k * (3 - 2 * k)


class HeadRelPointer:
    """Head as a mouse with acceleration (the 'head' mode)."""

    def __init__(self, screen_w: int, screen_h: int) -> None:
        self.W, self.H = screen_w, screen_h
        self._fy = OneEuroFilter(ANGLE_MIN_CUTOFF, ANGLE_BETA)
        self._fp = OneEuroFilter(ANGLE_MIN_CUTOFF, ANGLE_BETA)
        self._last: tuple[int, float, float] | None = None
        self._raw: tuple[float, float] | None = None
        self._wild = 0
        self._carry = (0.0, 0.0)
        self._recent: deque[tuple[int, float, float]] = deque()   # t_ms, filtered yaw and pitch (the rest gate)
        self.speed = 0.0
        self.step = (0.0, 0.0)     # how far the cursor moved on the last frame (head-eyes: which way the head goes)
        self.rest_gate = True
        self.push = False          # head-push: only turns away from home move the cursor
        self.home: tuple[float, float] | None = None   # head-push: resting yaw and pitch (filtered), degrees
        self.stats = PointerStats()
        self.center()

    def center(self) -> None:
        """Cursor to the screen center; the next frame starts a new relative movement."""
        self.cursor = ((self.W - 1) / 2, (self.H - 1) / 2)
        self.home = None           # head-push: the next frame's angle is the new home
        self.rebase()

    def rebase(self) -> None:
        """Keep the cursor where it is; the next frame is the new starting angle (no movement from it)."""
        self._last = None
        self._raw = None
        self._wild = 0
        self._carry = (0.0, 0.0)
        self._recent.clear()
        self.speed = 0.0
        self.step = (0.0, 0.0)
        self._fy.reset()
        self._fp.reset()

    @staticmethod
    def gain(speed: float) -> float:
        k = min(max((speed - SPEED_SLOW) / (SPEED_FAST - SPEED_SLOW), 0.0), 1.0)
        k = k * k * (3 - 2 * k)   # smoothstep: no sudden jump in gain
        return GAIN_SLOW + (GAIN_FAST - GAIN_SLOW) * k

    def gate(self, t_ms: int, yaw: float, pitch: float) -> tuple[float, float]:
        """(x, y) multipliers 0-1: 0 for an axis whose filtered angle has barely changed over REST_WINDOW_MS (the
        head is resting; that change is wobble), 1 once it really moves."""
        self._recent.append((t_ms, yaw, pitch))
        while len(self._recent) > 2 and t_ms - self._recent[1][0] >= REST_WINDOW_MS:
            self._recent.popleft()
        if not self.rest_gate:
            return 1.0, 1.0
        _, y0, p0 = self._recent[0]
        return _ramp(abs(yaw - y0), *REST_YAW_DEG), _ramp(abs(pitch - p0), *REST_PITCH_DEG)

    @staticmethod
    def frame_scale(dt: float) -> float:
        """1 for a frame of a 25+ fps camera; more for a longer frame (see FRAME_S)."""
        return min(max(dt / FRAME_S, 1.0), MAX_FRAME_SCALE)

    def update(self, pose: HeadPose) -> GazePoint:
        self.step = (0.0, 0.0)
        if self._raw is not None:
            jump = math.hypot(pose.yaw - self._raw[0], pose.pitch - self._raw[1])
            if jump > MAX_STEP_DEG * self.frame_scale((pose.t_ms - self._raw[2]) / 1000.0):
                self._wild += 1
                if self._wild < GLITCH_FRAMES:
                    # one wild frame: skip it (the next frame is compared with the last good one)
                    self.stats.glitch_skips += 1
                    return GazePoint(t_ms=pose.t_ms, x=self.cursor[0], y=self.cursor[1], confidence=1.0)
                self.stats.glitch_rebases += 1
                self.stats.glitch_dropped_deg += jump
                self.rebase()   # still far away: the head really is there now; carry on from it without a jump
        self._wild = 0
        self._raw = (pose.yaw, pose.pitch, pose.t_ms)
        t = pose.t_ms / 1000.0
        yaw, pitch = self._fy(t, pose.yaw), self._fp(t, pose.pitch)
        if self.push and self.home is None:
            self.home = (yaw, pitch)
        if self._last is not None:
            dt = max((pose.t_ms - self._last[0]) / 1000.0, 1e-3)
            dyaw, dpitch = yaw - self._last[1], pitch - self._last[2]
            if self.push:                                    # only the part of the move that goes away from home
                hy, hp = self.home
                pyaw = _away(self._last[1] - hy, yaw - hy)
                ppitch = _away(self._last[2] - hp, pitch - hp)
            else:
                pyaw, ppitch = dyaw, dpitch
            if dt < 0.5:
                k = self.frame_scale(dt)
                inst = math.hypot(dyaw, dpitch) / dt
                rate = SPEED_ATTACK if inst > self.speed else SPEED_RELEASE
                rate = 1.0 - (1.0 - rate) ** k              # the same smoothing per second at any frame rate
                self.speed += rate * (inst - self.speed)
                g = self.gain(self.speed) * max(settings.sensitivity, 0.1)
                kx, ky = self.gate(pose.t_ms, yaw, pitch)
                mx, my = pyaw * g * kx, ppitch * g * VERTICAL_BOOST * ky   # chin down = cursor down
                rx, ry = dyaw * g * kx, dpitch * g * VERTICAL_BOOST * ky   # the head's own move (= mx, my unless head-push)
                cx, cy = self._carry
                if rx * cx + ry * cy < 0 and math.hypot(dyaw, dpitch) > REVERSE_DEG * k:
                    cx = cy = 0.0                            # turning back: forget the throw's cut part
                step, limit = math.hypot(mx, my), MAX_STEP_PX * k
                if step > limit:                             # cut to the limit, as always; remember the cut part
                    self.stats.capped_px += step - limit
                    cx, cy = cx + mx * (1 - limit / step), cy + my * (1 - limit / step)
                    mx, my = mx * limit / step, my * limit / step
                    left = math.hypot(cx, cy)
                    if left > MAX_CARRY_PX:
                        cx, cy = cx * MAX_CARRY_PX / left, cy * MAX_CARRY_PX / left
                # The cut part is used only where it reaches past an edge: it finishes the throw to that edge,
                # in steps that keep within the limit.
                ex, ey = self.cursor[0] + mx + cx, self.cursor[1] + my + cy
                ux = cx if not 0.0 <= ex <= self.W - 1.0 else 0.0
                uy = cy if not 0.0 <= ey <= self.H - 1.0 else 0.0
                u, room = math.hypot(ux, uy), limit - math.hypot(mx, my)
                if u > 0 and room > 0:
                    f = min(1.0, room / u)
                    mx, my, cx, cy = mx + ux * f, my + uy * f, cx - ux * f, cy - uy * f
                    self.stats.edge_px += u * f
                if self.speed < SPEED_SLOW:                  # the throw is over: a cut part that ends on the
                    cx, cy = (cx if ux else 0.0), (cy if uy else 0.0)    # screen is dropped, as before
                x, y = self.cursor[0] + mx, self.cursor[1] + my
                nx, ny = min(max(x, 0.0), self.W - 1.0), min(max(y, 0.0), self.H - 1.0)
                if nx != x:
                    cx = 0.0                                 # at the edge: nothing left to carry that way
                if ny != y:
                    cy = 0.0
                self._carry = (cx, cy)
                self.step = (nx - self.cursor[0], ny - self.cursor[1])
                self.cursor = (nx, ny)
                if self.push and self.speed < SPEED_SLOW:    # resting: home slowly follows the head (posture)
                    a = 1.0 - math.exp(-dt / HOME_FOLLOW_S)
                    self.home = (self.home[0] + a * (yaw - self.home[0]), self.home[1] + a * (pitch - self.home[1]))
        else:
            self.gate(pose.t_ms, yaw, pitch)     # the first frame starts the rest gate's window
        self._last = (pose.t_ms, yaw, pitch)
        return GazePoint(t_ms=pose.t_ms, x=self.cursor[0], y=self.cursor[1], confidence=1.0)


@dataclass
class PointerStats:
    """What the head pointer dropped, for tests/pointer_edges.py --replay. Counting only: no effect on moves."""
    glitch_skips: int = 0          # single wild frames skipped
    glitch_rebases: int = 0        # wild frames in a row taken as a real turn: that jump was not applied
    glitch_dropped_deg: float = 0.0
    hold_rebases: int = 0          # long 'blinks' (eyes unusable over BLINK_MAX_MS): the angle change was dropped
    hold_dropped_deg: float = 0.0
    capped_px: float = 0.0         # movement cut by MAX_STEP_PX in fast throws...
    edge_px: float = 0.0           # ...and how much of that was given back to finish a throw to an edge


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
        self.gaze_point: tuple[float, float] | None = None   # head-eyes: where the eyes are (smoothed), px
        self.eye_pull_px = 0.0                  # head-eyes: how far the eyes have pulled the cursor in all (tuning)
        self.nose_assist_px = 0.0               # head-nose: how far the nose has carried the cursor in all (tuning)
        self._nose_t: float | None = None
        self._eye_t: float | None = None
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
        if mode in NEEDS_GAZE and self.model is None:
            return False
        self.mode = mode
        self.rel.push = mode == "head-push"
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
        self._eye_t = None
        self._nose_t = None
        self.gaze_point = None
        self._hold_since = None
        self._was_holding = False

    def place(self, x: float, y: float) -> None:
        """Head modes: continue from (x, y), e.g. the button UI-Venus just found. Other modes: nothing."""
        if self.mode in REL_MODES:
            self.rel.cursor = (float(x), float(y))
            self.rel.rebase()

    def raw_gaze(self, feats: EyeFeatures | None, pose: HeadPose) -> tuple[float, float] | None:
        """Where the gaze model says you look, before smoothing and before the calibration shift."""
        if self.model is None or feats is None or not feats.usable:
            return None
        return self.model.predict(feats, pose)

    def update(self, t_ms: int, feats: EyeFeatures | None, pose: HeadPose, nose=None) -> GazePoint:
        usable = feats is None or feats.usable
        if not usable and self._hold_since is None:
            self._hold_since = t_ms
        holding = (not usable and self._held is not None and t_ms - self._hold_since <= BLINK_MAX_MS)
        if not holding:
            if self._was_holding and self.mode in REL_MODES:
                self.rel.stats.hold_rebases += 1
                if self._held is not None:
                    self.rel.stats.hold_dropped_deg += math.hypot(pose.yaw - self._held.yaw,
                                                                  pose.pitch - self._held.pitch)
                self.rel.rebase()     # restart from the current angle: the change during the hold isn't applied
            if usable:
                self._hold_since = None
            self._held = pose
        self._was_holding = holding
        if self.mode in REL_MODES:
            if holding:
                self._eye_t = None
                return GazePoint(t_ms=t_ms, x=self.rel.cursor[0], y=self.rel.cursor[1], confidence=1.0)
            point = self.rel.update(pose)
            if self.mode == "head-nose":
                return self._nose_assist(t_ms, nose, pose, point)
            return self._eye_pull(t_ms, feats, pose, point) if self.mode == "head-eyes" else point
        point_pose = pose if not holding else HeadPose(pose.t_ms, self._held.yaw, self._held.pitch, self._held.roll)
        gaze = None
        if self.mode in ("hybrid", "gaze"):
            g = self.raw_gaze(feats, pose)
            if g is not None:
                gx, gy = g[0] + self.gaze_shift[0], g[1] + self.gaze_shift[1]
                sx, sy = self.smoother(t_ms / 1000.0, gx / self.W, gy / self.H)
                gaze = GazePoint(t_ms=t_ms, x=sx * self.W, y=sy * self.H, confidence=1.0)
        return self.head.update(point_pose, gaze)

    def _nose_assist(self, t_ms: int, nose, pose: HeadPose, point: GazePoint) -> GazePoint:
        """head-nose: when the nose says the head is turned toward an edge (past NOSE_ZONE of the turn that reaches it)
        and the matrix agrees, carry the cursor toward the spot that turn points at, never backward. Returns the cursor
        point; the head pointer carries on from it."""
        last, self._nose_t = self._nose_t, t_ms / 1000.0
        if nose is None or last is None:
            return point
        dt = self._nose_t - last
        if not 0.0 < dt < 0.5:
            return point
        cx, cy = self.rel.cursor
        out = [cx, cy]
        size = (self.W, self.H)
        for axis, (angle, matrix, edge) in enumerate(((nose.yaw, pose.yaw, NOSE_EDGE_DEG[0]),
                                                      (nose.pitch, pose.pitch, NOSE_EDGE_DEG[1]))):
            a = angle / edge
            direction = 1.0 if a > 0 else -1.0
            m = abs(a)
            if m <= NOSE_ZONE or matrix * direction < 0.5 * NOSE_ZONE * edge:
                continue                                  # not far enough, or the matrix doesn't agree
            if self.rel.step[axis] * direction < -0.5:
                continue                                  # the head is already coming back
            mid = (size[axis] - 1) / 2
            target = min(max(mid + direction * min(m, 1.0) * mid, 0.0), size[axis] - 1.0)
            gap = (target - out[axis]) * direction
            if gap <= 0:
                continue                                  # the cursor is already there or beyond
            k = (1.0 - math.exp(-NOSE_PULL_RATE * dt)) * _ramp(m, NOSE_ZONE, 1.0)
            step = min(gap * k, NOSE_MAX_STEP_PX * HeadRelPointer.frame_scale(dt))
            out[axis] += direction * step
            self.nose_assist_px += step
        if (out[0], out[1]) == (cx, cy):
            return point
        self.rel.cursor = (out[0], out[1])
        return GazePoint(t_ms=t_ms, x=out[0], y=out[1], confidence=point.confidence)

    def _eye_pull(self, t_ms: int, feats: EyeFeatures | None, pose: HeadPose, point: GazePoint) -> GazePoint:
        """head-eyes: pull the head pointer's cursor toward where the eyes look, but only the part of the way that
        lies outside the gaze's error ellipse, only while the head moves (EYE_SPEED_DEG) and only when it moves that
        way (EYE_ALIGN). Returns the cursor point; the head pointer carries on from it."""
        g = self.raw_gaze(feats, pose)
        if g is None:                                    # blinking, or no face: nothing to go on
            self._eye_t = None
            return point
        t = t_ms / 1000.0
        sx, sy = self.smoother(t, (g[0] + self.gaze_shift[0]) / self.W, (g[1] + self.gaze_shift[1]) / self.H)
        self.gaze_point = (sx * self.W, sy * self.H)
        last, self._eye_t = self._eye_t, t
        if last is None or not 0.0 < t - last < 0.5:
            return point
        dt = t - last
        cx, cy = self.rel.cursor
        dx, dy = self.gaze_point[0] - cx, self.gaze_point[1] - cy
        e = math.hypot(dx / EYE_RADIUS_PX[0], dy / EYE_RADIUS_PX[1])
        hx, hy = self.rel.step
        head = math.hypot(hx, hy)
        if e <= 1.0 or head < 1e-6:
            return point
        w = _ramp(self.rel.speed, *EYE_SPEED_DEG) * _ramp((hx * dx + hy * dy) / (head * math.hypot(dx, dy)), *EYE_ALIGN)
        if w <= 0.0:
            return point
        strength = max(0.0, float(getattr(settings, "eye_pull", 1.0)))   # 0 = the head alone, 0.5 = half as strong
        k = (1.0 - 1.0 / e) * (1.0 - math.exp(-EYE_PULL_RATE * strength * dt)) * w
        px, py = dx * k, dy * k
        n, limit = math.hypot(px, py), EYE_MAX_STEP_PX * HeadRelPointer.frame_scale(dt)
        if n > limit:
            px, py = px * limit / n, py * limit / n
        nx, ny = min(max(cx + px, 0.0), self.W - 1.0), min(max(cy + py, 0.0), self.H - 1.0)
        self.eye_pull_px += math.hypot(nx - cx, ny - cy)
        self.rel.cursor = (nx, ny)
        return GazePoint(t_ms=t_ms, x=nx, y=ny, confidence=point.confidence)
