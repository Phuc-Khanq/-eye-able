"""Checks head_pose, head_gestures and head_pointer on made-up motion, no camera needed.

Run:  .\\.venv\\Scripts\\python.exe -m tests.head_gestures_check
Each scenario is a head motion with the gestures it should produce. They include the awkward real-life
cases: a nod that settles at a different height, a tracking glitch, a lost face, slow looking around.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from engine.core.types import FaceFrame, GazePoint, HeadPose
from engine.eyes.head_gestures import GestureDetector
from engine.eyes.head_pointer import HeadPointer
from engine.eyes.head_pose import HeadPoseEstimator, angle_diff, raw_angles

FRAME_MS = 33
N = 90  # 3 seconds of frames


def bump(n: int, start: int, length: int, size: float) -> np.ndarray:
    """A smooth out-and-back of the given size."""
    out = np.zeros(n)
    out[start:start + length] = size * np.sin(np.linspace(0, np.pi, length))
    return out


def dip(n: int, start: int, down: int, up: int, size: float, settle: float = 0.0) -> np.ndarray:
    """A quick move of `size`, then back to `settle` (a different resting height than the start)."""
    out = np.zeros(n)
    out[start:start + down] = size * np.sin(np.linspace(0, np.pi / 2, down))
    out[start + down:start + down + up] = settle + (size - settle) * np.cos(np.linspace(0, np.pi / 2, up))
    out[start + down + up:] = settle
    return out


def shake_track(n: int, start: int, amp: float = 14, cycles: float = 2, frames: int = 24, offset: float = 0.0):
    out = np.full(n, offset)
    out[start:start + frames] += amp * np.sin(np.linspace(0, cycles * np.pi, frames))
    return out


def run(yaw, pitch, roll=None, seed=0, drop=(), spike=None) -> list[str]:
    """Feed angle tracks, one value per frame, with 0.3 degrees of tracking noise.
    drop: frame numbers where the face was lost (no pose at all). spike: (frame, degrees) glitch on pitch."""
    rng = np.random.default_rng(seed)
    yaw, pitch = np.asarray(yaw, float), np.asarray(pitch, float)
    roll = np.zeros_like(yaw) if roll is None else np.asarray(roll, float)
    det = GestureDetector(nod_deg=8, shake_deg=8, tilt_deg=15, tilt_hold_ms=400, cooldown_ms=800)
    found = []
    for i in range(len(yaw)):
        if i in drop:
            continue
        noise = rng.normal(0, 0.3, 3)
        p = pitch[i] + (spike[1] if spike and i == spike[0] else 0.0)
        g = det.update(HeadPose(t_ms=i * FRAME_MS, yaw=yaw[i] + noise[0], pitch=p + noise[1], roll=roll[i] + noise[2]))
        if g is not None:
            found.append(g.type)
    return found


flat = np.zeros(N)


def tilt_track(n=N, start=20, length=30, size=20.0):
    r = np.zeros(n)
    r[start:start + length] = size
    return r


def flicker(r: np.ndarray) -> np.ndarray:
    """A held tilt whose roll dips under the threshold for a frame or two."""
    r = r.copy()
    r[30], r[38] = 12.0, 13.0
    return r


# (name, function returning the list of gestures, what it should be)
SCENARIOS = [
    ("still head", lambda: run(flat, flat), []),
    ("nod down", lambda: run(flat, bump(N, 20, 12, 14)), ["nod"]),
    ("nod up", lambda: run(flat, bump(N, 20, 12, -14)), ["nod"]),
    ("two nods", lambda: run(flat, bump(N, 5, 12, 14) + bump(N, 50, 12, 14)), ["nod", "nod"]),
    ("small dip is not a nod", lambda: run(flat, bump(N, 20, 12, 4)), []),
    ("looking down and staying", lambda: run(flat, np.clip(np.arange(N) - 20.0, 0, 15)), []),
    ("slow look down and back (2.3 s)", lambda: run(flat, bump(N, 10, 70, 14)), []),
    ("phone glance: down 1.2 s then back", lambda: run(flat, bump(N, 20, 38, 18)), []),
    ("nod that settles 5 deg lower", lambda: run(flat, dip(N, 20, 5, 7, 14, settle=-5.0)), ["nod"]),
    ("nod that settles 5 deg higher", lambda: run(flat, dip(N, 20, 5, 7, -14, settle=5.0)), ["nod"]),
    ("nod while head drifts down slowly", lambda: run(flat, bump(N, 25, 12, 14) + np.linspace(0, 6, N)), ["nod"]),
    ("single-frame tracking glitch", lambda: run(flat, flat, spike=(40, 25.0)), []),
    ("face lost for 0.4 s mid-nod", lambda: run(flat, bump(N, 20, 24, 14), drop=range(30, 43)), []),
    ("nod with a little sideways drift", lambda: run(bump(N, 20, 12, 3), bump(N, 20, 12, 14)), ["nod"]),
    ("shake", lambda: run(shake_track(N, 20), flat), ["shake"]),
    ("shake starting off center", lambda: run(shake_track(N, 20, offset=6.0), flat), ["shake"]),
    ("look right and back is not a shake", lambda: run(bump(N, 20, 24, 20), flat), []),
    ("slow scan across the screen", lambda: run(shake_track(N, 5, amp=12, cycles=2, frames=80), flat), []),
    ("shake that settles 5 deg off", lambda: run(shake_track(N, 20) + np.clip(np.arange(N) - 44.0, 0, 5), flat), ["shake"]),
    ("tilt right held", lambda: run(flat, flat, tilt_track()), ["tilt_right"]),
    ("tilt left held", lambda: run(flat, flat, -tilt_track()), ["tilt_left"]),
    ("tilt right with roll flicker", lambda: run(flat, flat, flicker(tilt_track())), ["tilt_right"]),
    ("brief tilt", lambda: run(flat, flat, tilt_track(length=5)), []),
    ("tilt held 3 s fires once", lambda: run(np.zeros(200), np.zeros(200), tilt_track(200, 20, 90)), ["tilt_right"]),
    ("tilt right, release, tilt right again", lambda: run(np.zeros(240), np.zeros(240),
        tilt_track(240, 20, 30) + tilt_track(240, 120, 30)), ["tilt_right", "tilt_right"]),
    ("shaking for 4 s is one shake", lambda: run(shake_track(200, 20, amp=14, cycles=8, frames=120), np.zeros(200)), ["shake"]),
    ("two nods 1.6 s apart", lambda: run(np.zeros(200), bump(200, 20, 14, 14) + bump(200, 75, 14, 14)), ["nod", "nod"]),
    ("three quick nods in a row count once", lambda: run(np.zeros(200), bump(200, 20, 14, 14) + bump(200, 38, 14, 14)
                                                           + bump(200, 56, 14, 14)), ["nod"]),
    ("nod, pause, shake", lambda: run(shake_track(200, 90), bump(200, 20, 14, 14)), ["nod", "shake"]),
    ("tilt right while nodding is not a nod", lambda: run(np.zeros(200), bump(200, 20, 14, 14),
        tilt_track(200, 20, 40, 18)), ["tilt_right"]),
    ("tilt right, then a nod after it", lambda: run(np.zeros(200), bump(200, 100, 14, 14),
        tilt_track(200, 20, 30, 20)), ["tilt_right", "nod"]),
    ("tilt right starts while the head is still in a shake's cooldown", lambda: run(shake_track(200, 20), np.zeros(200),
        tilt_track(200, 60, 30, 20)), ["shake", "tilt_right"]),
]


def check(name: str, got, want) -> bool:
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got}, want {want}")
    return ok


def matrix(rx=0.0, ry=0.0, rz=0.0) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = Rotation.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix()
    return m


def face(m: np.ndarray, t_ms: int = 0) -> FaceFrame:
    return FaceFrame(t_ms, 1280, 720, np.zeros((478, 3)), {}, m)


def main() -> None:
    results = [check(name, fn(), want) for name, fn, want in SCENARIOS]

    # head_pose signs: build matrices the way MediaPipe would for each motion.
    yaw, pitch, roll = raw_angles(matrix(ry=-10))   # face normal toward image left = subject's right
    results.append(check("turn right gives yaw > 0", yaw > 9, True))
    yaw, pitch, roll = raw_angles(matrix(rx=10))    # face normal toward -y = chin down
    results.append(check("chin down gives pitch > 0", pitch > 9, True))
    scaled = matrix(rz=10)
    scaled[:3, :3] *= 1.3
    results.append(check("scale in the matrix is ignored", round(raw_angles(scaled)[2]), 10))

    est = HeadPoseEstimator()
    est.set_neutral(face(matrix(ry=-5)))
    results.append(check("neutral pose reads 0", round(est.compute(face(matrix(ry=-5), 33)).yaw, 3), 0.0))

    rng = np.random.default_rng(3)
    est = HeadPoseEstimator()
    for i in range(20):   # a steady look at the center with a noisy frame or two
        wobble = 12.0 if i == 18 else rng.normal(0, 0.4)
        est.compute(face(matrix(rx=3 + wobble * 0.0, ry=-5 + wobble), i * 33))
    est.set_neutral()
    pose = est.compute(face(matrix(rx=3, ry=-5), 700))
    results.append(check("neutral ignores one bad frame", abs(pose.yaw) < 1.0 and abs(pose.pitch) < 1.0, True))

    results.append(check("angle difference wraps at 180", round(angle_diff(-179.0, 179.0)), 2))

    # head_pointer
    hp = HeadPointer(1920, 1080)
    results.append(check("neutral points at center", tuple(round(v) for v in hp.to_cursor(HeadPose(0, 0, 0, 0))), (960, 540)))
    results.append(check("full right turn reaches the edge", round(hp.to_cursor(HeadPose(0, 30, 0, 0))[0]), 1919))
    hp = HeadPointer(1920, 1080)
    hp.update(HeadPose(0, 0, 0, 0), GazePoint(0, 500, 400, 0.9))
    results.append(check("gaze jump puts the cursor there", tuple(round(v) for v in hp.cursor), (500, 400)))
    for i in range(40):   # head turned 2 deg and tipped -1 deg and held; the head filter settles
        g = hp.update(HeadPose(400 + i * FRAME_MS, 2, -1, 0), GazePoint(400 + i * FRAME_MS, 560, 380, 0.9))
    ok = abs(g.x - 540) < 4 and abs(g.y - 380) < 4
    results.append(check("small gaze move keeps head fine-tuning", ok, True))
    g = hp.update(HeadPose(2000, 2, -1, 0), GazePoint(2000, 1500, 800, 0.9))
    results.append(check("big gaze move jumps", (round(g.x), round(g.y)), (1500, 800)))

    # head jitter must not shake the cursor: 0.3 degrees of noise at 20 px per degree is 6 px if unfiltered
    hp = HeadPointer(1920, 1080)
    hp.update(HeadPose(0, 0, 0, 0), GazePoint(0, 900, 500, 1.0))
    xs = []
    for i in range(1, 120):
        pose = HeadPose(i * FRAME_MS + 500, rng.normal(0, 0.4), rng.normal(0, 0.4), 0)
        xs.append(hp.update(pose, None).x)
    jitter = float(np.std(xs[30:]))
    results.append(check("head jitter on a still head stays under 3.5 px", jitter < 3.5, True))
    hp = HeadPointer(1920, 1080)
    hp.update(HeadPose(0, 0, 0, 0), GazePoint(0, 900, 500, 1.0))
    last = None
    for i in range(1, 40):   # a steady 20 deg/s turn must not lag far behind the unfiltered position
        last = hp.update(HeadPose(i * FRAME_MS + 500, i * FRAME_MS / 1000 * 20, 0, 0), None).x
    unfiltered = 900 + (39 * FRAME_MS / 1000 * 20) * 20.0
    results.append(check("head pointer lag at 20 deg/s under 40 px", unfiltered - last < 40, True))

    print(f"\n{sum(results)} of {len(results)} passed")
    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
