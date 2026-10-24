"""Checks the eyebrow-raise detector on made-up blendshape scores, no camera needed.

Run:  .\\.venv\\Scripts\\python.exe -m tests.brow_check
Each scenario is a score track (one value per frame, 33 ms apart) with the raises it should produce.
"""
import numpy as np

from engine.eyes.brow_gesture import BrowRaiseDetector

FRAME_MS = 33
N = 120  # 4 seconds
REST = 0.08


def track(*raises, rest=REST, n=N) -> np.ndarray:
    """Resting score with (start frame, length in frames, height above rest) raises, smooth edges."""
    out = np.full(n, rest)
    for start, length, height in raises:
        ramp = min(3, length // 2)
        shape = np.ones(length)
        shape[:ramp] = np.linspace(0, 1, ramp + 1)[1:]
        shape[length - ramp:] = np.linspace(1, 0, ramp + 1)[:-1]
        out[start:start + length] += height * shape
    return out


def run(score, seed=0, noise=0.01, blink=(), drop=(), split=False, frame_ms=FRAME_MS) -> list[int]:
    """Feed scores as blendshapes; returns the frame numbers where brow_raise fired.
    blink: frames with eyes closed. drop: frames with no face. split: inner brow only (outer at rest).
    frame_ms: time between frames (a webcam in dim light can slow to 125 ms, 8 fps)."""
    rng = np.random.default_rng(seed)
    det = BrowRaiseDetector()
    fired = []
    for i, s in enumerate(score):
        if i in drop:
            continue
        s = float(np.clip(s + rng.normal(0, noise), 0, 1))
        outer = REST if split else s
        shapes = {"browInnerUp": s, "browOuterUpLeft": outer, "browOuterUpRight": outer,
                  "eyeBlinkLeft": 0.9 if i in blink else 0.05, "eyeBlinkRight": 0.9 if i in blink else 0.05}
        if det.update(i * frame_ms, shapes) is not None:
            fired.append(i)
    return fired


def talking(n=N, seed=1) -> np.ndarray:
    """Brows bob with speech: small, quick, up to about 0.15 above rest, never held."""
    rng = np.random.default_rng(seed)
    out = np.full(n, REST)
    for start in range(15, n - 6, 7):
        out[start:start + 4] += rng.uniform(0.05, 0.15)
    return out


def surprise_flash(n=N) -> np.ndarray:
    """A quick 'oh!' flash: high but only about 130 ms."""
    return track((40, 4, 0.45), n=n)


SCENARIOS = [
    ("clean raise fires once", lambda: len(run(track((30, 20, 0.4)))), 1),
    ("clean raise fires about hold_ms after it starts", lambda: 34 <= run(track((30, 20, 0.4)))[0] <= 42, True),
    ("raise too short (130 ms) fires nothing", lambda: run(track((30, 4, 0.4))), []),
    ("raise too small (0.15) fires nothing", lambda: run(track((30, 30, 0.15))), []),
    ("long hold (3 s) fires once", lambda: len(run(track((20, 90, 0.4)))), 1),
    ("two raises 1.5 s apart fire twice", lambda: len(run(track((15, 15, 0.4), (60, 15, 0.4)))), 2),
    ("talking wobble fires nothing", lambda: run(talking()), []),
    ("talking with more noise fires nothing", lambda: run(talking(seed=3), noise=0.03), []),
    ("quick surprised flash fires nothing", lambda: run(surprise_flash()), []),
    ("still face fires nothing", lambda: run(np.full(N, REST), noise=0.03), []),
    ("naturally high brows (rest 0.35) still fire", lambda: len(run(track((40, 20, 0.35), rest=0.35))), 1),
    ("naturally high brows don't fire at rest", lambda: run(np.full(N, 0.35), noise=0.02), []),
    ("inner brows only still fire", lambda: len(run(track((30, 20, 0.4)), split=True)), 1),
    ("blinks during a raise don't break it", lambda: len(run(track((30, 25, 0.4)), blink={33, 34, 40})), 1),
    ("a blink alone fires nothing", lambda: run(np.full(N, REST), blink=set(range(30, 36))), []),
    ("face lost mid-raise: no click from two halves",
     lambda: run(track((30, 12, 0.4)), drop=set(range(35, 50))), []),
    ("slow camera (8 fps): a 0.75 s raise fires once", lambda: len(run(track((20, 6, 0.5)), frame_ms=125)), 1),
    ("slow camera (8 fps): two raises fire twice",
     lambda: len(run(track((20, 6, 0.5), (45, 6, 0.5)), frame_ms=125)), 2),
    ("slow camera (8 fps): talking fires nothing", lambda: run(talking(), frame_ms=125), []),
    ("hold flickering at the edge fires once",
     lambda: len(run(track((20, 60, 0.4)) - 0.18 * (np.arange(N) % 9 == 0) * (np.arange(N) > 30))), 1),
]


def check(name: str, got, want) -> bool:
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got}, want {want}")
    return ok


def main() -> None:
    results = [check(name, fn(), want) for name, fn, want in SCENARIOS]

    det = BrowRaiseDetector()
    det.set_thresholds(threshold=0.4, hold_ms=150)
    results.append(check("set_thresholds changes threshold and hold", (det.threshold, det.hold_ms), (0.4, 150)))

    print(f"\n{sum(results)} of {len(results)} passed")
    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
