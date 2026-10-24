"""Offline check of engine.eyes.look_click with a fake finder (no camera, no model).  Run: python -m tests.look_click_check"""
from concurrent.futures import Future

from engine.core.types import TargetResult
from engine.eyes.look_click import LookClick


class Fake:
    def __init__(self, answers):
        self.answers, self.asked = list(answers), []

    def submit(self, req):
        self.asked.append(req)
        f = Future()
        f.set_result(self.answers.pop(0))
        return f


def feed(lc, t0, x, y, n=30):
    for i in range(n):
        lc.add_gaze(t0 + i * 0.033, x, y)
    return t0 + n * 0.033


def main() -> None:
    f = Fake([TargetResult("avatar", True, 1000, 500), TargetResult("nothing", False), TargetResult("avatar", True, 1000, 500)])
    lc = LookClick(f, 1920, 1080)
    assert "No gaze" in lc.request("x", 0.0)
    t = feed(lc, 0.0, 900, 450)
    print(lc.request("avatar", t)); assert lc.state == "searching" and lc.busy
    acts = lc.poll(t); print(acts); assert acts[0] == ("move", 1000, 500) and lc.state == "confirm"
    assert lc.confirm() == ("click", 1000, 500) and lc.state == "idle"
    assert lc.shift == (50.0, 25.0), lc.shift                      # half of the (100, 50) error
    t = feed(lc, t + 1, 900, 450)
    lc.request("nothing", t); acts = lc.poll(t); print(acts); assert "not found" in acts[0][1] and lc.state == "idle"
    assert f.asked[0].gaze_x == 900 and f.asked[1].gaze_x == 950, "learned shift should move the next anchor"
    t = feed(lc, t + 1, 900, 450)
    lc.request("avatar", t); lc.poll(t); assert lc.poll(t + 10)[0][1].startswith("No nod") and lc.state == "idle"
    assert lc.confirm() is None
    # cap and far-answer rule
    lc2 = LookClick(Fake([TargetResult("a", True, 1900, 1000)]), 1920, 1080)
    t = feed(lc2, 0, 100, 100); lc2.request("a", t); lc2.poll(t); lc2.confirm()
    assert lc2.shift == (0.0, 0.0), "answers over 450 px away are not learned from"
    # auto mode
    lc3 = LookClick(Fake([TargetResult("a", True, 700, 400)]), 1920, 1080, auto=True)
    t = feed(lc3, 0, 650, 380); lc3.request("a", t); acts = lc3.poll(t); assert acts[-1] == ("click", 700, 400), acts
    # cancel while searching drops a late answer
    lc4 = LookClick(Fake([TargetResult("a", True, 10, 10)]), 1920, 1080)
    t = feed(lc4, 0, 50, 50); lc4.request("a", t); assert lc4.cancel() and lc4.poll(t) == [] and lc4.state == "idle"
    print("look_click ok", lc.stats)


if __name__ == "__main__":
    main()


def check_confirm_nod() -> None:
    from engine.eyes.look_click import ConfirmNod
    c = ConfirmNod(deg=6)
    t, fired = 0, []
    def run(pitches):
        nonlocal t
        for p in pitches:
            t += 33
            fired.append(c.update(t, p))
    run([1.0] * 40)                                  # resting
    run([3, 6, 8, 9, 8, 5, 2, 1, 1])                 # a modest nod: 8 deg down and back
    assert sum(fired) == 1, fired
    fired.clear(); c.reset(); run([1.0] * 40)
    run([3, 6, 8, 9] + [9] * 60)                     # looked down and stayed: no click
    assert sum(fired) == 0
    fired.clear(); c.reset(); run([1.0] * 40)
    run([2, 3, 4, 4, 3, 2])                          # too small
    assert sum(fired) == 0 and 2.5 < c.deepest < 3.5, c.deepest
    fired.clear(); c.reset(); run([1.0] * 40)
    run([4, 7, 8.7, 8, 6, 4.5, 4, 4, 4])             # 7.7 deg dip, head settles 3 deg lower: still a nod
    assert sum(fired) == 1, fired
    print("confirm nod ok")


def check_cursor() -> None:
    import math
    import random
    from engine.eyes.cursor_driver import CursorDriver
    from engine.eyes.smoothing import StableCursor
    s = StableCursor(6)
    s(100, 100)
    assert s(104, 103) == (100, 100), "wobble inside the dead zone doesn't move it"
    x, y = s(120, 100)
    assert abs(x - 114) < 1e-6 and y == 100, (x, y)
    moves = []
    d = CursorDriver(lambda x, y: moves.append((x, y)), glide_ms=45, still_px=6)
    random.seed(1)
    # head held still with the demo PC's measured sway: 0.13 deg at 64 px/deg, plus 2 px noise
    raw, out = [], []
    for i in range(300):
        sway = 8 * math.sin(i / 9.0)
        px, py = 900 + sway + random.gauss(0, 2), 500 + 0.5 * sway + random.gauss(0, 2)
        raw.append((px, py))
        if i % 4 == 0:                    # camera at 30 Hz, glide at 120 Hz
            d.set_target(px, py)
        out.append(d.step(1 / 120))
    def rms(pts):
        mx = sum(p[0] for p in pts) / len(pts); my = sum(p[1] for p in pts) / len(pts)
        return math.sqrt(sum((p[0] - mx) ** 2 + (p[1] - my) ** 2 for p in pts) / len(pts))
    print(f"still head: raw wobble {rms(raw[60:]):.1f} px -> cursor {rms(out[60:]):.1f} px")
    assert rms(out[60:]) < 0.5 * rms(raw[60:])
    # a 600 px move: the cursor should arrive within 0.25 s and stop within the dead zone
    d.snap(100, 500)
    t = 0
    for i in range(120):
        if i % 4 == 0:
            d.set_target(100 + min(600, 600 * i / 40), 500)
        p = d.step(1 / 120)
        if t == 0 and p[0] >= 690:
            t = i / 120
    assert 0 < t < 0.6, t
    print(f"600 px move reaches the target in {t:.2f} s; ends {700 - p[0]:.1f} px short (dead zone)")


if __name__ == "__main__":
    check_confirm_nod()
    check_cursor()
