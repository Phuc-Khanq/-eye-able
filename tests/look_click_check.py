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
