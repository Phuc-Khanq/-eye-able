"""Dwell only on clickable things, with no Windows: DwellClicker's gate (looser radius, grace, clamp),
ClickableProbe's answers from fake cursor readings, the eyes service's ready field, and the meter's
ring states. The Win32 and Tk parts are tried live (see engine/eyes/clickable.py and dwell_meter.py).

Run with: python -m unittest tests.test_dwell_clickable
"""
import math
import unittest

from engine.core import messages
from engine.eyes.clickable import ClickableProbe, Over
from engine.eyes.dwell_click import DwellClicker
from engine.eyes.dwell_meter import DwellMeter, ring_state
from tests.test_dwell_click import FRAME_MS, ButtonOs, come, still
from tests.test_eyes_service import Rig

BUTTON = (440, 370, 560, 430)   # 120 x 60, around (500, 400)


def feed(d, points, over=None, t0=0, gate=True):
    """Like test_dwell_click.feed, with what's under the cursor: over is an Over or f(i, x, y) -> Over."""
    out = []
    for i, (x, y) in enumerate(points):
        o = over(i, x, y) if callable(over) else over
        at = d.update(t0 + i * FRAME_MS, x, y, over=o, gate=gate)
        if at is not None:
            out.append((i, at))
    return out


def on_button(i, x, y, rect=BUTTON):
    left, top, right, bottom = rect
    return Over(True, rect) if left <= x <= right and top <= y <= bottom else Over(False)


def circle(x, y, frames, r):
    """Swaying round (x, y) at r px, slower than a frame's settle speed: a wobbly head on a button."""
    return [(x + r * math.cos(i * 0.12), y + r * math.sin(i * 0.12)) for i in range(frames)]


class TestGate(unittest.TestCase):
    def test_no_click_on_unknown_or_blank(self):
        path = come(500, 400) + still(500, 400, 60)
        self.assertEqual(feed(DwellClicker(), path, over=None), [])
        self.assertEqual(feed(DwellClicker(), path, over=Over(False)), [])
        self.assertEqual(DwellClicker().progress, 0.0)

    def test_clicks_on_a_button(self):
        path = come(500, 400)
        clicks = feed(DwellClicker(), path + still(500, 400, 40), over=on_button)
        self.assertEqual(len(clicks), 1)
        self.assertAlmostEqual((clicks[0][0] - len(path)) * FRAME_MS, 1000, delta=70)

    def test_wobble_of_50_px_clicks_on_a_button_but_not_off_one(self):
        rest = circle(500, 400, 45, 50)                    # 100 px across: 40 px used to break this up
        big = (380, 320, 620, 480)
        self.assertEqual(len(feed(DwellClicker(), come(550, 400) + rest, over=lambda i, x, y: Over(True, big))), 1)
        self.assertEqual(feed(DwellClicker(), come(550, 400) + rest, gate=False), [])   # the old 40 px

    def test_hand_cursor_only_allows_60_px_not_100(self):
        self.assertEqual(len(feed(DwellClicker(), come(525, 400) + circle(500, 400, 45, 25), over=Over(True))), 1)
        self.assertEqual(feed(DwellClicker(), come(540, 400) + circle(500, 400, 45, 40), over=Over(True)), [])

    def test_click_point_stays_inside_the_button(self):
        small = (480, 390, 520, 410)                       # a 40 x 20 button; the cursor rests below its edge
        clicks = feed(DwellClicker(), come(500, 414) + still(500, 414, 40), over=lambda i, x, y: Over(True, small))
        self.assertEqual(len(clicks), 1)
        x, y = clicks[0][1]
        self.assertTrue(480 < x < 520 and 390 < y < 410, (x, y))

    def test_short_drop_holds_long_drop_starts_over_in_place(self):
        path = come(500, 400) + still(500, 400, 40)
        start = len(come(500, 400))

        def dropping(n):
            return lambda i, x, y: Over(False) if start + 15 <= i < start + 15 + n else Over(True, BUTTON)
        short = feed(DwellClicker(), path, over=dropping(5))     # 165 ms off the edge
        self.assertEqual(len(short), 1)
        self.assertAlmostEqual((short[0][0] - start) * FRAME_MS, 1000 + 5 * FRAME_MS, delta=70)   # held, not counted
        long = feed(DwellClicker(), path + still(500, 400, 30), over=dropping(12))   # 400 ms: the stop is gone...
        self.assertEqual(len(long), 1)
        self.assertGreater((long[0][0] - start) * FRAME_MS, 1000 + 27 * FRAME_MS)   # ...and starts over, no move needed

    def test_another_button_starts_a_new_stop(self):
        a, b = (440, 370, 520, 430), (521, 370, 600, 430)       # side by side
        path = come(500, 400) + still(500, 400, 20) + still(530, 400, 20)

        def over(i, x, y):
            return Over(True, a if x <= 520 else b)
        self.assertEqual(feed(DwellClicker(), path, over=over), [])   # 0.66 s on each: no click

    def test_progress_holds_during_the_grace(self):
        d = DwellClicker()
        feed(d, come(500, 400) + still(500, 400, 15), over=Over(True, BUTTON))
        before = d.progress
        self.assertGreater(before, 0)
        d.update(10_000_000, 500, 400, over=Over(False), gate=True)   # a long gap and blank: reset
        self.assertEqual(d.progress, 0.0)


class FakeReader:
    def __init__(self):
        self.at = (500, 400, True)

    def __call__(self):
        return self.at


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class TestProbe(unittest.TestCase):
    def setUp(self):
        self.clock, self.reader = Clock(), FakeReader()
        self.uia = {"answer": Over(False), "calls": 0}

        def ui_at(x, y):
            self.uia["calls"] += 1
            return self.uia["answer"]
        self.p = ClickableProbe(self.reader, ui_at, clock=self.clock)

    def test_hand_cursor_is_clickable_near_it(self):
        self.p.poll_hand()
        self.assertEqual(self.p.over_at(510, 405), Over(True))
        self.assertIsNone(self.p.over_at(700, 400))            # too far from where it was read

    def test_old_answers_are_unknown(self):
        self.p.poll_hand()
        self.clock.t += 0.5
        self.assertIsNone(self.p.over_at(500, 400))

    def test_arrow_cursor_is_not_clickable(self):
        self.reader.at = (500, 400, False)
        self.p.poll_hand()
        self.assertEqual(self.p.over_at(500, 400), Over(False))

    def test_ui_automation_gives_the_rectangle(self):
        self.reader.at = (500, 400, False)                     # a text box: an I-beam, not the hand
        self.uia["answer"] = Over(True, BUTTON)
        self.p.poll_hand()
        self.p.poll_uia()
        self.assertEqual(self.p.over_at(530, 420), Over(True, BUTTON))
        self.assertEqual(self.p.over_at(500, 445), Over(False))  # off the rectangle: the hand says no

    def test_ui_automation_asks_again_only_after_a_move_or_a_while(self):
        self.p.poll_hand()
        self.p.poll_uia()
        self.p.poll_uia()
        self.assertEqual(self.uia["calls"], 1)
        self.reader.at = (510, 400, True)
        self.p.poll_hand()
        self.p.poll_uia()
        self.assertEqual(self.uia["calls"], 2)

    def test_own_window_keeps_the_last_answer(self):
        self.uia["answer"] = Over(True, BUTTON)
        self.p.poll_hand()
        self.p.poll_uia()
        self.uia["answer"] = "own"
        self.reader.at = (520, 400, False)
        self.p.poll_hand()
        self.p.poll_uia()
        self.assertEqual(self.p.over_at(520, 400), Over(True, BUTTON))


class FakeProbe:
    def __init__(self):
        self.clickable = True

    def over_at(self, x, y, now=None):
        return Over(True, (x - 80, y - 40, x + 80, y + 40)) if self.clickable else Over(False)

    def start(self):
        pass

    def stop(self):
        pass


class TestServiceReady(unittest.TestCase):
    def setUp(self):
        self.r = Rig()
        self.r.os = self.r.out.os = ButtonOs()
        self.probe = self.r.eyes.probe = FakeProbe()
        self.r.eyes.set_dwell(True)
        self.r.frames(60)

    def tearDown(self):
        self.r.out.stop()

    def move_and_rest(self, yaw=4.0, frames=45):
        self.r.frames(8, yaw=yaw)
        self.r.frames(frames, yaw=yaw)
        self.r.out.flush()

    def test_clicks_on_something_clickable_and_says_ready(self):
        self.move_and_rest()
        self.assertEqual(len(self.r.os.clicks), 1)
        sent = self.r.ui(messages.DWELL)
        self.assertTrue(any(m["ready"] for m in sent))
        self.assertEqual((sent[-1]["progress"], sent[-1]["ready"]), (0.0, False))   # disarmed after the click
        pairs = [(m["progress"], m["ready"]) for m in sent]
        self.assertTrue(all(a != b for a, b in zip(pairs, pairs[1:])))             # sent only on a change

    def test_no_click_on_blank_space(self):
        self.probe.clickable = False
        self.move_and_rest(frames=60)
        self.assertEqual(self.r.os.clicks, [])
        self.assertFalse(any(m["ready"] for m in self.r.ui(messages.DWELL)))

    def test_clickable_only_off_clicks_anywhere(self):
        self.probe.clickable = False
        self.r.eyes.settings.dwell_clickable_only = False
        try:
            self.move_and_rest()
        finally:
            self.r.eyes.settings.dwell_clickable_only = True
        self.assertEqual(len(self.r.os.clicks), 1)


class FakeBus:
    def subscribe(self, topic, fn):
        pass

    def publish(self, topic, payload):
        pass


class TestMeter(unittest.TestCase):
    def test_states(self):
        self.assertEqual(ring_state(0.0, False, 0.0, 1.0), ("hidden", 0.0))
        self.assertEqual(ring_state(0.0, True, 0.0, 1.0), ("ready", 0.0))
        state, extent = ring_state(0.5, True, 0.0, 1.0)
        self.assertEqual(state, "filling")
        self.assertAlmostEqual(extent, -180, delta=1)       # half way, clockwise
        self.assertEqual(ring_state(0.0, False, 2.0, 1.0)[0], "done")

    def test_click_flashes_then_hides(self):
        clock = Clock()
        m = DwellMeter(FakeBus(), read_cursor=lambda: None, clock=clock)
        m.on_ui(messages.make(messages.DWELL, progress=0.5, ready=True))
        self.assertEqual(m.state()[0], "filling")
        m.on_ui(messages.make(messages.DWELL, progress=1.0, ready=False))
        m.on_ui(messages.make(messages.DWELL, progress=0.0, ready=False))
        self.assertEqual(m.state()[0], "done")
        clock.t += 0.3
        self.assertEqual(m.state()[0], "hidden")

    def test_hides_while_ui_venus_searches(self):
        clock = Clock()
        m = DwellMeter(FakeBus(), read_cursor=lambda: None, clock=clock)
        m.on_ui(messages.make(messages.DWELL, progress=0.0, ready=True))
        self.assertEqual(m.state()[0], "ready")
        m.on_ui(messages.make(messages.TARGET_SEARCHING, label="send"))
        self.assertEqual(m.state()[0], "hidden")
        m.on_ui(messages.make(messages.TARGET_NONE, label="send"))
        m.on_ui(messages.make(messages.DWELL, progress=0.0, ready=True))
        clock.t += 5.0                                      # resting on a link: no new message, still shown
        self.assertEqual(m.state()[0], "ready")

    def test_old_ui_message_without_ready(self):
        m = DwellMeter(FakeBus(), read_cursor=lambda: None, clock=Clock())
        m.on_ui({"type": "dwell", "progress": 0.3})
        self.assertEqual(m.state()[0], "filling")


if __name__ == "__main__":
    unittest.main()
