"""Dwell = left click and eyebrow raise = right click, with no camera: DwellClicker on made-up cursor
paths, then the eyes service end to end (tests.test_eyes_service's Rig: real pointer, detectors and mode
manager, fake os_input).

Run with: python -m unittest tests.test_dwell_click
"""
import math
import unittest

from engine.core import messages, topics
from engine.eyes.dwell_click import DwellClicker
from tests.test_eyes_service import FakeOs, Rig

FRAME_MS = 33


def feed(d: DwellClicker, points, t0=0):
    """(x, y) per frame, 33 ms apart; returns [(frame, click point)]."""
    out = []
    for i, (x, y) in enumerate(points):
        at = d.update(t0 + i * FRAME_MS, x, y)
        if at is not None:
            out.append((i, at))
    return out


def still(x, y, frames, wobble=0.0):
    """Resting at (x, y), swaying up to `wobble` px each way, about once a second like a real head."""
    return [(x + wobble * math.sin(i * 0.23), y + wobble * math.cos(i * 0.31)) for i in range(frames)]


def come(x, y, frm=200):
    """Rest 2 s somewhere else, then move to (x, y) in 0.33 s: a deliberate move onto a button."""
    return still(x - frm, y, 60) + [(x - frm + frm * i / 10, y) for i in range(1, 11)]


class TestDwellClicker(unittest.TestCase):
    def test_rest_one_second_clicks_once(self):
        path = come(500, 400)
        clicks = feed(DwellClicker(), path + still(500, 400, 40))
        self.assertEqual(len(clicks), 1)
        self.assertAlmostEqual((clicks[0][0] - len(path)) * FRAME_MS, 1000, delta=70)   # 1 s after it stopped

    def test_parked_cursor_clicks_only_once(self):
        path = come(500, 400) + still(500, 400, 200)   # then about 6.6 s of reading
        self.assertEqual(len(feed(DwellClicker(), path)), 1)

    def test_wobble_inside_the_radius_still_clicks_at_the_middle(self):
        clicks = feed(DwellClicker(), come(500, 400) + still(500, 400, 40, wobble=14))
        self.assertEqual(len(clicks), 1)
        x, y = clicks[0][1]
        self.assertLess(math.hypot(x - 500, y - 400), 5)

    def test_slow_drift_does_not_click(self):
        path = [(300 + 1.6 * i, 400) for i in range(120)]   # about 50 px/s, still moving
        self.assertEqual(feed(DwellClicker(), path), [])

    def test_reading_creep_does_not_click(self):
        """Reading moves the head: the cursor creeps 30 px, settles 1.5 s, creeps again (Hung's log)."""
        path = still(500, 400, 60)
        for k in range(1, 7):
            path += [(500 + 30 * (k - 1) + 6 * i, 400 + 3 * k) for i in range(1, 6)] + still(500 + 30 * k, 400 + 3 * k, 45)
        self.assertEqual(feed(DwellClicker(), path), [])

    def test_click_lands_on_the_rest_not_the_arrival(self):
        arrive = [(560 - 12 * i, 400) for i in range(5)]   # decelerating in from the right
        clicks = feed(DwellClicker(), still(800, 400, 60) + arrive + still(500, 400, 40))
        self.assertLess(abs(clicks[0][1][0] - 500), 6)

    def test_move_away_and_rest_clicks_again(self):
        path = come(500, 400) + still(500, 400, 40) + [(500 + 15 * i, 400) for i in range(1, 11)] + still(650, 400, 40)
        self.assertEqual(len(feed(DwellClicker(), path)), 2)

    def test_disarmed_spot_needs_a_move_first(self):
        d = DwellClicker()
        d.disarm(500, 400)     # e.g. a right click just opened a menu here
        self.assertEqual(feed(d, still(505, 400, 60)), [])
        path = [(505 + 12 * i, 400) for i in range(1, 11)] + still(625, 400, 40)
        self.assertEqual(len(feed(d, path, t0=60 * FRAME_MS)), 1)

    def test_off_means_off(self):
        d = DwellClicker()
        d.set_enabled(False)
        self.assertEqual(feed(d, come(500, 400) + still(500, 400, 60)), [])
        self.assertEqual(d.progress, 0.0)

    def test_lost_face_mid_stop_does_not_add_up(self):
        d = DwellClicker()
        path = come(500, 400) + still(500, 400, 20)                   # 0.66 s on the spot
        self.assertEqual(feed(d, path), [])
        self.assertEqual(feed(d, still(500, 400, 15), t0=len(path) * FRAME_MS + 850), [])   # gap, then 0.5 s


class ButtonOs(FakeOs):
    """Records the button too: (x, y) for left clicks, (x, y, "right") for right clicks."""

    def click(self, x, y, button="left"):
        self.clicks.append((x, y) if button == "left" else (x, y, button))


class TestEyesServiceClicks(unittest.TestCase):
    def setUp(self):
        self.r = Rig()
        self.r.os = self.r.out.os = ButtonOs()
        self.r.eyes.set_dwell(True)
        self.r.frames(60)   # the one-look calibration

    def tearDown(self):
        self.r.out.stop()

    def move_and_rest(self, yaw=4.0, frames=45):
        self.r.frames(8, yaw=yaw)      # turn: the head pointer moves the cursor
        self.r.frames(frames, yaw=yaw)
        self.r.out.flush()

    def gaze(self):
        g = self.r.bus.of(topics.GAZE)[-1]
        return g.x, g.y

    def test_no_click_on_the_center_after_calibrating(self):
        self.r.frames(60)
        self.r.out.flush()
        self.assertEqual(self.r.os.clicks, [])

    def test_rest_left_clicks_once_where_it_rested(self):
        self.move_and_rest()
        self.r.frames(90, yaw=4.0)     # keep reading with the cursor parked
        self.r.out.flush()
        self.assertEqual(len(self.r.os.clicks), 1)
        x, y = self.r.os.clicks[0]
        gx, gy = self.gaze()
        self.assertLess(math.hypot(x - gx, y - gy), 3)

    def test_dwell_progress_fills_then_clears(self):
        self.move_and_rest()
        progress = [m["progress"] for m in self.r.ui(messages.DWELL)]
        self.assertGreater(len(progress), 20)
        self.assertEqual(progress[progress.index(1.0) + 1], 0.0)   # full on the click, then the ring goes
        self.assertEqual(progress[:progress.index(1.0)], sorted(progress[:progress.index(1.0)]))

    def test_raise_freezes_the_cursor_where_it_started(self):
        r = self.r
        self.move_and_rest(frames=5)             # stopped on a spot, well before a dwell
        start = self.gaze()
        r.frames(12, yaw=6.0, brow=0.45)         # brows up, and the head drifts right with them
        during = [(g.x, g.y) for g in r.bus.of(topics.GAZE)[-10:]]
        fired = [g for g in r.bus.of(topics.GESTURE) if g.type == "brow_raise"]
        self.assertEqual(len(fired), 1)
        for x, y in during:
            self.assertLess(math.hypot(x - start[0], y - start[1]), 4)   # held at the onset spot
        r.frames(3, yaw=6.0)                     # brows down
        x, y = self.gaze()
        self.assertLess(abs(x - start[0]), 30)   # carries on from there, no jump to where the head went

    def test_no_dwell_during_a_raise_or_after_a_right_click(self):
        r = self.r
        self.move_and_rest(frames=5)
        r.frames(40, yaw=4.0, brow=0.45)         # a long raise on a resting cursor: 1.3 s
        x, y = self.gaze()
        r.bus.publish(topics.OS_CLICK, {"x": int(x), "y": int(y), "button": "right"})   # what Khoi's line will do
        r.frames(60, yaw=4.0)                    # the context menu is open right under the cursor
        r.out.flush()
        self.assertEqual(r.os.clicks, [(int(x), int(y), "right")])

    def test_right_click_reaches_os_input_with_the_button(self):
        self.r.bus.publish(topics.OS_CLICK, {"x": 300, "y": 200, "button": "right"})
        self.r.bus.publish(topics.OS_CLICK, {"x": 310, "y": 210})
        self.r.out.flush()
        self.assertEqual(self.r.os.clicks, [(300, 200, "right"), (310, 210)])

    def test_no_dwell_in_dictation(self):
        self.r.say("start dictation")
        self.move_and_rest(frames=60)
        self.assertEqual(self.r.os.clicks, [])

    def test_click_by_name_then_rest_clicks_once(self):
        r = self.r
        r.say("click subscribe")
        found = r.ui(messages.TARGET_FOUND)[-1]
        r.raise_brows()
        r.frames(60)                             # the cursor stays on the button it just clicked
        r.out.flush()
        self.assertEqual(r.os.clicks, [(found["x"], found["y"])])

    def test_no_dwell_while_a_preview_waits(self):
        r = self.r
        r.say("type hello there")
        self.assertTrue(r.eyes.preview_waiting)
        self.move_and_rest(frames=60)
        self.assertEqual(r.os.clicks, [])

    def test_stop_clicking_and_start_again(self):
        r = self.r
        r.eyes.set_dwell(False)
        self.move_and_rest(frames=60)
        self.assertEqual(r.os.clicks, [])
        r.eyes.set_dwell(True)                   # turned on while resting: not this spot
        r.frames(60, yaw=4.0)
        r.out.flush()
        self.assertEqual(r.os.clicks, [])
        self.move_and_rest(yaw=-2.0)             # move, then rest: clicks
        self.assertEqual(len(r.os.clicks), 1)

    def test_no_dwell_click_after_scrolling_until_the_cursor_moves(self):
        r = self.r
        self.move_and_rest(frames=10)            # stop on a spot (no click yet)...
        r.set_mode("scroll")                     # ...say "scroll": the cursor stays there while the page moves
        r.frames(20)
        r.frames(60, pitch=6.0)
        r.set_mode("pointer")
        r.frames(60, pitch=6.0)                  # back to pointing, the cursor still resting on that spot
        r.out.flush()
        self.assertEqual(r.os.clicks, [])        # the page under it changed: no surprise click
        self.move_and_rest(yaw=-2.0)             # move, then rest: dwell works again
        self.assertEqual(len(r.os.clicks), 1)


if __name__ == "__main__":
    unittest.main()
