"""Head scrolling (engine/eyes/scroll.py): the head sets a speed, the speed is eased into small wheel steps.
Run: python -m unittest tests.test_scroll"""
import unittest

from engine.eyes.scroll import (DEAD_DEG, GLIDE_NOTCHES, MAX_RATE, MIN_UNITS, SETTLE_MS, UNITS_PER_NOTCH,
                                HeadScroll, SmoothScroller)


def settled_head(pitch=0.0):
    """A HeadScroll that has finished learning 'straight ahead' at this pitch (30 fps)."""
    h = HeadScroll()
    t = 0
    while not h.settled:
        t += 33
        h.update(t, pitch)
    h.t = t
    return h


def hold(h, pitch, frames=30):
    for _ in range(frames):
        h.t += 33
        r = h.update(h.t, pitch)
    return r


class TestHeadScroll(unittest.TestCase):
    def test_nothing_happens_while_it_learns_straight_ahead(self):
        h, t = HeadScroll(), 0
        while t < SETTLE_MS - 40:
            t += 33
            self.assertEqual(h.update(t, 5.0 + (t % 3)), 0.0)   # even a moving head: no scrolling yet
        self.assertFalse(h.settled)

    def test_straight_ahead_is_wherever_you_were(self):
        h = settled_head(pitch=7.0)                              # sits tilted 7 degrees from the calibration
        self.assertAlmostEqual(hold(h, 7.0), 0.0)
        self.assertLess(hold(h, 7.0 + 6.0), 0)                   # chin down from there: scroll down

    def test_dead_zone(self):
        h = settled_head()
        self.assertEqual(hold(h, DEAD_DEG - 0.3), 0.0)
        self.assertEqual(hold(h, -(DEAD_DEG - 0.3)), 0.0)

    def test_chin_down_scrolls_down_and_up_scrolls_up(self):
        self.assertLess(hold(settled_head(), 6.0), 0)            # negative = down (os_input: positive = up)
        self.assertGreater(hold(settled_head(), -6.0), 0)

    def test_gentle_for_reading_fast_when_far(self):
        small = abs(hold(settled_head(), DEAD_DEG + 1.0))        # 1 degree past the dead zone
        big = abs(hold(settled_head(), DEAD_DEG + 6.0))
        self.assertLess(small, 2.5)                              # about 150 px a second in a browser
        self.assertGreater(small, 0.5)
        self.assertGreater(big, 10)
        self.assertGreater(big, 4 * small)

    def test_capped(self):
        self.assertAlmostEqual(abs(hold(settled_head(), 11.0, frames=60)), MAX_RATE, delta=1.0)   # about 10.5 degrees
        self.assertLessEqual(abs(hold(settled_head(), 20.0, frames=60)), MAX_RATE)

    def test_one_wild_frame_is_skipped(self):
        h = settled_head()
        hold(h, 5.0)
        before = h.rate
        h.t += 33
        self.assertEqual(h.update(h.t, 40.0), before)            # a tracking glitch: keep the speed
        self.assertAlmostEqual(hold(h, 5.0, frames=3), before, delta=1.0)

    def test_still_head_noise_does_not_scroll(self):
        import random
        random.seed(3)
        h = settled_head()
        for _ in range(300):
            h.t += 33
            self.assertEqual(h.update(h.t, random.gauss(0, 0.15)), 0.0)   # measured rest noise on the demo PC


class Wheel:
    def __init__(self):
        self.steps = []

    def __call__(self, units):
        self.steps.append(units)

    @property
    def total(self):
        return sum(self.steps)


def run(sc, seconds, hz=60):
    for _ in range(int(seconds * hz)):
        sc.step(1.0 / hz)


class TestSmoothScroller(unittest.TestCase):
    def setUp(self):
        self.w = Wheel()
        self.sc = SmoothScroller(self.w)

    def test_speed_eases_in_no_jump(self):
        self.sc.set_rate(-10.0)
        rates = []
        for _ in range(60):
            self.sc.step(1 / 60)
            rates.append(self.sc.rate)
        self.assertGreater(rates[0], -2.5)                            # the first tick is 1/4 of the speed at most
        self.assertTrue(all(b <= a for a, b in zip(rates, rates[1:])))   # speeds up steadily, never overshoots
        self.assertGreaterEqual(rates[-1], -10.0)
        self.assertTrue(all(abs(b - a) < 1.5 for a, b in zip(rates, rates[1:])))   # no big change tick to tick
        run(self.sc, 1.5)
        self.assertAlmostEqual(self.sc.rate, -10.0, delta=0.1)

    def test_rate_converts_to_the_right_distance(self):
        self.sc.set_rate(10.0)
        run(self.sc, 3.0)                                             # 10 notches/s, eased in, 3 s
        notches = self.w.total / UNITS_PER_NOTCH
        self.assertGreater(notches, 25)
        self.assertLess(notches, 30)

    def test_coasts_to_a_stop_like_a_trackpad(self):
        self.sc.set_rate(-20.0)
        run(self.sc, 2.0)
        released = self.w.total
        self.sc.set_rate(0.0)
        n0 = len(self.w.steps)
        run(self.sc, 0.2)
        after_a_moment = self.w.total
        self.assertGreater(len(self.w.steps), n0 + 3)                 # it keeps going for a moment...
        self.assertLess(after_a_moment, released - 300)               # (down = negative)
        run(self.sc, 3.0)
        self.assertEqual(self.sc.rate, 0.0)
        end = self.w.total
        self.assertLess(end, after_a_moment - 300)                    # ...and glides on, slowing down
        run(self.sc, 1.0)
        self.assertEqual(self.w.total, end)                           # ...then truly stops

    def test_steps_are_small_and_frequent(self):
        self.sc.set_rate(25.0)
        run(self.sc, 2.0)
        steady = self.w.steps[-30:]
        self.assertLess(max(steady), UNITS_PER_NOTCH)                 # less than a notch per step
        self.assertGreaterEqual(len(steady), 30)                      # a step nearly every tick

    def test_no_tiny_steps(self):
        self.sc.set_rate(0.3)                                         # a very slow crawl
        run(self.sc, 5.0)
        self.assertTrue(all(abs(u) >= MIN_UNITS for u in self.w.steps))
        self.assertGreater(abs(self.w.total), 100)

    def test_reversing_brakes_first(self):
        self.sc.set_rate(-15.0)
        run(self.sc, 1.5)
        self.sc.set_rate(15.0)
        signs = []
        for _ in range(90):
            self.sc.step(1 / 60)
            signs.append(self.sc.rate)
        self.assertTrue(all(b >= a for a, b in zip(signs, signs[1:])))   # one smooth swing through zero
        self.assertGreater(signs[-1], 10)

    def test_voice_glide_is_smooth_and_five_notches(self):
        self.sc.glide("down")
        run(self.sc, 1.2)
        self.assertAlmostEqual(self.w.total / UNITS_PER_NOTCH, -GLIDE_NOTCHES, delta=0.1)
        self.assertGreaterEqual(len(self.w.steps), 20)               # many steps, not one 5-notch jump
        self.assertLess(max(abs(u) for u in self.w.steps), 0.4 * UNITS_PER_NOTCH * GLIDE_NOTCHES / 5 * 5)
        self.assertEqual(self.sc.rate, 0.0)

    def test_glide_up_and_overlapping_glides_add(self):
        self.sc.glide("up")
        self.sc.glide("up")
        run(self.sc, 1.5)
        self.assertAlmostEqual(self.w.total / UNITS_PER_NOTCH, 2 * GLIDE_NOTCHES, delta=0.1)

    def test_idle_sends_nothing(self):
        run(self.sc, 3.0)
        self.assertEqual(self.w.steps, [])
        self.assertFalse(self.sc.busy)

    def test_stop_clears_everything(self):
        self.sc.set_rate(20.0)
        self.sc.glide("down")
        run(self.sc, 0.5)
        self.sc.stop()
        n = len(self.w.steps)
        run(self.sc, 1.0)
        self.assertEqual(len(self.w.steps), n)

    def test_thread_runs_and_stops(self):
        import time
        sc = SmoothScroller(self.w, rate_hz=100)
        sc.start()
        sc.set_rate(20.0)
        time.sleep(0.5)
        sc.stop()
        self.assertGreater(self.w.total, 500)
        n = len(self.w.steps)
        time.sleep(0.1)
        self.assertEqual(len(self.w.steps), n)

    def test_a_failing_wheel_call_does_not_kill_the_thread(self):
        import time
        errors, calls = [], []

        def boom(units):
            calls.append(units)
            raise OSError("no wheel")
        sc = SmoothScroller(boom, rate_hz=100, on_error=errors.append)
        sc.start()
        sc.set_rate(20.0)
        time.sleep(0.4)
        sc.stop()
        self.assertGreater(len(calls), 5)
        self.assertEqual(len(errors), 1)                              # reported once, not every tick


if __name__ == "__main__":
    unittest.main()
