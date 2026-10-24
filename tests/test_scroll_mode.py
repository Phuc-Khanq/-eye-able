"""Run with: python -m unittest tests.test_scroll_mode

Task 7: scroll mode. Say "scroll", tilt the head to scroll (the eyes service does that part), say
"stop". The mode manager only switches the mode: the eyes service reads mode "scroll", holds the
cursor and scrolls, and reports its speed in the "scroll" UI message (used for the 10 s timeout).
"""
import unittest
from types import SimpleNamespace

from engine.core import messages, topics
from engine.core.mode_manager import Mode, ModeManager
from engine.core.types import GazePoint, Gesture, TargetResult
from engine.voice.commands import parse
from engine.voice.dictation import Dictation, DictationWorker
from tests.helpers import FakeBus, FakeWhisperModel
from tests.test_mouth_gate import FakeMouth


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class Base(unittest.TestCase):
    def setUp(self, mouth=None):
        self.bus, self.clock = FakeBus(), Clock()
        self.worker = DictationWorker(self.bus, Dictation(model=FakeWhisperModel()), settings=None)
        self.mm = ModeManager(self.bus, clock=self.clock, mouth=mouth,
                              settings=SimpleNamespace(scroll_idle_s=10.0), log=lambda s: None)
        self.bus.publish(topics.GAZE, GazePoint(0, 500, 300, 0.9))

    def say(self, text):
        self.worker.process(text)

    def gesture(self, kind, t_ms=100_000):
        self.bus.publish(topics.GESTURE, Gesture(t_ms=t_ms, type=kind))

    def eyes_scroll(self, rate):
        # what the eyes service sends about 10 times a second in scroll mode
        self.bus.publish(topics.TO_UI, messages.make(messages.SCROLL, rate=rate, offset=0.0, ready=True))

    def modes(self):
        return [m["mode"] for m in self.bus.ui(messages.MODE)]

    def os(self):
        return [(t, p) for t, p in self.bus.log if t.startswith("os.")]


class TestEntering(Base):
    def test_the_phrases(self):
        for text in ["scroll", "Scroll.", "scroll mode", "start scrolling", "can you scroll", "help me scroll",
                     "I want to scroll", "let me scroll", "um, scroll please", "computer scroll", "Venus, scroll"]:
            with self.subTest(text=text):
                self.setUp()
                self.say(text)
                self.assertEqual(self.mm.mode, Mode.SCROLL)
                self.assertEqual(self.modes(), ["scroll"])
                self.assertEqual(self.os(), [])                     # entering doesn't scroll by itself

    def test_scroll_alone_never_scrolls_up(self):
        # the bug from step 0: "scroll" fuzzy-matched "scroll up"
        for text in ["scroll", "can you scroll", "Scroll."]:
            with self.subTest(text=text):
                self.assertEqual(parse(text).name, "scroll_mode")

    def test_near_misses_stay_speech(self):
        for text in ["I scrolled past it", "the scroll bar is broken", "stop scrolling", "don't scroll",
                     "no scrolling", "I like to scroll through photos at night"]:
            with self.subTest(text=text):
                self.setUp()
                self.say(text)
                self.assertEqual(self.mm.mode, Mode.POINTER)
                self.assertEqual(self.os(), [])

    def test_scroll_up_and_down_stay_one_shot_in_pointer(self):
        self.say("scroll down")
        self.say("scroll up")
        self.assertEqual(self.mm.mode, Mode.POINTER)
        self.assertEqual(self.os(), [(topics.OS_SCROLL, {"direction": "down"}), (topics.OS_SCROLL, {"direction": "up"})])

    def test_never_from_dictation_it_is_text(self):
        self.say("start dictation")
        self.say("scroll")
        self.say("Venus, scroll")
        self.assertEqual(self.mm.mode, Mode.DICTATION)
        self.assertEqual(self.mm._preview_text(), "scroll scroll")

    def test_from_the_menu_and_back_to_where_the_menu_was_opened(self):
        self.say("menu")
        self.say("scroll")
        self.assertEqual(self.mm.mode, Mode.SCROLL)
        self.say("stop")
        self.assertEqual(self.mm.mode, Mode.POINTER)
        self.assertEqual(self.modes(), ["menu", "scroll", "pointer"])

    def test_ignored_while_paused(self):
        self.say("pause")
        self.say("scroll")
        self.assertEqual(self.mm.mode, Mode.PAUSED)


class TestInScrollMode(Base):
    def setUp(self, mouth=None):
        super().setUp(mouth)
        self.say("scroll")

    def test_stop_done_and_stop_scrolling_go_back_to_pointer(self):
        for text in ["stop", "Stop.", "done", "stop scrolling", "okay, stop"]:
            with self.subTest(text=text):
                self.setUp()
                self.say(text)
                self.assertEqual(self.mm.mode, Mode.POINTER)
                self.assertEqual(self.modes(), ["scroll", "pointer"])

    def test_another_command_leaves_then_runs(self):
        self.say("click send")
        self.assertEqual(self.mm.mode, Mode.POINTER)
        self.assertEqual(self.bus.of(topics.TARGET_REQUEST)[-1].label, "Send")
        self.setUp()
        self.say("start dictation")
        self.assertEqual(self.modes(), ["scroll", "pointer", "dictation"])

    def test_scroll_up_down_glide_and_stay(self):
        self.say("scroll down")
        self.say("scroll up")
        self.assertEqual(self.mm.mode, Mode.SCROLL)
        self.assertEqual(self.os(), [(topics.OS_SCROLL, {"direction": "down"}), (topics.OS_SCROLL, {"direction": "up"})])

    def test_other_speech_is_ignored_quietly(self):
        self.say("this page is really long")
        self.assertEqual(self.mm.mode, Mode.SCROLL)
        self.assertEqual(self.bus.ui(messages.STATUS), [])

    def test_a_raise_leaves_and_never_right_clicks(self):
        self.gesture("brow_raise")
        self.assertEqual(self.mm.mode, Mode.POINTER)
        self.assertEqual(self.bus.of(topics.OS_CLICK), [])

    def test_a_raise_while_talking_stays(self):
        self.setUp(mouth=FakeMouth(moving=[(98_000, 101_000)]))
        self.gesture("brow_raise")
        self.assertEqual(self.mm.mode, Mode.SCROLL)

    def test_head_gestures_do_nothing(self):
        for kind in ("nod", "shake", "tilt_left", "tilt_right"):
            self.gesture(kind)
        self.assertEqual(self.mm.mode, Mode.SCROLL)
        self.assertEqual(self.os(), [])

    def test_no_dwell_click(self):
        self.bus.publish(topics.DWELL_DONE, GazePoint(0, 300, 200, 0.9))
        self.assertEqual(self.bus.of(topics.OS_CLICK), [])


class TestTimeout(Base):
    def setUp(self, mouth=None):
        super().setUp(mouth)
        self.say("scroll")

    def test_ten_seconds_without_scrolling_ends_it(self):
        self.clock.t = 9.9
        self.mm.tick()
        self.assertEqual(self.mm.mode, Mode.SCROLL)
        self.clock.t = 10.0
        self.mm.tick()
        self.assertEqual(self.mm.mode, Mode.POINTER)

    def test_scrolling_keeps_it_going(self):
        for t in (5.0, 12.0, 19.0):
            self.clock.t = t
            self.eyes_scroll(rate=6.0)
            self.mm.tick()
        self.clock.t = 28.9
        self.mm.tick()
        self.assertEqual(self.mm.mode, Mode.SCROLL)          # 9.9 s since the last scroll
        self.clock.t = 29.0
        self.mm.tick()
        self.assertEqual(self.mm.mode, Mode.POINTER)

    def test_holding_still_is_not_scrolling(self):
        self.clock.t = 5.0
        self.eyes_scroll(rate=0.2)                            # inside the dead zone: barely moving
        self.clock.t = 10.0
        self.mm.tick()
        self.assertEqual(self.mm.mode, Mode.POINTER)

    def test_scroll_messages_outside_scroll_mode_are_ignored(self):
        self.say("stop")
        self.eyes_scroll(rate=6.0)
        self.assertEqual(self.mm.mode, Mode.POINTER)


class TestWithTheEyesService(unittest.TestCase):
    """End to end on Khang's rig: the real EyesService and ModeManager on one bus and one clock,
    with a simulated face. Saying "scroll" must make the eyes service scroll with the head."""

    def setUp(self):
        from tests.test_eyes_service import Rig

        self.r = Rig()
        self.r.frames(60)                         # the one-look calibration

    def scroll_rates(self):
        return [m["rate"] for m in self.r.bus.ui(messages.SCROLL)]

    def frames(self, n, pitch=0.0):
        """Rig frames, stepping the SmoothScroller by hand: in the app its own 60 Hz thread eases the
        speed that the "scroll" UI message reports; the rig doesn't start that thread."""
        for _ in range(n):
            self.r.frames(1, pitch=pitch)
            self.r.eyes.scroller.step(0.033)

    def test_say_scroll_tilt_then_stop(self):
        r = self.r
        r.say("scroll")
        self.assertEqual((r.mm.mode, r.eyes.mode), (Mode.SCROLL, "scroll"))
        self.frames(20)                           # it learns "straight ahead"
        self.frames(45, pitch=6.0)                # chin down: scrolls down
        self.assertTrue(any(rate < -SCROLL_FAST for rate in self.scroll_rates()))
        r.say("stop")
        self.assertEqual((r.mm.mode, r.eyes.mode), (Mode.POINTER, "pointer"))

    def test_scrolling_keeps_it_on_holding_still_ends_it(self):
        r = self.r
        r.say("scroll")
        self.frames(20)
        for _ in range(5):                        # 5 x 3 s of scrolling: longer than the 10 s timeout
            self.frames(90, pitch=6.0)
            r.mm.tick()
        self.assertEqual(r.mm.mode, Mode.SCROLL)
        # head straight: the speed coasts below 0.5 notches/s within about 0.9 s, then 10 s of idle
        self.frames(270)                          # about 8.9 s: still in scroll mode
        r.mm.tick()
        self.assertEqual(r.mm.mode, Mode.SCROLL)
        self.frames(90)                           # about 11.9 s: ended by itself
        r.mm.tick()
        self.assertEqual((r.mm.mode, r.eyes.mode), (Mode.POINTER, "pointer"))

    def test_a_raise_leaves_scroll_mode_without_a_right_click(self):
        r = self.r
        r.say("scroll")
        r.raise_brows()
        r.out.flush()
        self.assertEqual(r.mm.mode, Mode.POINTER)
        self.assertEqual(r.os.clicks, [])


SCROLL_FAST = 0.5   # notches/s


if __name__ == "__main__":
    unittest.main()
