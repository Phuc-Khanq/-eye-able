"""Run with: python -m unittest tests.test_clicks

With nothing waiting: an eyebrow raise = the LEFT click, snapped onto the nearest button by UI-Venus
(more in tests/test_snap_click.py); opening the mouth and closing it, or "right click" = the RIGHT click.
brow_click "right" (settings) brings back the old raise = right click (Task 6).
A raise still means yes when a question, a preview or a UI-Venus search is waiting.
"""
import unittest
from types import SimpleNamespace

from engine.core import messages, topics
from engine.core.mode_manager import Mode, ModeManager
from engine.core.types import GazePoint, Gesture, TargetResult
from engine.voice.dictation import Dictation, DictationWorker
from tests.helpers import FakeBus, FakeWhisperModel
from tests.test_mouth_gate import FakeMouth


class Base(unittest.TestCase):
    def setUp(self, mouth=None, gaze=(500, 300)):
        self.bus = FakeBus()
        self.worker = DictationWorker(self.bus, Dictation(model=FakeWhisperModel()), settings=None)
        self.mm = ModeManager(self.bus, mouth=mouth, log=lambda s: None)
        if gaze:
            self.bus.publish(topics.GAZE, GazePoint(0, *gaze, 0.9))

    def say(self, text):
        self.worker.process(text)

    def raise_brows(self, t_ms=100_000):
        self.bus.publish(topics.GESTURE, Gesture(t_ms=t_ms, type="brow_raise"))

    def blink(self, t_ms=100_000):
        self.bus.publish(topics.GESTURE, Gesture(t_ms=t_ms, type="blink"))

    def clicks(self):
        return self.bus.of(topics.OS_CLICK)


class TestRaiseLeftClicks(Base):
    def answer(self, found=True, x=0, y=0):
        req = self.bus.of(topics.TARGET_REQUEST)[-1]
        self.bus.publish(topics.TARGET_RESULT, TargetResult(req.label, found, x, y))
        return req

    def test_a_blink_with_nothing_waiting_clicks_the_exact_pixel(self):
        self.blink()
        self.assertEqual(self.bus.of(topics.TARGET_REQUEST), [])          # no UI-Venus for a blink
        self.assertEqual(self.clicks(), [{"x": 500, "y": 300}])

    def test_it_clicks_where_the_cursor_froze(self):
        self.bus.publish(topics.GAZE, GazePoint(0, 612.7, 410.2, 0.9))   # the frozen spot during the blink
        self.blink()
        self.assertEqual(self.clicks(), [{"x": 612, "y": 410}])

    def test_the_voice_click_snaps(self):
        self.say("click")
        req = self.answer(True, 512, 296)
        self.assertEqual((req.gaze_x, req.gaze_y, req.snap), (500, 300, True))
        self.assertEqual(self.clicks(), [{"x": 512, "y": 296}])

    def test_no_cursor_yet_says_so_and_does_not_click(self):
        self.setUp(gaze=None)
        self.blink()
        self.assertEqual(self.clicks(), [])
        self.assertIn("Can't click", self.bus.ui(messages.STATUS)[-1]["text"])

    def test_a_raise_while_talking_does_not_click(self):
        self.setUp(mouth=FakeMouth(moving=[(98_000, 101_000)]))
        self.raise_brows()
        self.assertEqual(self.bus.of(topics.TARGET_REQUEST), [])
        self.assertEqual(self.clicks(), [])

    def test_brow_click_right_keeps_the_old_right_click(self):
        self.bus = FakeBus()
        self.mm = ModeManager(self.bus, settings=SimpleNamespace(brow_click="right"), log=lambda s: None)
        self.bus.publish(topics.GAZE, GazePoint(0, 500, 300, 0.9))
        self.raise_brows()
        self.assertEqual(self.clicks(), [{"x": 500, "y": 300, "button": "right"}])


class TestRightClick(Base):
    def test_mouth_open_right_clicks_at_the_cursor(self):
        self.bus.publish(topics.GESTURE, Gesture(t_ms=100_000, type="mouth_open"))
        self.assertEqual(self.clicks(), [{"x": 500, "y": 300, "button": "right"}])

    def test_saying_right_click(self):
        self.say("right click")
        self.assertEqual(self.clicks(), [{"x": 500, "y": 300, "button": "right"}])

    def test_mouth_open_while_the_mic_hears_speech_is_ignored(self):
        self.bus.publish(topics.TO_UI, messages.make(messages.LISTENING, level=0.4, speaking=True))
        self.bus.publish(topics.GESTURE, Gesture(t_ms=100_000, type="mouth_open"))
        self.assertEqual(self.clicks(), [])

    def test_mouth_open_does_not_answer_a_question(self):
        self.say("click send")
        req = self.bus.of(topics.TARGET_REQUEST)[-1]
        self.bus.publish(topics.TARGET_RESULT, TargetResult(req.label, True, 410, 220))
        self.bus.publish(topics.GESTURE, Gesture(t_ms=100_000, type="mouth_open"))
        self.assertEqual(self.clicks(), [])
        self.assertIsNotNone(self.mm.question)


class TestStillYesWhenSomethingWaits(Base):
    def test_answers_an_open_question_with_a_left_click(self):
        self.say("click send")
        req = self.bus.of(topics.TARGET_REQUEST)[-1]
        self.bus.publish(topics.TARGET_RESULT, TargetResult(req.label, True, 410, 220))
        self.raise_brows()
        self.assertEqual(self.clicks(), [{"x": 410, "y": 220}])

    def test_types_a_waiting_preview(self):
        self.say("type see you soon")
        self.raise_brows()
        self.assertEqual(self.bus.of(topics.OS_TYPE), [{"text": "see you soon "}])
        self.assertEqual(self.clicks(), [])

    def test_preconfirms_a_running_search(self):
        self.say("click send")
        self.raise_brows()
        self.assertTrue(self.mm.search.preconfirmed)
        self.assertEqual(self.clicks(), [])

    def test_while_dictating_with_nothing_waiting_does_nothing(self):
        self.say("start dictation")
        self.raise_brows()
        self.assertEqual(self.clicks(), [])
        self.assertEqual(self.mm.mode, Mode.DICTATION)

    def test_paused_and_menu_are_unchanged(self):
        self.say("pause")
        self.raise_brows()
        self.assertEqual(self.clicks(), [])
        self.say("resume")
        self.say("menu")
        self.raise_brows()
        self.assertEqual(self.clicks(), [])
        self.assertEqual(self.bus.ui(messages.GESTURE)[-1], {"type": "gesture", "gesture": "brow_raise"})


class TestDwellIsTheLeftClick(Base):
    def test_dwell_left_clicks_with_the_same_payload_as_before(self):
        self.bus.publish(topics.DWELL_DONE, GazePoint(0, 300, 200, 0.9))
        self.assertEqual(self.clicks(), [{"x": 300, "y": 200}])          # no "button": left by default


class TestDwellOnOff(Base):
    """Task 6b: "stop clicking" / "start clicking" turn dwell (the left click) off / on."""

    def dwell(self):
        return self.bus.of(topics.DWELL)

    def test_pointer_mode(self):
        self.say("Stop clicking.")
        self.say("start clicking")
        self.assertEqual(self.dwell(), [{"on": False}, {"on": True}])
        self.assertEqual(self.bus.ui(messages.STATUS), [])     # the eyes service shows "Dwell clicking on/off"

    def test_while_dictating_it_is_a_command_not_text(self):
        self.say("start dictation")
        self.say("stop clicking")
        self.assertEqual(self.dwell(), [{"on": False}])
        self.assertEqual(self.mm.preview, [])

    def test_after_a_trigger_word_inside_dictation(self):
        self.say("start dictation")
        self.say("hi mom")
        self.say("Venus, stop clicking")
        self.assertEqual(self.dwell(), [{"on": False}])
        self.assertEqual(self.mm._preview_text(), "hi mom")

    def test_near_misses_stay_speech(self):
        for text in ["stop clicking on that", "I'll start clicking around", "stopped clicking", "clicking"]:
            with self.subTest(text=text):
                self.setUp()
                self.say(text)
                self.say("start dictation")
                self.say(text)
                self.assertEqual(self.dwell(), [])
                self.assertEqual(self.mm._preview_text(), text)

    def test_ignored_while_paused(self):
        self.say("pause")
        self.say("stop clicking")
        self.assertEqual(self.dwell(), [])


if __name__ == "__main__":
    unittest.main()
