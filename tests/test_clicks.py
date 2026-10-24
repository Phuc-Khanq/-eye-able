"""Run with: python -m unittest tests.test_clicks

Task 6 (team decision): dwell = LEFT click, an eyebrow raise = RIGHT click when nothing is waiting.
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

    def clicks(self):
        return self.bus.of(topics.OS_CLICK)


class TestRightClick(Base):
    def test_a_raise_with_nothing_waiting_right_clicks_at_the_cursor(self):
        self.raise_brows()
        self.assertEqual(self.clicks(), [{"x": 500, "y": 300, "button": "right"}])

    def test_it_clicks_where_the_cursor_froze(self):
        self.bus.publish(topics.GAZE, GazePoint(0, 612.7, 410.2, 0.9))   # the frozen spot during the raise
        self.raise_brows()
        self.assertEqual(self.clicks(), [{"x": 612, "y": 410, "button": "right"}])

    def test_no_cursor_yet_says_so_and_does_not_click(self):
        self.setUp(gaze=None)
        self.raise_brows()
        self.assertEqual(self.clicks(), [])
        self.assertIn("right-click", self.bus.ui(messages.STATUS)[-1]["text"])

    def test_a_raise_while_talking_does_not_right_click(self):
        self.setUp(mouth=FakeMouth(moving=[(98_000, 101_000)]))
        self.raise_brows()
        self.assertEqual(self.clicks(), [])


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


if __name__ == "__main__":
    unittest.main()
