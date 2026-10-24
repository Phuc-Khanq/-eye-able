"""Run with: python -m unittest tests.test_review_checks

Part D review: Khang's mode-manager timing rules and main.py's console log, each pinned by a test.
"""
import io
import unittest
from contextlib import redirect_stdout

from engine.core import messages, topics
from engine.core.mode_manager import CONFIRM_TIMEOUT_S, REPEAT_CLICK_S, TARGET_TIMEOUT_S, Mode, ModeManager
from engine.core.types import GazePoint, Gesture, TargetResult
from engine.voice.dictation import Dictation, DictationWorker
from tests.helpers import FakeBus, FakeWhisperModel


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class Base(unittest.TestCase):
    def setUp(self):
        self.bus, self.clock = FakeBus(), Clock()
        self.worker = DictationWorker(self.bus, Dictation(model=FakeWhisperModel()), settings=None)
        self.mm = ModeManager(self.bus, clock=self.clock)
        self.bus.publish(topics.GAZE, GazePoint(0, 500, 300, 0.9))

    def say(self, text):
        self.worker.process(text)

    def answer(self, found=True):
        req = self.bus.of(topics.TARGET_REQUEST)[-1]
        self.bus.publish(topics.TARGET_RESULT, TargetResult(req.label, found, 410, 220))

    def clicks(self):
        return [p for t, p in self.bus.log if t == topics.OS_CLICK]

    def states(self):
        return [q["state"] for q in self.bus.ui(messages.CONFIRM)]


class TestQuestionTimeout(Base):
    def test_open_until_just_before_12_s_then_expired_and_counts_as_no(self):
        self.assertEqual(CONFIRM_TIMEOUT_S, 12.0)
        self.say("start dictation")
        self.say("hi mom")
        self.say("stop dictation")                       # "Insert "hi mom" before stopping?"
        self.clock.t = CONFIRM_TIMEOUT_S - 0.01
        self.mm.tick()
        self.assertEqual(self.states(), ["open"])
        self.clock.t = CONFIRM_TIMEOUT_S
        self.mm.tick()
        self.assertEqual(self.states(), ["open", "expired"])
        # counted as no: nothing typed, the preview dropped, dictation left
        self.assertEqual(self.bus.of(topics.OS_TYPE), [])
        self.assertEqual(self.mm.preview, [])
        self.assertEqual(self.mm.mode, Mode.POINTER)


class TestEarlyYes(Base):
    def test_clicks_once_when_found(self):
        self.say("click send")
        self.say("yes")
        self.say("yes")                                   # impatient: still one click
        self.answer()
        self.assertEqual(self.clicks(), [{"x": 410, "y": 220}])
        self.assertEqual(self.mm.question, None)

    def test_no_click_if_not_found(self):
        self.say("click send")
        self.say("yes")
        self.answer(found=False)
        self.assertEqual(self.clicks(), [])
        self.assertEqual(self.bus.ui(messages.TARGET_NONE)[-1]["label"], "Send")

    def test_no_click_if_the_search_times_out(self):
        self.say("click send")
        self.say("yes")
        self.clock.t = TARGET_TIMEOUT_S
        self.mm.tick()
        self.answer()                                    # UI-Venus answers too late
        self.assertEqual(self.clicks(), [])


class TestRepeats(Base):
    def test_repeating_click_send_while_asked_answers_yes(self):
        self.say("click send")
        self.answer()
        self.say("click send")
        self.assertEqual(self.clicks(), [{"x": 410, "y": 220}])
        self.assertEqual(len(self.bus.of(topics.TARGET_REQUEST)), 1)

    def test_again_within_3_s_is_ignored_after_3_s_is_a_new_search(self):
        self.assertEqual(REPEAT_CLICK_S, 3.0)
        self.say("click send")
        self.answer()
        self.say("yes")                                   # clicked at t = 0
        self.clock.t = REPEAT_CLICK_S - 0.01
        self.say("click send")
        self.assertEqual(len(self.bus.of(topics.TARGET_REQUEST)), 1)
        self.clock.t = REPEAT_CLICK_S
        self.say("click send")
        self.assertEqual(len(self.bus.of(topics.TARGET_REQUEST)), 2)


class TestConsoleLog(unittest.TestCase):
    def test_every_handled_message_prints_without_crashing(self):
        from engine.main import log_to_console

        bus = FakeBus()
        out = io.StringIO()
        with redirect_stdout(out):
            log_to_console(bus)
            for msg in [messages.make(messages.STATUS, level="info", text="hello"),
                        messages.make(messages.MODE, mode="dictation"),
                        messages.make(messages.TARGET_SEARCHING, label="Send"),
                        messages.make(messages.TARGET_FOUND, x=1, y=2, label="Send"),
                        messages.make(messages.TARGET_NONE, label="Send"),
                        messages.make(messages.CONFIRM, id="q1", prompt="Send?", state="open", expires_in_s=12),
                        messages.make(messages.CONFIRM, id="q1", prompt="Send?", state="confirmed", expires_in_s=0),
                        messages.make(messages.DICTATION_PREVIEW, words=[{"text": "hi", "conf": 0.9}]),
                        messages.make(messages.DICTATION_PREVIEW, words=[]),
                        messages.make(messages.DICTATION_PREVIEW, words=[{"text": "Hi.", "conf": 1.0}],
                                      original="hi", cleaned=True),
                        messages.make(messages.CALIB_RESULT, error_px=30, ok=True),
                        messages.make(messages.LISTENING, level=0.2, speaking=False),     # not printed
                        messages.make(messages.GAZE, x=1, y=2, confidence=0.9)]:          # not printed
                bus.publish(topics.TO_UI, msg)
            bus.publish(topics.TRANSCRIPT, {"text": "hi there", "words": [], "ms": 1.0})
            bus.publish(topics.GESTURE, Gesture(0, "brow_raise"))
            bus.publish(topics.OS_CLICK, {"x": 1, "y": 2})
            bus.publish(topics.OS_TYPE, {"text": "hi "})
            bus.publish(topics.OS_KEY, {"key": "enter"})
            bus.publish(topics.OS_SCROLL, {"direction": "down"})
        lines = out.getvalue().splitlines()
        for tag in ("[info] hello", "[mode] dictation", "[ui-venus] looking for 'Send'", "[ui-venus] found 'Send'",
                    "[ui-venus] couldn't find 'Send'", "[question] Send? -> open", "[question] Send? -> confirmed",
                    "[preview] hi", "[preview] Hi.", "[calibrate] done", '[heard] "hi there"', "[gesture] brow_raise",
                    "[click] (1, 2)", "[type] 'hi '", "[key] enter", "[scroll] down"):
            self.assertTrue(any(line.startswith(tag) for line in lines), tag)
        self.assertEqual(sum(line.startswith("[preview]") for line in lines), 2)   # the empty one hides


if __name__ == "__main__":
    unittest.main()
