"""Run with: python -m unittest discover -s tests -t ."""
import unittest

from engine.core import messages, topics
from engine.core.mode_manager import CONFIRM_TIMEOUT_S, TARGET_TIMEOUT_S, Mode, ModeManager
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
        self.worker = DictationWorker(self.bus, Dictation(model=FakeWhisperModel()))
        self.mm = ModeManager(self.bus, clock=self.clock)

    def say(self, text):
        self.worker.process(text)   # same as an utterance, without the thread

    def gesture(self, kind):
        self.bus.publish(topics.GESTURE, Gesture(t_ms=0, type=kind))

    def look(self, x=500, y=300):
        self.bus.publish(topics.GAZE, GazePoint(t_ms=0, x=x, y=y, confidence=0.9))

    def os(self):
        return [(t, p) for t, p in self.bus.log if t.startswith("os.")]


class TestDictation(Base):
    def test_preview_has_word_confidence(self):
        self.say("start dictation")
        self.say("see you at noon? today")
        words = self.bus.ui(messages.DICTATION_PREVIEW)[-1]["words"]
        self.assertEqual([w["text"] for w in words], ["see", "you", "at", "noon", "today"])
        self.assertLess(words[3]["conf"], 0.5)

    def test_nod_inserts_shake_discards(self):
        self.say("start dictation")
        self.say("wrong words")
        self.gesture("shake")
        self.say("right words")
        self.gesture("nod")
        self.assertEqual(self.os(), [(topics.OS_TYPE, {"text": "right words "})])

    def test_utterances_join_the_bubble(self):
        self.say("start dictation")
        self.say("first part")
        self.say("second part")
        self.assertEqual(self.mm._preview_text(), "first part second part")

    def test_commands_never_show_as_preview(self):
        self.say("start dictation")
        self.say("send it")
        self.assertEqual(self.bus.ui(messages.DICTATION_PREVIEW), [])

    def test_target_words_are_text_in_dictation(self):
        self.say("start dictation")
        self.say("open the door when you get home")
        self.assertEqual(self.bus.of(topics.TARGET_REQUEST), [])
        self.assertEqual(len(self.mm.preview), 7)

    def test_ui_edit_word(self):
        self.say("start dictation")
        self.say("meet at nine? today")
        self.bus.publish(topics.from_ui(messages.DICTATION),
                         {"action": "edit_word", "index": 2, "text": "five"})
        self.bus.publish(topics.from_ui(messages.DICTATION), {"action": "insert"})
        self.assertEqual(self.os()[-1], (topics.OS_TYPE, {"text": "meet at five today "}))

    def test_stop_with_preview_asks(self):
        self.say("start dictation")
        self.say("keep this")
        self.say("stop dictation")
        self.assertEqual(self.mm.mode, Mode.DICTATION)
        self.gesture("nod")
        self.assertEqual(self.mm.mode, Mode.POINTER)
        self.assertEqual(self.os(), [(topics.OS_TYPE, {"text": "keep this "})])

    def test_new_line_is_shift_enter(self):
        self.say("start dictation")
        self.say("new line")
        self.assertEqual(self.os(), [(topics.OS_KEY, {"key": "shift+enter"})])


class TestConfirm(Base):
    def test_send_always_asks(self):
        self.say("send it")
        self.assertEqual(self.os(), [])
        self.assertEqual(self.bus.ui(messages.CONFIRM)[-1]["state"], "open")
        self.gesture("nod")
        self.assertEqual(self.os(), [(topics.OS_KEY, {"key": "enter"})])

    def test_send_inserts_preview_then_presses_enter(self):
        self.say("start dictation")
        self.say("on my way")
        self.say("send it")
        self.say("yes")
        self.assertEqual(self.os(), [(topics.OS_TYPE, {"text": "on my way "}),
                                     (topics.OS_KEY, {"key": "enter"})])

    def test_shake_cancels_send_and_keeps_preview(self):
        self.say("start dictation")
        self.say("not ready")
        self.say("send it")
        self.gesture("shake")
        self.assertEqual(self.os(), [])
        self.assertEqual(self.mm._preview_text(), "not ready")

    def test_question_expires(self):
        self.say("send it")
        self.clock.t += CONFIRM_TIMEOUT_S + 0.1
        self.gesture("nod")   # too late: expires first, then the nod has nothing to answer
        self.assertEqual(self.bus.ui(messages.CONFIRM)[-1]["state"], "expired")
        self.assertEqual(self.os(), [])

    def test_ui_button_answers(self):
        self.say("send it")
        qid = self.bus.ui(messages.CONFIRM)[-1]["id"]
        self.bus.publish(topics.from_ui(messages.CONFIRM_ANSWER), {"id": "wrong", "yes": True})
        self.assertEqual(self.os(), [])
        self.bus.publish(topics.from_ui(messages.CONFIRM_ANSWER), {"id": qid, "yes": True})
        self.assertEqual(self.os(), [(topics.OS_KEY, {"key": "enter"})])

    def test_delete_voice_data_asks(self):
        self.say("delete my voice data")
        self.assertEqual(self.bus.of(topics.DELETE_VOICE_DATA), [])
        self.gesture("nod")
        self.assertEqual(self.bus.of(topics.DELETE_VOICE_DATA), [{}])


class TestLookAndSay(Base):
    def test_found_then_nod_clicks(self):
        self.look(800, 450)
        self.say("click send")
        req = self.bus.of(topics.TARGET_REQUEST)[0]
        self.assertEqual((req.gaze_x, req.gaze_y, req.label), (800, 450, "Send"))
        self.bus.publish(topics.TARGET_RESULT, TargetResult("Send", True, 812, 440, 120.0))
        self.assertEqual(self.bus.ui(messages.TARGET_FOUND)[-1]["x"], 812)
        self.assertEqual(self.os(), [])           # not yet: needs a nod
        self.gesture("nod")
        self.assertEqual(self.os(), [(topics.OS_CLICK, {"x": 812, "y": 440})])

    def test_found_then_shake_cancels(self):
        self.look()
        self.say("click send")
        self.bus.publish(topics.TARGET_RESULT, TargetResult("Send", True, 1, 2))
        self.gesture("shake")
        self.assertEqual(self.os(), [])

    def test_ui_target_click(self):
        self.look()
        self.say("click send")
        self.bus.publish(topics.TARGET_RESULT, TargetResult("Send", True, 1, 2))
        self.bus.publish(topics.from_ui(messages.TARGET), {"action": "click"})
        self.assertEqual(self.os(), [(topics.OS_CLICK, {"x": 1, "y": 2})])

    def test_not_found(self):
        self.look()
        self.say("click send")
        self.bus.publish(topics.TARGET_RESULT, TargetResult("Send", False))
        self.assertEqual(self.bus.ui(messages.TARGET_NONE)[-1]["label"], "Send")
        self.assertIsNone(self.mm.question)

    def test_ui_venus_too_slow(self):
        self.look()
        self.say("click send")
        self.clock.t += TARGET_TIMEOUT_S + 0.1
        self.mm.tick()
        self.bus.publish(topics.TARGET_RESULT, TargetResult("Send", True, 1, 2))  # late
        self.assertIsNone(self.mm.question)
        self.assertEqual(len(self.bus.ui(messages.TARGET_NONE)), 1)

    def test_needs_a_gaze_first(self):
        self.say("click send")
        self.assertEqual(self.bus.of(topics.TARGET_REQUEST), [])
        self.assertEqual(self.bus.ui(messages.STATUS)[-1]["level"], "warn")


class TestModes(Base):
    def test_pointer_dwell_and_tilts(self):
        self.bus.publish(topics.DWELL_DONE, GazePoint(0, 100.6, 200.2, 0.9))
        self.gesture("tilt_right")
        self.assertEqual(self.os(), [(topics.OS_CLICK, {"x": 100, "y": 200}),
                                     (topics.OS_SCROLL, {"direction": "down"})])

    def test_no_dwell_click_while_dictating(self):
        self.say("start dictation")
        self.bus.publish(topics.DWELL_DONE, GazePoint(0, 1, 2, 0.9))
        self.assertEqual(self.os(), [])

    def test_paused_ignores_all_but_resume(self):
        self.say("start dictation")
        self.say("pause")
        self.say("scroll down")
        self.say("hello there")
        self.gesture("nod")
        self.bus.publish(topics.DWELL_DONE, GazePoint(0, 1, 2, 0.9))
        self.assertEqual(self.os(), [])
        self.say("resume")
        self.assertEqual(self.mm.mode, Mode.DICTATION)

    def test_menu(self):
        self.say("menu")
        self.assertEqual(self.mm.mode, Mode.MENU)
        self.gesture("tilt_left")
        self.assertEqual(self.bus.ui(messages.GESTURE)[-1]["gesture"], "tilt_left")
        self.bus.publish(topics.from_ui(messages.MENU_SELECT), {"item": "start_dictation"})
        self.assertEqual(self.mm.mode, Mode.DICTATION)

    def test_menu_shake_closes(self):
        self.say("start dictation")
        self.say("menu")
        self.gesture("shake")
        self.assertEqual(self.mm.mode, Mode.DICTATION)

    def test_mode_messages_for_ui(self):
        self.say("start dictation")
        self.assertEqual(self.bus.ui(messages.MODE)[-1], {"type": "mode", "mode": "dictation"})

    def test_whisper_error_becomes_status(self):
        class Broken:
            def transcribe(self, audio, **kw):
                raise RuntimeError("CUDA out of memory")

        bus = FakeBus()
        DictationWorker(bus, Dictation(model=Broken())).process("x")
        self.assertIn("CUDA out of memory", bus.ui(messages.STATUS)[0]["text"])


if __name__ == "__main__":
    unittest.main()
