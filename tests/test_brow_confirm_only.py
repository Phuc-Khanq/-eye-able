"""An eyebrow raise is for confirming, not clicking. A long blink is the click.

Run with: python -m unittest tests.test_brow_confirm_only
"""
import unittest
from types import SimpleNamespace

from engine.config import settings
from engine.core import messages, topics
from engine.core.mode_manager import Mode, ModeManager
from engine.core.types import GazePoint, Gesture, TargetResult
from tests.test_eyes_service import Rig
from tests.test_mode_manager import Base


class TestDefaults(unittest.TestCase):
    def test_the_raise_click_is_off_by_default(self):
        self.assertEqual(settings.brow_click, "off")
        self.assertTrue(settings.blink_click)


class TestModeManager(Base):
    def test_a_raise_with_nothing_waiting_does_nothing(self):
        self.look(500, 300)
        self.gesture("brow_raise")
        self.assertEqual(self.bus.of(topics.TARGET_REQUEST), [])
        self.assertEqual(self.os(), [])
        self.assertEqual(self.bus.ui(messages.GESTURE), [])        # not shown as acted on either

    def test_a_blink_with_nothing_waiting_clicks_the_exact_pixel(self):
        self.look(500, 300)
        self.gesture("blink")
        self.assertEqual(self.bus.of(topics.TARGET_REQUEST), [])
        self.assertEqual(self.os(), [(topics.OS_CLICK, {"x": 500, "y": 300})])

    def test_a_raise_still_answers_a_look_and_say_search(self):
        self.look()
        self.say("click send")
        req = self.bus.of(topics.TARGET_REQUEST)[-1]
        self.bus.publish(topics.TARGET_RESULT, TargetResult(req.label, True, 410, 220))
        self.gesture("brow_raise")
        self.assertEqual(self.os(), [(topics.OS_CLICK, {"x": 410, "y": 220})])

    def test_a_raise_still_answers_a_question_yes(self):
        self.say("start dictation")
        self.say("on my way")
        self.say("send it")
        self.assertTrue(self.mm.question)
        self.gesture("brow_raise")
        self.assertIsNone(self.mm.question)
        self.assertEqual(self.bus.ui(messages.CONFIRM)[-1]["state"], "confirmed")

    def test_a_raise_still_inserts_a_dictation_preview(self):
        self.say("start dictation")
        self.say("on my way")
        self.gesture("brow_raise")
        self.assertEqual(self.os(), [(topics.OS_TYPE, {"text": "on my way "})])

    def test_a_raise_still_leaves_scroll_mode(self):
        self.say("scroll")
        self.assertEqual(self.mm.mode, Mode.SCROLL)
        self.gesture("brow_raise")
        self.assertEqual(self.mm.mode, Mode.POINTER)

    def test_a_raise_does_nothing_in_dictation_with_no_preview(self):
        self.say("start dictation")
        self.gesture("brow_raise")
        self.assertEqual(self.os(), [])
        self.assertEqual(self.mm.mode, Mode.DICTATION)

    def test_the_old_click_can_be_switched_back_on(self):
        for side, button in (("left", None), ("right", "right")):
            with self.subTest(side=side):
                self.setUp()
                self.mm._brow_click = side
                self.look(500, 300)
                self.gesture("brow_raise")
                if side == "left":
                    self.assertEqual(self.os(), [(topics.OS_CLICK, {"x": 500, "y": 300})])   # the exact pixel
                else:
                    self.assertEqual(self.os()[-1][1].get("button"), "right")

    def test_the_setting_reaches_the_mode_manager(self):
        mm = ModeManager(self.bus, settings=SimpleNamespace(brow_click="left"), log=lambda s: None)
        self.assertEqual(mm._brow_click, "left")


class TestEyesService(unittest.TestCase):
    def setUp(self):
        self.r = Rig()
        self.r.frames(60)       # the one-look calibration

    def tearDown(self):
        self.r.out.stop()

    def test_a_raise_does_not_click_or_ask_ui_venus(self):
        r = self.r
        r.raise_brows()
        self.assertEqual(r.finder.asked, [])
        self.assertEqual(r.os.clicks, [])

    def test_a_raise_does_not_freeze_the_cursor(self):
        r = self.r
        self.assertFalse(r.eyes._face_held()[0])
        r.frames(12, yaw=6.0, brow=0.45)
        self.assertFalse(r.eyes._face_held()[0])

    def test_with_the_old_click_on_a_raise_freezes_it_again(self):
        r = self.r
        r.brow_click_on()
        r.frames(15)
        r.frames(8, yaw=0.0, brow=0.45)
        self.assertTrue(r.eyes._face_held()[0])


if __name__ == "__main__":
    unittest.main()
