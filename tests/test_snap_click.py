"""The snap click and the mouth right click, with no camera or model: MouthOpenDetector on made-up jaw
curves, ScreenTarget.find in snap mode with a fake model, TargetService's "already on a link" answer,
the mode manager's snap flow and the new voice commands, and the eyes service end to end.

Run with: python -m unittest tests.test_snap_click
"""
import math
import unittest
from types import SimpleNamespace

from engine.core import messages, topics
from engine.core.mode_manager import SNAP_TIMEOUT_S, Mode
from engine.core.types import TargetRequest, TargetResult
from engine.eyes.clickable import Over
from engine.eyes.mouth_gesture import MouthOpenDetector
from engine.eyes.screen_target import SNAP_TARGET, Answer, ScreenTarget
from engine.eyes.target_service import TargetService
from tests.helpers import FakeBus
from tests.test_dwell_click import ButtonOs
from tests.test_eyes_service import Rig, face
from tests.test_mode_manager import Base

FRAME_MS = 33


def jaw_feed(d, levels, t0=0):
    """jawOpen per frame, 33 ms apart; returns the frame indexes where it fired."""
    out = []
    for i, j in enumerate(levels):
        if d.update(t0 + i * FRAME_MS, {"jawOpen": j}) is not None:
            out.append(i)
    return out


def closed(frames):
    return [0.05] * frames


def held_open(ms, level=0.65):
    return [level] * round(ms / FRAME_MS)


class TestMouthOpen(unittest.TestCase):
    def test_open_and_close_fires_once_on_the_close(self):
        levels = closed(30) + held_open(500) + closed(30)
        self.assertEqual(jaw_feed(MouthOpenDetector(), levels), [30 + len(held_open(500))])

    def test_a_yawn_never_fires(self):
        self.assertEqual(jaw_feed(MouthOpenDetector(), closed(30) + held_open(3000) + closed(30)), [])

    def test_talking_never_fires(self):
        syllables = closed(30)
        for _ in range(15):                                  # 3 s of 0.1 s open, 0.1 s closed
            syllables += [0.5] * 3 + [0.05] * 3
        self.assertEqual(jaw_feed(MouthOpenDetector(), syllables), [])

    def test_fast_talk_hovering_half_open_never_fires(self):
        hover = [0.32 + 0.1 * math.sin(i * 1.3) for i in range(30)]   # 1 s between about 0.22 and 0.42
        self.assertEqual(jaw_feed(MouthOpenDetector(), closed(30) + hover + closed(30)), [])

    def test_too_short_does_not_fire(self):
        self.assertEqual(jaw_feed(MouthOpenDetector(), closed(30) + held_open(200) + closed(30)), [])

    def test_held_while_open_for_the_cursor_freeze(self):
        d = MouthOpenDetector()
        jaw_feed(d, closed(30) + held_open(300))
        self.assertTrue(d.held)
        jaw_feed(d, closed(3), t0=(30 + 9) * FRAME_MS)
        self.assertFalse(d.held)

    def test_a_mouth_resting_open_still_works(self):
        levels = [0.2] * 30 + held_open(500, level=0.75) + [0.2] * 30
        self.assertEqual(len(jaw_feed(MouthOpenDetector(), levels)), 1)


class FakeModel(ScreenTarget):
    """ScreenTarget with the model swapped for a fixed answer, in 0-1000 of the image it was shown."""

    def __init__(self, answer="[500,500]", **kw):
        super().__init__(model_id="fake", device="cpu", crop=(1280, 720), **kw)
        self.answer, self.asked = answer, []

    def ask(self, image, instruction):
        from engine.eyes.screen_target import parse_point
        self.asked.append((image.size, instruction))
        return Answer(raw=self.answer, point=parse_point(self.answer), ms=100.0)


def screen():
    from PIL import Image
    return Image.new("RGB", (1920, 1080), (255, 255, 255))


class TestScreenTargetSnap(unittest.TestCase):
    def test_snap_asks_the_zoom_box_only_for_the_nearest_clickable(self):
        m = FakeModel("[550,520]")
        r = m.find(TargetRequest(800, 500, "here", snap=True), screen=screen(), origin=(0, 0))
        self.assertEqual(len(m.asked), 1)
        self.assertEqual(m.asked[0][0], (1280, 720))                       # 640x360 enlarged 2x
        self.assertTrue(m.asked[0][1].startswith(SNAP_TARGET))
        self.assertTrue(r.found)
        self.assertEqual((r.x, r.y), (480 + 352, 320 + 187))                # 0.55 * 640, 0.52 * 360

    def test_snap_too_far_from_the_pointer_is_not_found(self):
        m = FakeModel("[950,500]")                                         # 288 px right of the pointer
        r = m.find(TargetRequest(800, 500, "here", snap=True), screen=screen(), origin=(0, 0))
        self.assertFalse(r.found)
        self.assertEqual(len(m.asked), 1)                                  # and no wide box after it

    def test_named_targets_still_try_the_wide_box(self):
        m = FakeModel("[-1,-1]")
        r = m.find(TargetRequest(800, 500, "Send"), screen=screen(), origin=(0, 0))
        self.assertFalse(r.found)
        self.assertEqual(len(m.asked), 2)


class FakeFinder:
    def __init__(self):
        self.asked = []

    def submit(self, req):
        from concurrent.futures import Future
        self.asked.append(req)
        f = Future()
        f.set_result(TargetResult(req.label, True, req.gaze_x + 40, req.gaze_y))
        return f


class FakeProbe:
    def __init__(self, over):
        self.over = over

    def over_at(self, x, y, now=None):
        return self.over


class TestTargetServiceSnap(unittest.TestCase):
    def service(self, over, trust=True):
        bus, finder = FakeBus(), FakeFinder()
        ts = TargetService(bus, SimpleNamespace(use_ui_venus=True, snap_trust_cursor=trust), finder=finder,
                           probe=FakeProbe(over), log=lambda s: None)
        return bus, finder, ts

    def test_already_on_a_link_clicks_there_without_the_model(self):
        bus, finder, _ = self.service(Over(True))
        bus.publish(topics.TARGET_REQUEST, TargetRequest(800, 500, "here", snap=True))
        self.assertEqual(finder.asked, [])
        r = bus.of(topics.TARGET_RESULT)[-1]
        self.assertEqual((r.found, r.x, r.y), (True, 800, 500))

    def test_on_a_control_edge_clicks_inside_it(self):
        bus, finder, _ = self.service(Over(True, (700, 450, 790, 490)))
        bus.publish(topics.TARGET_REQUEST, TargetRequest(800, 500, "here", snap=True))
        r = bus.of(topics.TARGET_RESULT)[-1]
        self.assertEqual((r.x, r.y), (788, 488))

    def test_off_target_asks_ui_venus(self):
        for over in (Over(False), None):
            with self.subTest(over=over):
                bus, finder, _ = self.service(over)
                bus.publish(topics.TARGET_REQUEST, TargetRequest(800, 500, "here", snap=True))
                self.assertEqual(len(finder.asked), 1)
                self.assertEqual(bus.of(topics.TARGET_RESULT)[-1].x, 840)

    def test_named_requests_always_ask_ui_venus(self):
        bus, finder, _ = self.service(Over(True))
        bus.publish(topics.TARGET_REQUEST, TargetRequest(800, 500, "Send"))
        self.assertEqual(len(finder.asked), 1)

    def test_trust_cursor_off_always_asks(self):
        bus, finder, _ = self.service(Over(True), trust=False)
        bus.publish(topics.TARGET_REQUEST, TargetRequest(800, 500, "here", snap=True))
        self.assertEqual(len(finder.asked), 1)


class TestModeManagerSnap(Base):
    def requests(self):
        return self.bus.of(topics.TARGET_REQUEST)

    def test_click_words_snap_click_with_nothing_waiting(self):
        for text in ["click", "click here", "click this video", "Click it baby"]:
            with self.subTest(text=text):
                self.setUp()
                self.look(500, 300)
                self.say(text)
                req = self.requests()[-1]
                self.assertTrue(req.snap)
                self.bus.publish(topics.TARGET_RESULT, TargetResult(req.label, True, 520, 310))
                self.assertEqual(self.os(), [(topics.OS_CLICK, {"x": 520, "y": 310})])
                found = self.bus.ui(messages.TARGET_FOUND)[-1]
                self.assertTrue(found["snap"])

    def test_click_it_still_answers_a_question(self):
        self.look()
        self.say("click send")
        req = self.requests()[-1]
        self.bus.publish(topics.TARGET_RESULT, TargetResult(req.label, True, 410, 220))
        self.say("click it baby")                            # bug 3: was a new search for "It Baby"
        self.assertEqual(self.os(), [(topics.OS_CLICK, {"x": 410, "y": 220})])
        self.assertEqual(len(self.requests()), 1)

    def test_nothing_found_clicks_where_it_pointed(self):
        self.look(500, 300)
        self.gesture("brow_raise")
        req = self.requests()[-1]
        self.look(700, 300)                                  # the head moved on meanwhile
        self.bus.publish(topics.TARGET_RESULT, TargetResult(req.label, False))
        self.assertEqual(self.os(), [(topics.OS_CLICK, {"x": 500, "y": 300})])

    def test_no_answer_in_time_does_not_click(self):
        self.look()
        self.gesture("brow_raise")
        req = self.requests()[-1]
        self.clock.t += SNAP_TIMEOUT_S + 0.1
        self.mm.tick()
        self.bus.publish(topics.TARGET_RESULT, TargetResult(req.label, True, 520, 310))   # too late
        self.assertEqual(self.os(), [])
        self.assertIn("didn't answer in time", self.bus.ui(messages.STATUS)[-1]["text"])

    def test_a_second_raise_while_snapping_clicks_once(self):
        self.look()
        self.gesture("brow_raise")
        self.gesture("brow_raise")
        req = self.requests()[-1]
        self.bus.publish(topics.TARGET_RESULT, TargetResult(req.label, True, 520, 310))
        self.assertEqual(len(self.requests()), 1)
        self.assertEqual(len(self.os()), 1)

    def test_bare_click_while_dictating_is_text_but_click_this_clicks(self):
        self.say("start dictation")
        self.say("click")
        self.assertEqual(self.mm._preview_text(), "click")
        self.assertEqual(self.requests(), [])
        self.mm._discard_preview()
        self.look()
        self.say("click this video")
        self.assertTrue(self.requests()[-1].snap)
        self.assertEqual(self.mm.mode, Mode.DICTATION)

    def test_go_back_and_escape(self):
        self.say("go back")
        self.say("escape")
        self.assertEqual(self.os(), [(topics.OS_KEY, {"key": "alt+left"}), (topics.OS_KEY, {"key": "esc"})])


class TestMeterSnap(unittest.TestCase):
    def test_spinner_waits_for_the_screenshot_then_flashes_on_the_click(self):
        from engine.eyes.dwell_meter import SEARCH_SHOW_S, DwellMeter
        from tests.test_dwell_clickable import Clock, FakeBus as QuietBus
        clock = Clock()
        m = DwellMeter(QuietBus(), read_cursor=lambda: None, clock=clock)
        m.on_ui(messages.make(messages.TARGET_SEARCHING, label="here", snap=True))
        self.assertEqual(m.state()[0], "hidden")             # UI-Venus is taking its screenshot
        clock.t += SEARCH_SHOW_S + 0.05
        self.assertEqual(m.state()[0], "searching")
        m.on_ui(messages.make(messages.TARGET_FOUND, x=1, y=2, label="here", snap=True))
        self.assertEqual(m.state()[0], "done")
        clock.t += 0.3
        self.assertEqual(m.state()[0], "hidden")

    def test_named_target_found_does_not_flash(self):
        from engine.eyes.dwell_meter import DwellMeter
        from tests.test_dwell_clickable import Clock, FakeBus as QuietBus
        m = DwellMeter(QuietBus(), read_cursor=lambda: None, clock=Clock())
        m.on_ui(messages.make(messages.TARGET_SEARCHING, label="Send"))
        m.on_ui(messages.make(messages.TARGET_FOUND, x=1, y=2, label="Send"))
        self.assertEqual(m.state()[0], "hidden")            # it asks "Click Send?" first


def jaw_face(t_ms, jaw, yaw=0.0):
    f = face(t_ms, yaw=yaw)
    f.blendshapes["jawOpen"] = jaw
    return f


class TestEyesServiceMouth(unittest.TestCase):
    def setUp(self):
        self.r = Rig()
        self.r.os = self.r.out.os = ButtonOs()
        self.r.frames(60)   # the one-look calibration

    def tearDown(self):
        self.r.out.stop()

    def jaw_frames(self, n, jaw, yaw=0.0):
        r = self.r
        for _ in range(n):
            r.t_ms += 33
            r.clock.t += 0.033
            r.eyes.process(jaw_face(r.t_ms, jaw, r.sit[0] + yaw), r.clock.t)

    def test_open_and_close_right_clicks_where_the_mouth_opened(self):
        self.jaw_frames(20, 0.05, yaw=4.0)                   # point, mouth closed
        self.jaw_frames(10, 0.05, yaw=4.0)
        start = self.r.bus.of(topics.GAZE)[-1]
        self.jaw_frames(15, 0.7, yaw=7.0)                    # open about 0.5 s; the head tips with the jaw
        self.jaw_frames(5, 0.05, yaw=7.0)
        self.r.out.flush()
        clicks = [c for c in self.r.os.clicks if len(c) == 3]
        self.assertEqual(len(clicks), 1)
        x, y, button = clicks[0]
        self.assertEqual(button, "right")
        self.assertLess(math.hypot(x - start.x, y - start.y), 15)   # the frozen spot, not where the head went


if __name__ == "__main__":
    unittest.main()
