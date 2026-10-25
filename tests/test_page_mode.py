"""The website's demo page as the place things happen: page mode (typing, keys, scrolling and "open ..." go to the page),
the webcam picture stream, the card and flash staying hidden, and the origin check on the WebSocket.

Run with: python -m unittest tests.test_page_mode
"""
import asyncio
import importlib.util
import unittest
from types import SimpleNamespace

import numpy as np

from engine.config import Settings
from engine.core import messages, topics
from engine.core.events import EventBus, WebSocketRelay
from engine.core.page_mode import PageMode
from engine.eyes.click_feedback import ClickFeedback
from engine.eyes.dictation_card import DictationCard
from engine.eyes.os_output import OsOutput
from engine.eyes.preview import PreviewStream, encode_frame
from tests.helpers import FakeBus
from tests.test_mode_manager import Base

S = Settings()


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class FakeOs:
    def __init__(self):
        self.calls = []

    def type_text(self, text):
        self.calls.append(("type", text))

    def combo(self, key):
        self.calls.append(("key", key))

    def scroll(self, n):
        self.calls.append(("scroll", n))

    def click(self, x, y, button="left"):
        self.calls.append(("click", x, y, button))


def rig(page_on: bool):
    bus, clock = FakeBus(), Clock()
    page = PageMode(bus, clock=clock)
    if page_on:
        bus.publish(topics.from_ui(messages.PAGE_MODE), {"type": "page_mode", "on": True})
    os_api, launched = FakeOs(), []
    out = OsOutput(bus, os_api=os_api, settings=S, launcher=lambda kind, target: launched.append((kind, target)),
                   page=page)
    return bus, clock, page, out, os_api, launched


class TestPageMode(unittest.TestCase):
    def test_on_for_a_few_seconds_after_the_last_message(self):
        bus, clock, page, *_ = rig(False)
        self.assertFalse(page.active)
        bus.publish(topics.from_ui(messages.PAGE_MODE), {"on": True})
        self.assertTrue(page.active)
        clock.t += 5.9
        self.assertTrue(page.active)
        clock.t += 0.2                      # the page stopped talking: a closed tab can't swallow typing
        self.assertFalse(page.active)

    def test_the_page_can_switch_it_off(self):
        bus, clock, page, *_ = rig(True)
        bus.publish(topics.from_ui(messages.PAGE_MODE), {"on": False})
        self.assertFalse(page.active)


class TestRouting(unittest.TestCase):
    def test_typing_keys_scroll_and_undo_go_to_the_page(self):
        bus, clock, page, out, os_api, launched = rig(True)
        bus.publish(topics.OS_TYPE, {"text": "see you at noon "})
        bus.publish(topics.OS_KEY, {"key": "enter"})
        bus.publish(topics.OS_KEY, {"key": "ctrl+t"})
        bus.publish(topics.OS_SCROLL, {"direction": "up"})
        bus.publish(topics.OS_UNDO, {})
        out.flush()
        self.assertEqual(os_api.calls, [])                     # nothing reached Windows
        acts = [(m["action"], m.get("text"), m.get("key"), m.get("direction")) for m in bus.ui(messages.PAGE_ACTION)]
        self.assertEqual(acts, [("type", "see you at noon ", None, None), ("key", None, "enter", None),
                                ("key", None, "ctrl+t", None), ("scroll", None, None, "up"),
                                ("undo", None, None, None)])

    def test_open_goes_to_the_page_and_never_starts_anything(self):
        bus, clock, page, out, os_api, launched = rig(True)
        bus.publish(topics.OS_LAUNCH, {"app": "chrome"})
        bus.publish(topics.OS_LAUNCH, {"url": "https://youtube.com"})
        bus.publish(topics.OS_LAUNCH, {"url": "http://evil.example"})
        out.flush()
        self.assertEqual(launched, [])
        msgs = bus.ui(messages.PAGE_ACTION)
        self.assertEqual([(m["action"], m.get("app"), m.get("url")) for m in msgs],
                         [("launch", "chrome", None), ("launch", None, "https://youtube.com")])
        self.assertEqual(bus.ui(messages.STATUS)[-1]["level"], "warn")        # http:// is still refused

    def test_clicks_stay_real_when_the_cursor_is_not_over_the_page(self):
        bus, clock, page, out, os_api, launched = rig(True)
        bus.publish(topics.OS_CLICK, {"x": 400, "y": 300})
        out.flush()
        self.assertEqual(os_api.calls, [("click", 400, 300, "left")])
        self.assertEqual(bus.ui(messages.PAGE_ACTION), [])

    def test_a_click_over_the_page_is_done_by_the_page(self):
        bus, clock, page, out, os_api, launched = rig(True)
        snaps = []
        out.driver = SimpleNamespace(snap=lambda x, y: snaps.append((x, y)))
        bus.publish(topics.from_ui(messages.PAGE_MODE), {"on": True, "inside": True})
        bus.publish(topics.OS_CLICK, {"x": 400, "y": 300})
        bus.publish(topics.OS_CLICK, {"x": 410, "y": 310, "button": "right"})
        out.flush()
        self.assertEqual(os_api.calls, [])                 # nothing went through Windows
        self.assertEqual(snaps, [(400, 300), (410, 310)])  # the real cursor still went there
        self.assertEqual([(m["action"], m["x"], m["y"], m["button"]) for m in bus.ui(messages.PAGE_ACTION)],
                         [("click", 400, 300, "left"), ("click", 410, 310, "right")])

    def test_the_cursor_leaving_the_page_makes_clicks_real_again(self):
        bus, clock, page, out, os_api, launched = rig(True)
        bus.publish(topics.from_ui(messages.PAGE_MODE), {"on": True, "inside": True})
        bus.publish(topics.from_ui(messages.PAGE_MODE), {"on": True, "inside": False})
        bus.publish(topics.OS_CLICK, {"x": 5, "y": 5})
        out.flush()
        self.assertEqual(os_api.calls, [("click", 5, 5, "left")])

    def test_inside_does_not_outlive_page_mode(self):
        bus, clock, page, out, os_api, launched = rig(True)
        bus.publish(topics.from_ui(messages.PAGE_MODE), {"on": True, "inside": True})
        clock.t += 7
        self.assertFalse(page.has_pointer)

    def test_volume_and_brightness_go_to_the_page(self):
        bus, clock, page, out, os_api, launched = rig(True)
        ran = []
        out.system = lambda action, os_api: ran.append(action) or "x"
        for name in ("volume_up", "volume_down", "mute", "brightness_up", "brightness_down"):
            bus.publish(topics.OS_SYSTEM, {"action": name})
        out.flush()
        self.assertEqual(ran, [])                        # the PC's own volume and screen are left alone
        self.assertEqual([m["name"] for m in bus.ui(messages.PAGE_ACTION) if m["action"] == "system"],
                         ["volume_up", "volume_down", "mute", "brightness_up", "brightness_down"])

    def test_volume_is_real_without_the_page(self):
        bus, clock, page, out, os_api, launched = rig(False)
        ran = []
        out.system = lambda action, os_api: ran.append(action) or "Volume up"
        bus.publish(topics.OS_SYSTEM, {"action": "volume_up"})
        out.flush()
        self.assertEqual(ran, ["volume_up"])

    def test_without_the_page_everything_is_real(self):
        bus, clock, page, out, os_api, launched = rig(False)
        bus.publish(topics.OS_TYPE, {"text": "hi "})
        bus.publish(topics.OS_KEY, {"key": "enter"})
        bus.publish(topics.OS_LAUNCH, {"app": "chrome"})
        out.flush()
        self.assertEqual(os_api.calls, [("type", "hi "), ("key", "enter")])
        self.assertEqual(launched, [("app", "chrome.exe")])
        self.assertEqual(bus.ui(messages.PAGE_ACTION), [])

    def test_when_the_page_goes_quiet_it_is_real_again(self):
        bus, clock, page, out, os_api, launched = rig(True)
        clock.t += 7
        bus.publish(topics.OS_TYPE, {"text": "hi "})
        out.flush()
        self.assertEqual(os_api.calls, [("type", "hi ")])


class TestOverlaysStayHidden(unittest.TestCase):
    def test_card_hidden_in_page_mode_only(self):
        bus, clock, page, *_ = rig(True)
        card = DictationCard(bus)
        card.page = page
        bus.publish(topics.TO_UI, messages.make(messages.COMMAND, label="Volume up"))
        self.assertEqual(card.view()[0].kind, "hidden")
        clock.t += 7
        self.assertNotEqual(card.view()[0].kind, "hidden")

    def test_flash_skipped_in_page_mode_sound_kept(self):
        bus, clock, page, *_ = rig(True)
        played = []
        fx = ClickFeedback(bus, clock=clock, player=played.append)
        fx.page = page
        bus.publish(topics.OS_CLICK, {"x": 10, "y": 20})
        self.assertIsNone(fx.flash.frame())
        self.assertEqual(played, ["left"])
        clock.t += 7
        bus.publish(topics.OS_CLICK, {"x": 10, "y": 20})
        self.assertIsNotNone(fx.flash.frame())


class TestPreview(unittest.TestCase):
    def frame(self):
        f = np.zeros((720, 1280, 3), dtype=np.uint8)
        f[:, :640] = (255, 0, 0)       # left half blue (BGR)
        return f

    def test_encode_shrinks_and_mirrors(self):
        import base64
        import cv2
        jpeg, w, h = encode_frame(self.frame(), 480, 65, True)
        self.assertEqual((w, h), (480, 270))
        img = cv2.imdecode(np.frombuffer(base64.b64decode(jpeg), np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(img.shape[:2], (270, 480))
        self.assertGreater(img[135, 400, 0], 200)      # the blue half is on the right after the mirror
        self.assertLess(img[135, 80, 0], 60)

    def test_nothing_is_sent_while_the_page_is_not_showing(self):
        bus, clock, page, *_ = rig(False)
        sent = []
        ps = PreviewStream(bus, S, page, clock=clock, encoder=lambda *a: sent.append(a) or ("x", 1, 1))
        ps.offer(self.frame())
        self.assertFalse(ps.send_latest())
        self.assertEqual(bus.ui(messages.DEBUG_FRAME), [])
        self.assertEqual(sent, [])

    def test_frames_are_rate_limited_and_arrive_as_debug_frame(self):
        bus, clock, page, *_ = rig(True)
        ps = PreviewStream(bus, SimpleNamespace(preview_fps=10, preview_width=480, preview_quality=65,
                                                preview_mirror=True), page, clock=clock)
        ps.offer(self.frame())
        ps.offer(self.frame())                 # same instant: skipped
        self.assertTrue(ps.send_latest())
        self.assertFalse(ps.send_latest())     # nothing newer
        clock.t += 0.2
        ps.offer(self.frame())
        self.assertTrue(ps.send_latest())
        got = bus.ui(messages.DEBUG_FRAME)
        self.assertEqual(len(got), 2)
        self.assertEqual({"jpeg", "w", "h"}, set(got[0]) - {"type"})

    def test_a_bad_frame_is_dropped_by_the_worker_not_raised(self):
        import time
        bus, clock, page, *_ = rig(True)

        def boom(*a):
            raise RuntimeError("bad")
        ps = PreviewStream(bus, S, page, clock=clock, encoder=boom)
        ps.offer(self.frame())
        ps.start()
        try:
            for _ in range(100):
                if ps._frame is None:                 # the worker took it (and the encoder raised)
                    break
                time.sleep(0.02)
            time.sleep(0.05)
            self.assertTrue(ps._thread.is_alive())    # still running
        finally:
            ps.stop()
        self.assertIsNone(ps._frame)
        self.assertEqual(bus.ui(messages.DEBUG_FRAME), [])


class TestCardButtons(Base):
    """The page's card buttons send the same dictation actions as a nod / shake / "use original"."""

    def test_ui_insert_and_discard(self):
        self.say("start dictation")
        self.say("see you at noon")
        self.bus.publish(topics.from_ui(messages.DICTATION), {"type": "dictation", "action": "insert"})
        self.assertEqual(self.os(), [(topics.OS_TYPE, {"text": "see you at noon "})])
        self.say("wrong words")
        self.bus.publish(topics.from_ui(messages.DICTATION), {"type": "dictation", "action": "discard"})
        self.assertEqual(self.mm.preview, [])

    def test_ui_use_original_is_harmless_without_a_clean_up(self):
        self.say("start dictation")
        self.say("see you at noon")
        self.bus.publish(topics.from_ui(messages.DICTATION), {"type": "dictation", "action": "use_original"})
        self.assertEqual(self.mm._preview_text(), "see you at noon")


class TestMessageContract(unittest.TestCase):
    def test_the_new_messages_are_in_the_contract(self):
        self.assertIn(messages.PAGE_ACTION, messages.FIELDS)
        self.assertIn(messages.DEBUG_FRAME, messages.FIELDS)
        self.assertEqual(messages.PAGE_MODE, "page_mode")


@unittest.skipIf(importlib.util.find_spec("websockets") is None, "websockets not installed")
class TestOriginCheck(unittest.TestCase):
    def connect(self, origins, origin):
        from websockets.asyncio.client import connect

        async def go():
            relay = WebSocketRelay(EventBus())
            await relay.start("127.0.0.1", 0, origins=origins)
            try:
                try:
                    async with connect(f"ws://127.0.0.1:{relay.port}", origin=origin, open_timeout=3):
                        return True
                except Exception:
                    return False
            finally:
                await relay.close()
        return asyncio.run(go())

    def test_only_the_demo_pages_may_connect(self):
        allowed = list(S.ws_allowed_origins)
        self.assertTrue(self.connect(allowed, "http://localhost:8000"))
        self.assertTrue(self.connect(allowed, None))                       # a program with no Origin (tests, tools)
        self.assertFalse(self.connect(allowed, "https://some-other-site.example"))
        self.assertFalse(self.connect(allowed, "null"))                    # a file:// page or a sandboxed frame

    def test_no_list_means_anyone_as_before(self):
        self.assertTrue(self.connect(None, "https://some-other-site.example"))


if __name__ == "__main__":
    unittest.main()
