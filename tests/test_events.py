"""Run with: python -m unittest tests.test_events"""
import asyncio
import importlib.util
import threading
import unittest
import unittest.mock
from contextlib import redirect_stdout
from io import StringIO

import numpy as np

from engine.core import messages, topics
from engine.core.events import EventBus, WebSocketRelay, decode, encode


class TestEventBus(unittest.TestCase):
    def test_publish_reaches_every_subscriber_in_order(self):
        bus, got = EventBus(), []
        bus.subscribe("a", lambda p: got.append(("first", p)))
        bus.subscribe("a", lambda p: got.append(("second", p)))
        bus.subscribe("b", lambda p: got.append(("other", p)))
        bus.publish("a", 1)
        self.assertEqual(got, [("first", 1), ("second", 1)])

    def test_no_subscribers_is_fine(self):
        EventBus().publish("nobody", {})

    def test_broken_handler_does_not_stop_the_others(self):
        bus, got = EventBus(), []

        def broken(_):
            raise ValueError("boom")

        bus.subscribe("a", broken)
        bus.subscribe("a", got.append)
        with redirect_stdout(StringIO()), unittest.mock.patch("traceback.print_exc"):
            bus.publish("a", 7)
        self.assertEqual(got, [7])

    def test_unsubscribe(self):
        bus, got = EventBus(), []
        bus.subscribe("a", got.append)
        bus.unsubscribe("a", got.append)
        bus.unsubscribe("a", got.append)        # twice is harmless
        bus.publish("a", 1)
        self.assertEqual(got, [])

    def test_handler_can_publish(self):
        bus, got = EventBus(), []
        bus.subscribe("a", lambda p: bus.publish("b", p + 1))
        bus.subscribe("b", got.append)
        bus.publish("a", 1)
        self.assertEqual(got, [2])

    def test_many_threads(self):
        bus, lock, count = EventBus(), threading.Lock(), [0]

        def add(_):
            with lock:
                count[0] += 1

        bus.subscribe("a", add)

        def work():
            for _ in range(500):
                bus.publish("a", None)
                bus.subscribe("x", add)       # subscribing while others publish

        threads = [threading.Thread(target=work) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(count[0], 8 * 500)


class TestJson(unittest.TestCase):
    def test_numpy_values_become_plain_numbers(self):
        msg = messages.make(messages.GAZE, x=np.float32(812.5), y=np.int64(440),
                            confidence=0.9, pts=np.array([1, 2]))
        self.assertEqual(decode(encode(msg)),
                         {"type": "gaze", "x": 812.5, "y": 440, "confidence": 0.9, "pts": [1, 2]})

    def test_decode_rejects_bad_messages(self):
        for bad in ["not json", "[1, 2]", '{"no": "type"}', '{"type": 5}', None]:
            self.assertIsNone(decode(bad), bad)
        self.assertEqual(decode('{"type": "set_mode", "mode": "paused"}'),
                         {"type": "set_mode", "mode": "paused"})


@unittest.skipIf(importlib.util.find_spec("websockets") is None, "websockets not installed")
class TestWebSocketLoopback(unittest.TestCase):
    def test_engine_to_ui_and_back(self):
        bus, from_ui = EventBus(), []
        bus.subscribe(topics.from_ui(messages.SET_MODE), from_ui.append)

        async def run():
            from websockets.asyncio.client import connect

            relay = WebSocketRelay(bus)
            await relay.start("127.0.0.1", 0)
            try:
                async with connect(f"ws://127.0.0.1:{relay.port}") as ui:
                    while not relay.clients:            # wait until the server sees us
                        await asyncio.sleep(0.01)
                    # A camera / audio thread publishes; the UI receives it.
                    t = threading.Thread(target=bus.publish, args=(
                        topics.TO_UI, messages.make(messages.MODE, mode="dictation")))
                    t.start()
                    t.join()
                    got = await asyncio.wait_for(ui.recv(), 2)
                    self.assertEqual(decode(got), {"type": "mode", "mode": "dictation"})

                    await ui.send("garbage")             # ignored, connection stays open
                    await ui.send(encode(messages.make(messages.SET_MODE, mode="paused")))
                    for _ in range(200):
                        if from_ui:
                            break
                        await asyncio.sleep(0.01)
            finally:
                await relay.close()
            bus.publish(topics.TO_UI, messages.make(messages.MODE, mode="menu"))  # after close: no error

        with redirect_stdout(StringIO()):                # the "ignored garbage" note
            asyncio.run(run())
        self.assertEqual(from_ui, [{"type": "set_mode", "mode": "paused"}])

    def test_a_late_ui_window_still_gets_the_current_mode(self):
        bus = EventBus()

        async def run():
            from websockets.asyncio.client import connect

            relay = WebSocketRelay(bus)
            await relay.start("127.0.0.1", 0)
            try:
                bus.publish(topics.TO_UI, messages.make(messages.MODE, mode="pointer"))
                bus.publish(topics.TO_UI, messages.make(messages.MODE, mode="paused"))
                bus.publish(topics.TO_UI, messages.make(messages.LISTENING, level=0.1, speaking=False))
                await asyncio.sleep(0.05)                # let the broadcasts run (no one listening)
                async with connect(f"ws://127.0.0.1:{relay.port}") as ui:
                    return decode(await asyncio.wait_for(ui.recv(), 2))
            finally:
                await relay.close()

        self.assertEqual(asyncio.run(run()), {"type": "mode", "mode": "paused"})


class TestRelayMoreCases(unittest.TestCase):
    """Several UI windows, a window that leaves, bursts, bad payloads, ports."""

    def relay_with(self, n_clients, body, bus=None):
        """Start a relay on a free port, connect n UI windows, run body(bus, relay, uis)."""
        bus = bus or EventBus()

        async def run():
            from websockets.asyncio.client import connect

            relay = WebSocketRelay(bus)
            await relay.start("127.0.0.1", 0)
            uis = []
            try:
                for _ in range(n_clients):
                    uis.append(await connect(f"ws://127.0.0.1:{relay.port}"))
                while len(relay.clients) < n_clients:       # wait until the server sees all
                    await asyncio.sleep(0.01)
                return await body(bus, relay, uis)
            finally:
                for ui in uis:
                    await ui.close()
                await relay.close()

        return asyncio.run(run())

    def test_every_window_gets_every_message_in_order(self):
        sent = [messages.make(messages.DWELL, progress=i / 10) for i in range(10)]

        async def body(bus, relay, uis):
            for m in sent:
                bus.publish(topics.TO_UI, m)
            return [[decode(await asyncio.wait_for(ui.recv(), 2)) for _ in sent] for ui in uis]

        a, b = self.relay_with(2, body)
        self.assertEqual(a, sent)
        self.assertEqual(b, sent)

    def test_a_window_that_leaves_does_not_break_the_others(self):
        async def body(bus, relay, uis):
            await uis[0].close()
            while len(relay.clients) > 1:
                await asyncio.sleep(0.01)
            bus.publish(topics.TO_UI, messages.make(messages.MODE, mode="menu"))
            return decode(await asyncio.wait_for(uis[1].recv(), 2))

        self.assertEqual(self.relay_with(2, body), {"type": "mode", "mode": "menu"})

    def test_a_burst_does_not_block_the_publisher_and_all_arrive(self):
        """500 gaze messages from a camera thread: publish returns at once, none are lost."""
        import time

        async def body(bus, relay, uis):
            def camera():
                t0 = time.perf_counter()
                for i in range(500):
                    bus.publish(topics.TO_UI, messages.make(messages.GAZE, x=i, y=i, confidence=1.0))
                camera.seconds = time.perf_counter() - t0

            t = threading.Thread(target=camera)
            t.start()
            t.join()
            xs = [decode(await asyncio.wait_for(uis[0].recv(), 2))["x"] for _ in range(500)]
            return camera.seconds, xs

        seconds, xs = self.relay_with(1, body)
        self.assertLess(seconds, 1.0)
        self.assertEqual(xs, list(range(500)))

    def test_a_payload_that_cant_be_json_is_dropped_and_the_next_one_goes(self):
        async def body(bus, relay, uis):
            with redirect_stdout(StringIO()) as out:
                bus.publish(topics.TO_UI, {"type": "status", "bad": object()})
            bus.publish(topics.TO_UI, messages.make(messages.STATUS, level="info", text="ok"))
            return out.getvalue(), decode(await asyncio.wait_for(uis[0].recv(), 2))

        note, msg = self.relay_with(1, body)
        self.assertIn("dropped", note)
        self.assertEqual(msg["text"], "ok")

    def test_a_taken_port_raises_and_stopping_frees_it(self):
        import socket

        from engine.core.events import serve_websocket

        async def run():
            first = WebSocketRelay(EventBus())
            await first.start("127.0.0.1", 0)
            port = first.port
            try:
                with self.assertRaises(OSError):            # what engine.main reports to the user
                    await WebSocketRelay(EventBus()).start("127.0.0.1", port)
            finally:
                await first.close()
            task = asyncio.create_task(serve_websocket(EventBus(), "127.0.0.1", port))
            await asyncio.sleep(0.2)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            return port

        port = asyncio.run(run())
        with socket.socket() as s:
            s.bind(("127.0.0.1", port))                     # free again after Ctrl+C


if __name__ == "__main__":
    unittest.main()
