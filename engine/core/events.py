"""In-process event bus, plus the WebSocket server that relays messages to the UI.

Route: Khoi and Khoa (Khoa leads)
Publishes: everything the UI needs (see messages.py)
Subscribes: UI messages from the WebSocket
APIs: asyncio, websockets (serve on 127.0.0.1:8765), json

Handlers run on the publisher's thread, right away (the same as tests/helpers.FakeBus).
Keep them short: anything slow (Whisper, UI-Venus) puts work on its own queue and
returns, so the camera and audio threads are never blocked (CONTRIBUTING rule 9).
"""

from __future__ import annotations

import asyncio
import json
import threading
import traceback
from collections import defaultdict
from typing import Any, Callable

from engine.core import topics


class EventBus:
    """Thread-safe publish/subscribe. Camera, audio and UI-Venus threads publish here.

    publish() calls each handler right away on the publisher's thread, so handlers
    must be quick (queue the work, like DictationWorker.on_utterance does).
    A handler that raises is reported and skipped; it never breaks the publisher.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable[[Any], None]]] = defaultdict(list)
        self._lock = threading.Lock()

    def subscribe(self, topic: str, handler: Callable[[Any], None]) -> None:
        with self._lock:
            self._handlers[topic].append(handler)

    def unsubscribe(self, topic: str, handler: Callable[[Any], None]) -> None:
        with self._lock:
            if handler in self._handlers.get(topic, []):
                self._handlers[topic].remove(handler)

    def publish(self, topic: str, payload: Any) -> None:
        with self._lock:
            handlers = list(self._handlers.get(topic, ()))
        for handler in handlers:           # outside the lock: a handler may publish too
            try:
                handler(payload)
            except Exception:              # one broken module must not stop the others
                print(f"EventBus: handler for '{topic}' failed")
                traceback.print_exc()


# ------------------------------------------------------------------- JSON
def _jsonable(value):
    if hasattr(value, "tolist"):           # numpy scalars (a gaze x as np.float32) and arrays
        return value.tolist()
    raise TypeError(f"{type(value).__name__} can't be sent to the UI")


def encode(msg: dict) -> str:
    return json.dumps(msg, default=_jsonable, separators=(",", ":"))


def decode(text) -> dict | None:
    """A UI message, or None if it isn't a JSON object with a string "type"."""
    try:
        msg = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(msg, dict) or not isinstance(msg.get("type"), str):
        return None
    return msg


# -------------------------------------------------------------- WebSocket
STICKY = {"mode"}   # a UI window that connects later still gets the latest of these


class WebSocketRelay:
    """Bus topics.TO_UI -> every connected UI window; UI messages -> bus from_ui(type)."""

    def __init__(self, bus: EventBus) -> None:
        self.bus = bus
        self.clients: set = set()
        self._latest: dict[str, str] = {}   # STICKY type -> last JSON sent
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server = None

    async def start(self, host: str, port: int, origins: list[str] | None = None) -> None:
        """origins: the web addresses allowed to connect (None = anyone). A program that sends no Origin header is
        always allowed; a web page on another site is refused, so it can't read the camera picture or answer for you."""
        from websockets.asyncio.server import serve  # imported here so tests run without it

        self._loop = asyncio.get_running_loop()
        self._server = await serve(self._handle, host, port,
                                   origins=None if origins is None else [*origins, None])
        self.bus.subscribe(topics.TO_UI, self.send)

    async def close(self) -> None:
        self.bus.unsubscribe(topics.TO_UI, self.send)
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    @property
    def port(self) -> int:
        """The port actually used (useful when started on port 0 in tests)."""
        return self._server.sockets[0].getsockname()[1]

    async def serve_forever(self) -> None:
        await self._server.serve_forever()

    def send(self, msg: dict) -> None:
        """Called on any thread. The actual send happens on the asyncio loop."""
        try:
            text = encode(msg)
        except TypeError as exc:
            print(f"WebSocket: dropped a '{msg.get('type')}' message: {exc}")
            return
        if msg.get("type") in STICKY:
            self._latest[msg["type"]] = text
        try:
            self._loop.call_soon_threadsafe(self._broadcast, text)
        except RuntimeError:               # the loop already closed during shutdown
            pass

    def _broadcast(self, text: str) -> None:
        from websockets.asyncio.server import broadcast

        broadcast(self.clients, text)

    async def _handle(self, connection) -> None:
        self.clients.add(connection)
        try:
            for text in list(self._latest.values()):
                await connection.send(text)
            async for text in connection:
                msg = decode(text)
                if msg is None:
                    print(f"WebSocket: ignored a message that isn't {{\"type\": ...}}: {text!r:.80}")
                    continue
                self.bus.publish(topics.from_ui(msg["type"]), msg)
        except Exception:                  # a UI window closing mid-message is normal
            pass
        finally:
            self.clients.discard(connection)


async def serve_websocket(bus: EventBus, host: str, port: int) -> None:
    """Send engine messages to every connected UI window; put UI messages on the bus."""
    relay = WebSocketRelay(bus)
    await relay.start(host, port)
    try:
        await relay.serve_forever()
    finally:
        await relay.close()
