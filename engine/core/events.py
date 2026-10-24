"""In-process event bus, plus the WebSocket server that relays messages to the UI.

Route: Khoi and Khoa (Khoa leads)
Publishes: everything the UI needs (see messages.py)
Subscribes: UI messages from the WebSocket
APIs: asyncio, websockets (serve on 127.0.0.1:8765), json
"""

import threading
import traceback
from collections import defaultdict
from typing import Any, Callable


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

    def publish(self, topic: str, payload: Any) -> None:
        with self._lock:
            handlers = list(self._handlers.get(topic, ()))
        for handler in handlers:
            try:
                handler(payload)
            except Exception:
                print(f"EventBus: handler for '{topic}' failed")
                traceback.print_exc()


async def serve_websocket(bus: EventBus, host: str, port: int) -> None:
    """Send engine messages to every connected UI window; put UI messages on the bus."""
    raise NotImplementedError
