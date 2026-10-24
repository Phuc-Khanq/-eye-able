"""In-process event bus, plus the WebSocket server that relays messages to the UI.

Route: Khoi and Khoa (Khoa leads)
Publishes: everything the UI needs (see messages.py)
Subscribes: UI messages from the WebSocket
APIs: asyncio, websockets (serve on 127.0.0.1:8765), json
"""

from typing import Any, Callable


class EventBus:
    """Thread-safe publish/subscribe. Camera, audio and UI-Venus threads publish here."""

    def subscribe(self, topic: str, handler: Callable[[Any], None]) -> None:
        raise NotImplementedError

    def publish(self, topic: str, payload: Any) -> None:
        raise NotImplementedError


async def serve_websocket(bus: EventBus, host: str, port: int) -> None:
    """Send engine messages to every connected UI window; put UI messages on the bus."""
    raise NotImplementedError
