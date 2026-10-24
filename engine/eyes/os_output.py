"""Carries out the mode manager's actions on Windows: click, type, keys, scroll, undo.

Route: Khang and Hung (Hung leads os_input; this is the bus side of it)
Publishes: nothing
Subscribes: topics.OS_CLICK {"x", "y", "button": "left" | "right" (optional, left)}, OS_TYPE {"text"}, OS_KEY {"key": "enter" | "shift+enter" | ...},
            OS_SCROLL {"direction": "up" | "down"}, OS_UNDO {}
APIs: os_input (pynput). Only the mode manager publishes these topics, so nothing else can click or type.

Clicks go through the eyes service's CursorDriver when there is one: it snaps the cursor to the spot and
clicks there, so the 120 Hz glide can't pull the cursor away mid-click. Every action runs in order on one
worker thread, so typing a long sentence never stalls the camera or the bus (CONTRIBUTING rule 9).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partial

from engine.core import messages, topics

SCROLL_NOTCHES = 5   # one "scroll down" or tilt moves about a third of a page in a browser


class OsOutput:
    def __init__(self, bus, driver=None, os_api=None) -> None:
        if os_api is None:
            from engine.eyes import os_input as os_api
        self.bus, self.driver, self.os = bus, driver, os_api
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="os-output")
        bus.subscribe(topics.OS_CLICK, self.on_click)
        bus.subscribe(topics.OS_TYPE, self.on_type)
        bus.subscribe(topics.OS_KEY, self.on_key)
        bus.subscribe(topics.OS_SCROLL, self.on_scroll)
        bus.subscribe(topics.OS_UNDO, self.on_undo)

    def on_click(self, p: dict) -> None:
        x, y = int(p["x"]), int(p["y"])
        click = partial(self.os.click, button="right") if p.get("button") == "right" else self.os.click
        self._safely("click", lambda: self.driver.click(x, y, click) if self.driver else click(x, y))

    def on_type(self, p: dict) -> None:
        self._safely("type", lambda: self.os.type_text(p.get("text", "")))

    def on_key(self, p: dict) -> None:
        self._safely("key", lambda: self.os.combo(p.get("key", "")))

    def on_scroll(self, p: dict) -> None:
        notches = SCROLL_NOTCHES if p.get("direction") == "up" else -SCROLL_NOTCHES
        self._safely("scroll", lambda: self.os.scroll(notches))

    def on_undo(self, p: dict) -> None:
        self._safely("undo", lambda: self.os.combo("ctrl+z"))

    def _safely(self, what: str, action) -> None:
        def run() -> None:
            try:
                action()
            except Exception as exc:   # say what failed; never take the bus down
                self.bus.publish(topics.TO_UI, messages.make(
                    messages.STATUS, level="error", text=f"Couldn't {what}: {type(exc).__name__}: {exc}"))
        self._pool.submit(run)

    def flush(self, timeout: float = 5.0) -> None:
        """Wait until every queued action is done (tests, and before stopping)."""
        self._pool.submit(lambda: None).result(timeout=timeout)

    def stop(self) -> None:
        self._pool.shutdown(wait=True)
