"""Is the website's demo page open and showing? While it is, the engine acts inside the page.

Route: Khang and Hung
Subscribes: the UI message page_mode ({"on": true, "inside": true}), which the demo page repeats every second, and
            sends at once when the cursor enters or leaves the page
Used by: os_output (typing, keys, scrolling and "open ..." go to the page; so do clicks while the cursor is over it), preview (webcam picture to the page),
         the dictation card and click flash (the page draws its own, so the desktop ones stay hidden)

The page being open is not enough: it has to keep saying so. If the tab closes or the browser dies, the messages stop
and page mode ends 6 seconds later, so a forgotten page can never swallow what you type into a real app.
"""
from __future__ import annotations

import time

from engine.core import messages, topics

EXPIRES_S = 6.0


class PageMode:
    def __init__(self, bus, clock=time.monotonic, expires_s: float = EXPIRES_S) -> None:
        self.clock, self.expires_s = clock, expires_s
        self._until = 0.0
        self.inside = False        # the real cursor is over the page (the page says so as soon as it changes)
        bus.subscribe(topics.from_ui(messages.PAGE_MODE), self.on_message)

    def on_message(self, msg: dict) -> None:
        self._until = self.clock() + self.expires_s if msg.get("on", True) else 0.0
        self.inside = bool(msg.get("inside", False))

    @property
    def active(self) -> bool:
        return self.clock() < self._until

    @property
    def has_pointer(self) -> bool:
        """Page mode is on and the real cursor is over the page: a click there is the page's to do."""
        return self.active and self.inside
