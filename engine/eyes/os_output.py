"""Carries out the mode manager's actions on Windows: click, type, keys, scroll, undo, open an app or site.

Route: Khang and Hung (Hung leads os_input; this is the bus side of it)
Publishes: status (to the UI) when opening something, or when an action fails
Subscribes: topics.OS_CLICK {"x", "y", "button": "left" | "right" (optional, left)}, OS_TYPE {"text"}, OS_KEY {"key": "enter" | "shift+enter" | ...},
            OS_SCROLL {"direction": "up" | "down"}, OS_UNDO {},
            OS_LAUNCH {"app": a key of settings.launch_apps} or {"url": "https://..."},
            OS_SYSTEM {"action": "volume_up" | "volume_down" | "mute" | "brightness_up" | "brightness_down"}
APIs: os_input (pynput); os.startfile and webbrowser for OS_LAUNCH. Only the mode manager publishes these
topics, so nothing else can click or type. OS_LAUNCH only starts what settings.launch_apps lists (never a
name from the message itself, never a shell) and only opens https:// URLs.

Clicks go through the eyes service's CursorDriver when there is one: it snaps the cursor to the spot and
clicks there, so the 120 Hz glide can't pull the cursor away mid-click. Every action runs in order on one
worker thread, so typing a long sentence never stalls the camera or the bus (CONTRIBUTING rule 9).
With a scroller (the eyes service's SmoothScroller), OS_SCROLL is a smooth glide of 5 notches in 0.7 s
instead of one 5-notch jump.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partial

from engine.core import messages, topics

SCROLL_NOTCHES = 5   # one "scroll down" or tilt moves about a third of a page in a browser


def start(kind: str, target: str) -> None:
    """kind "app": os.startfile(target), which finds chrome.exe / msedge.exe through Windows App Paths and
    opens ms-settings: and mailto:. kind "url": the default browser. Windows only; no shell."""
    if kind == "url":
        import webbrowser

        webbrowser.open(target)
    else:
        import os

        os.startfile(target)


class OsOutput:
    def __init__(self, bus, driver=None, os_api=None, scroller=None, settings=None, launcher=None,
                 system=None) -> None:
        if os_api is None:
            from engine.eyes import os_input as os_api
        if settings is None:
            from engine.config import settings
        self.bus, self.driver, self.os, self.scroller = bus, driver, os_api, scroller
        self.settings, self.launcher = settings, launcher or start
        if system is None:
            from engine.eyes.system_controls import run_action as system
        self.system = system     # (action, os_api) -> status text; tests pass a fake
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="os-output")
        bus.subscribe(topics.OS_CLICK, self.on_click)
        bus.subscribe(topics.OS_TYPE, self.on_type)
        bus.subscribe(topics.OS_KEY, self.on_key)
        bus.subscribe(topics.OS_SCROLL, self.on_scroll)
        bus.subscribe(topics.OS_UNDO, self.on_undo)
        bus.subscribe(topics.OS_LAUNCH, self.on_launch)
        bus.subscribe(topics.OS_SYSTEM, self.on_system)

    def on_click(self, p: dict) -> None:
        x, y = int(p["x"]), int(p["y"])
        click = partial(self.os.click, button="right") if p.get("button") == "right" else self.os.click
        self._safely("click", lambda: self.driver.click(x, y, click) if self.driver else click(x, y))

    def on_type(self, p: dict) -> None:
        self._safely("type", lambda: self.os.type_text(p.get("text", "")))

    def on_key(self, p: dict) -> None:
        self._safely("key", lambda: self.os.combo(p.get("key", "")))

    def on_scroll(self, p: dict) -> None:
        if self.scroller is not None:
            self.scroller.glide("up" if p.get("direction") == "up" else "down", notches=SCROLL_NOTCHES)
            return
        notches = SCROLL_NOTCHES if p.get("direction") == "up" else -SCROLL_NOTCHES
        self._safely("scroll", lambda: self.os.scroll(notches))

    def on_undo(self, p: dict) -> None:
        self._safely("undo", lambda: self.os.combo("ctrl+z"))

    def on_system(self, p: dict) -> None:
        """Volume or brightness. A screen that can't be dimmed is a warning, not an error."""
        action = str(p.get("action", ""))

        def run() -> None:
            from engine.eyes.system_controls import BrightnessUnsupported
            try:
                self._status("info", self.system(action, self.os))
            except BrightnessUnsupported as exc:
                self._status("warn", f"Can't change the brightness: {exc}")
        self._safely(action.replace("_", " "), run)

    def on_launch(self, p: dict) -> None:
        """Start an app from settings.launch_apps, or open an https:// URL. Anything else: a warning only."""
        app, url = p.get("app"), p.get("url")
        if app is not None:
            apps = getattr(self.settings, "launch_apps", {}) or {}
            target = apps.get(app) if isinstance(app, str) else None
            if not target:
                self._status("warn", f"Not opening {app!r}: it isn't on the list of apps")
                return
            name = app.title()
            self._safely(f"open {name}", lambda: (self.launcher("app", target),
                                                  self._status("info", f"Opening {name}")))
        elif isinstance(url, str) and url.startswith("https://") and not any(c.isspace() for c in url):
            from urllib.parse import urlparse

            site = urlparse(url).netloc
            self._safely(f"open {site}", lambda: (self.launcher("url", url),
                                                  self._status("info", f"Opening {site}")))
        else:
            self._status("warn", f"Not opening {url!r}: only https:// sites")

    def _status(self, level: str, text: str) -> None:
        self.bus.publish(topics.TO_UI, messages.make(messages.STATUS, level=level, text=text))

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
