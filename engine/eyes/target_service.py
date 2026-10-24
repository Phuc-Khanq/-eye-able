"""UI-Venus in the engine: answers the mode manager's TargetRequest with a TargetResult.

Route: Khang and Hung (Khang leads)
Publishes: topics.TARGET_RESULT (TargetResult), status messages to the UI
Subscribes: topics.TARGET_REQUEST (TargetRequest: the cursor point and the spoken label)
APIs: ScreenTarget (UI-Venus 1.5 2B on the GPU, mss screenshots in memory) on its own worker thread,
      so the bus, camera and audio threads never wait for the model.

The model loads in the background when the engine starts (about 9 s on the demo PC, read from disk,
no download). A request that comes while it loads waits in line behind the load; the mode manager gives
up after 6 s and says it couldn't find the target, so the first request may need repeating.
With use_ui_venus off (config.local.json), every request is answered "not found" right away.
"""
from __future__ import annotations

from engine.config import settings as default_settings
from engine.core import messages, topics
from engine.core.types import TargetRequest, TargetResult


class TargetService:
    def __init__(self, bus, settings=default_settings, finder=None) -> None:
        self.bus = bus
        self.enabled = bool(getattr(settings, "use_ui_venus", True))
        self.finder = finder
        self.ready = False
        self.failed = ""
        bus.subscribe(topics.TARGET_REQUEST, self.on_request)

    def start(self) -> None:
        if not self.enabled:
            self._status("warn", "UI-Venus is off (use_ui_venus in config.local.json); look-and-say can't find targets")
            return
        if self.finder is None:
            from engine.eyes.screen_target import ScreenTarget
            self.finder = ScreenTarget()
        self._status("info", "Loading UI-Venus...")
        self.finder.preload().add_done_callback(self._loaded)

    def _loaded(self, fut) -> None:
        try:
            secs = fut.result()
        except Exception as exc:
            self.failed = f"{type(exc).__name__}: {exc}"
            self._status("error", f"UI-Venus didn't load: {self.failed}")
            return
        self.ready = True
        self._status("info", f"UI-Venus ready ({secs:.0f} s)")

    def on_request(self, req: TargetRequest) -> None:
        if not self.enabled or self.failed or self.finder is None:
            self.bus.publish(topics.TARGET_RESULT, TargetResult(label=req.label, found=False))
            return
        if not self.ready:
            self._status("info", "UI-Venus is still loading; this one may take a few seconds")
        self.finder.submit(req).add_done_callback(lambda fut: self._answer(req, fut))

    def _answer(self, req: TargetRequest, fut) -> None:
        try:
            result = fut.result()
        except Exception as exc:   # a bad screenshot or a model error must not stop the engine
            self._status("error", f"UI-Venus failed: {type(exc).__name__}: {exc}")
            result = TargetResult(label=req.label, found=False)
        self.bus.publish(topics.TARGET_RESULT, result)

    def stop(self) -> None:
        if self.finder is not None:
            self.finder.close()

    def _status(self, level: str, text: str) -> None:
        self.bus.publish(topics.TO_UI, messages.make(messages.STATUS, level=level, text=text))
