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

Snap requests (TargetRequest.snap: a brow raise or "click here", no name): when the cursor is already on
something clickable (engine/eyes/clickable.py: the hand cursor or UI Automation), the answer is that
spot at once, no model call (snap_trust_cursor). Otherwise UI-Venus finds the nearest clickable thing
around the pointer. "Not found" makes the mode manager click at the pointer anyway.
"""
from __future__ import annotations

from engine.config import settings as default_settings
from engine.core import messages, topics
from engine.core.types import TargetRequest, TargetResult


class TargetService:
    def __init__(self, bus, settings=default_settings, finder=None, probe=None, log=print) -> None:
        self.bus = bus
        self.probe = probe                # clickable.ClickableProbe, or None (off Windows, tests)
        self.trust_cursor = bool(getattr(settings, "snap_trust_cursor", True))
        self._log = log
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
        if getattr(req, "snap", False) and self._on_clickable(req):
            return
        if not self.enabled or self.failed or self.finder is None:
            self.bus.publish(topics.TARGET_RESULT, TargetResult(label=req.label, found=False))
            return
        if not self.ready:
            self._status("info", "UI-Venus is still loading; this one may take a few seconds")
        self.finder.submit(req).add_done_callback(lambda fut: self._answer(req, fut))

    def _on_clickable(self, req: TargetRequest) -> bool:
        """Snap with the cursor already on a link or button: click right there (inside its rectangle)."""
        if self.probe is None or not self.trust_cursor:
            return False
        try:
            over = self.probe.over_at(req.gaze_x, req.gaze_y)
        except Exception:
            return False
        if over is None or not over.clickable:
            return False
        x, y = req.gaze_x, req.gaze_y
        if over.rect is not None:
            left, top, right, bottom = over.rect
            x, y = min(max(x, left + 2), right - 2), min(max(y, top + 2), bottom - 2)
        self._log(f"[snap] on a {'link/button (hand cursor)' if over.rect is None else 'control'} already: "
                  f"clicking at ({x}, {y})")
        self.bus.publish(topics.TARGET_RESULT, TargetResult(label=req.label, found=True, x=int(x), y=int(y)))
        return True

    def _answer(self, req: TargetRequest, fut) -> None:
        try:
            result = fut.result()
        except Exception as exc:   # a bad screenshot or a model error must not stop the engine
            self._status("error", f"UI-Venus failed: {type(exc).__name__}: {exc}")
            result = TargetResult(label=req.label, found=False)
        if getattr(req, "snap", False):
            if result.found:
                moved = ((result.x - req.gaze_x) ** 2 + (result.y - req.gaze_y) ** 2) ** 0.5
                self._log(f"[snap] UI-Venus moved the click {moved:.0f} px to ({result.x}, {result.y}) "
                          f"in {result.ms:.0f} ms")
            else:
                self._log(f"[snap] UI-Venus found nothing clickable near ({req.gaze_x}, {req.gaze_y}) "
                          f"in {result.ms:.0f} ms")
        self.bus.publish(topics.TARGET_RESULT, result)

    def stop(self) -> None:
        if self.finder is not None:
            self.finder.close()

    def _status(self, level: str, text: str) -> None:
        self.bus.publish(topics.TO_UI, messages.make(messages.STATUS, level=level, text=text))
