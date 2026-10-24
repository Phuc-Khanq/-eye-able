"""The eyes in the engine: camera -> face mesh -> head pose and eye features -> pointer -> the real cursor,
plus head gestures, the one-look calibration and the confirm nod.

Route: Khang and Hung (Khang leads)
Publishes: topics.GAZE (GazePoint, every frame), topics.GESTURE (Gesture), topics.DWELL_DONE (dwell on only),
           topics.MOUTH (MouthSample, every frame with a face),
           to the UI: gaze (x, y, confidence), dwell, calib_target, calib_result, status
Subscribes: topics.CALIBRATE and the UI's calibrate, the UI's settings; and, read only, the mode manager's
           messages to the UI (mode, target_found, target_none, confirm, dictation_preview), to know when
           to hold the cursor still and when a nod is expected
APIs: one camera thread (OpenCV, MediaPipe) plus CursorDriver's glide thread. Frames stay in memory;
      nothing is saved.

What the cursor does:
  pointer, dictation   follows the head (or gaze); wobble under 8 px is ignored, moves glide at 120 Hz
  target found         jumps onto the element and holds still until the question closes
  menu                 holds still: head moves drive the radial menu instead
  scroll               holds still; the head's up-down tilt sets the scroll speed (engine/eyes/scroll.py);
                       the mode manager says "scroll" and ends it ("stop", a raise, any other command)
  paused               the engine leaves the cursor alone
  calibrating          holds on the screen center while you look at the dot (about 1.5 s)
The first face the camera sees starts a calibration, so the head's neutral pose is wherever you face
the screen center. Say "calibrate" (or the UI's calibrate) to redo it after moving your chair.

Clicks: an eyebrow raise = left click when nothing is waiting (the mode manager asks UI-Venus to snap it
onto the button, link or box nearest the cursor); open the mouth and close it again (MouthOpenDetector,
Gesture "mouth_open") = right click; a long blink (BlinkDetector, Gesture "blink", blink_click / --blink) = the
same click as a raise, without moving the head; dwell (rest the cursor dwell_ms; off by default, --dwell) = left
click through DWELL_DONE.
From the moment a raise starts (BrowRaiseDetector.raised), or once the eyes have been shut (BlinkDetector.held)
or the mouth open (MouthOpenDetector.held) a moment, the cursor freezes where it was, so the face moving can't nudge the click off the
button; the published gaze is the frozen spot, which is where the mode manager clicks. When the face
relaxes the pointer carries on from there, no jump.
Dwell is off, and comes back disarmed at the cursor (move away first), outside Pointer mode, while a
question or preview waits, while UI-Venus searches or holds a found target, during a raise and right
after calibrating. Every click disarms it at the click spot. set_dwell(on) is the on/off switch.
On Windows only clickable things count (settings.dwell_clickable_only): engine/eyes/clickable.py says
what is under the cursor from a background thread, and a stop on a button or link gets a looser radius.
The dwell message to the UI carries ready (resting on something clickable, dwell armed) besides the
progress, so the ring can show "this clicks" before it fills; it is sent only when either changes.

Yes: an eyebrow raise (Hung's BrowRaiseDetector, Gesture "brow_raise") is the default, because a nod
moves the head pointer and with it the cursor. confirm="nod" brings the nod back: Hung's strict
detector, plus ConfirmNod (5 degrees down and halfway back within 2 s) while a question or preview
waits; a second nod within 1 s is dropped. confirm="both" sends both. Shakes (no) and held tilts
(scroll) come from Hung's GestureDetector in every setting.
"""
from __future__ import annotations

import statistics
import threading
import time
from collections import deque

from engine.config import settings as default_settings
from engine.core import messages, topics
from engine.core.types import GazePoint, Gesture, MouthSample
from engine.eyes.cursor_driver import CursorDriver
from engine.eyes.head_gestures import GestureDetector
from engine.eyes.blink_gesture import BlinkDetector
from engine.eyes.brow_gesture import BrowRaiseDetector
from engine.eyes.dwell_click import DwellClicker
from engine.eyes.look_click import ConfirmNod
from engine.eyes.mouth_gesture import MouthOpenDetector
from engine.eyes.pointer import Pointer
from engine.eyes.scroll import HeadScroll, SmoothScroller

CALIB_SETTLE_S = 0.6      # look at the dot: the eyes and head settle first...
CALIB_COLLECT_S = 0.9     # ...then this long is measured
HOLD_TIMEOUT_S = 10.0     # a found target holds the cursor at most this long (the question expires at 8 s)
NOD_DEDUPE_S = 1.0        # a second nod this soon after one is the same nod
NO_FACE_S = 2.0
SCROLL_UI_S = 0.1         # how often the scroll speed goes to the UI
CONFIRMS = ("brow", "nod", "both")   # what says yes: eyebrow raise (default), nod, or either           # say so when no face was seen for this long
FREEZE_MODES = ("pointer", "dictation")
MAX_FREEZE_S = 2.0        # brows held up longer than this: the cursor follows the head again
FREEZE_LOOKBACK_MS = 100  # freeze where the cursor was this long before the raise crossed the threshold,
                          # since the brows (and the head with them) start moving a little before that
SEARCH_TIMEOUT_S = 10.0   # UI-Venus searching blocks dwell at most this long


class EyesService:
    def __init__(self, bus, settings=default_settings, pointer: str = "head", dwell: bool = False,
                 model_path: str | None = None, os_api=None, camera_factory=None, mesh_factory=None,
                 features=None, head_pose=None, clock=time.monotonic, confirm: str = "brow",
                 probe=None) -> None:
        if confirm not in CONFIRMS:
            raise ValueError(f"confirm must be one of {CONFIRMS}")
        self.confirm = confirm
        real_os = os_api is None
        if os_api is None:
            from engine.eyes import os_input as os_api
        self.bus, self.settings, self.os, self.clock = bus, settings, os_api, clock
        self.os.make_dpi_aware()   # first, so the screen size, crops and cursor all use physical pixels
        self.W, self.H = self.os.screen_size()
        self._camera_factory = camera_factory or self._real_camera
        self._mesh_factory = mesh_factory or self._real_mesh
        if features is None:
            from engine.eyes.eye_features import EyeFeatureExtractor
            features = EyeFeatureExtractor(iris_mode="center")
        if head_pose is None:
            from engine.eyes.head_pose import HeadPoseEstimator
            head_pose = HeadPoseEstimator()
        self.features, self.head = features, head_pose

        self.model, model_note = None, ""
        try:
            from engine.eyes.calibration import GazeModel
            self.model = GazeModel.load(model_path)
            if (self.model.screen_w, self.model.screen_h) != (self.W, self.H):
                model_note = f"gaze model is for {self.model.screen_w}x{self.model.screen_h}, not this screen"
                self.model = None
        except Exception as exc:   # no calibration yet, or an old file: head pointing still works
            model_note = f"no gaze model ({type(exc).__name__})"
        if pointer in ("hybrid", "gaze") and self.model is None:
            self._notes = [f"Pointer '{pointer}' needs a gaze model ({model_note}); using the head pointer."]
            pointer = "head"
        else:
            self._notes = []
        self.pointer = Pointer(self.W, self.H, pointer, self.model)
        self.driver = CursorDriver(self.os.move, still_px=self.pointer.still_px)
        self.gestures = GestureDetector()
        self.yes_nod = ConfirmNod()
        self.brows = BrowRaiseDetector()
        self.blink = (BlinkDetector(rise=float(getattr(settings, "blink_rise", 0.20)),
                                    closed_abs=float(getattr(settings, "blink_closed_abs", 0.55)))
                      if getattr(settings, "blink_click", False) else None)   # a long blink = click
        self.jaw = MouthOpenDetector() if getattr(settings, "mouth_click", True) else None   # open + close = right click
        self.dwell = DwellClicker(dwell_ms=getattr(settings, "dwell_ms", 1000)) if dwell else None
        self._dwell_blocked = True       # comes back disarmed at the cursor, so the first stop must be a new one
        self._dwell_sent = (0.0, False)  # the last (progress, ready) that went to the UI
        if probe is None and real_os:
            from engine.eyes.clickable import make_probe   # None off Windows: dwell clicks anywhere, as before
            probe, note = make_probe(getattr(settings, "dwell_uia", True))
            if note:
                self._notes.append(note)
        self.probe = probe               # what's under the cursor: for dwell, and for the snap click (target_service)
        self._frozen: tuple[float, float, float] | None = None   # x, y, since: cursor held during a raise
        self._freeze_spent = False       # this raise already used up MAX_FREEZE_S
        self._recent_cursor: deque[tuple[int, float, float]] = deque(maxlen=30)   # t_ms, x, y
        self.searching_until = 0.0       # UI-Venus is looking for a target until then
        self.head_scroll = HeadScroll()
        self.scroller = SmoothScroller(self._wheel, on_error=lambda exc: self._status(
            "error", f"Couldn't scroll: {type(exc).__name__}: {exc}"))
        self._scrolling = False          # camera thread only: head scrolling has started in scroll mode
        self._scroll_ui_at = 0.0
        self._wheel_left = 0.0

        self._lock = threading.Lock()
        self.mode = "pointer"
        self.question_open = False
        self.preview_waiting = False
        self._awaiting = False
        self.hold: tuple[float, float, float] | None = None   # x, y, deadline
        self.calib: dict | None = None
        self._calib_wanted = True        # the first face starts a calibration
        self._last_nod = -1e9
        self._last_face = None
        self._no_face_said = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.cursor: tuple[float, float] | None = None

        bus.subscribe(topics.TO_UI, self.on_ui)
        bus.subscribe(topics.CALIBRATE, self.on_calibrate)
        bus.subscribe(topics.from_ui(messages.CALIBRATE), self.on_calibrate)
        bus.subscribe(topics.from_ui(messages.SETTINGS), self.on_settings)
        bus.subscribe(topics.OS_CLICK, self.on_click)
        bus.subscribe(topics.DWELL, lambda m: self.set_dwell(bool(m.get("on"))))   # "stop/start clicking"

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _real_camera():
        from engine.eyes.camera import Camera
        return Camera()

    @staticmethod
    def _real_mesh():
        from engine.eyes.face_mesh import FaceMesh
        return FaceMesh()

    def start(self) -> None:
        for note in self._notes:
            self._status("warn", note)
        self.driver.start()
        self.scroller.start()
        if self.probe is not None:
            self.probe.start()
        self._thread = threading.Thread(target=self._run, name="eyes", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        self.scroller.stop()
        self.driver.stop()
        if self.probe is not None:
            self.probe.stop()

    def _wheel(self, units: int) -> None:
        """One smooth wheel step: 1/120 notch units where the OS takes them, else whole notches."""
        fn = getattr(self.os, "scroll_units", None)
        if fn is not None:
            fn(units)
            return
        self._wheel_left += units / 120.0
        whole = int(self._wheel_left)
        if whole:
            self._wheel_left -= whole
            self.os.scroll(whole)

    def _run(self) -> None:
        try:
            cam = self._camera_factory()
            mode = cam.start()
        except Exception as exc:
            self._status("error", f"Camera: {exc}")
            return
        self._status("info", f"Camera {mode.width}x{mode.height}, pointer: {self.pointer.mode}. "
                             "Look at the center of the screen to calibrate.")
        mesh = None
        try:
            mesh = self._mesh_factory()
            frame_id = 0
            while not self._stop.is_set():
                frame, t_ms, new_id = cam.wait_next(frame_id)
                if frame is None or new_id == frame_id:
                    continue
                frame_id = new_id
                self.process(mesh.process(frame, t_ms), self.clock())
        except Exception as exc:   # report it; the voice side keeps running
            self._status("error", f"Eyes stopped: {type(exc).__name__}: {exc}")
            raise
        finally:
            cam.stop()
            if mesh is not None:
                mesh.close()

    # ------------------------------------------------------- bus (any thread)
    def on_ui(self, msg: dict) -> None:
        """The mode manager's messages to the UI, read only. Quick: called for every UI message."""
        kind = msg.get("type")
        if kind == messages.MODE:
            self._set_mode(msg.get("mode", "pointer"))
        elif kind == messages.TARGET_SEARCHING:
            self.searching_until = self.clock() + SEARCH_TIMEOUT_S
        elif kind == messages.TARGET_FOUND:
            self.searching_until = 0.0
            self._hold_at(float(msg["x"]), float(msg["y"]))
        elif kind == messages.TARGET_NONE:
            self.searching_until = 0.0
            self._release()
        elif kind == messages.CONFIRM:
            self.question_open = msg.get("state") == "open"
            if not self.question_open:
                self._release()
        elif kind == messages.DICTATION_PREVIEW:
            self.preview_waiting = bool(msg.get("words"))

    def on_calibrate(self, msg: dict) -> None:
        if (msg or {}).get("action", "start") == "cancel":
            with self._lock:
                if self.calib is not None:
                    self.calib = None
                    self._ui(messages.CALIB_RESULT, error_px=0.0, ok=False)
        else:
            self._calib_wanted = True

    def on_settings(self, msg: dict) -> None:
        if "sensitivity" in msg:
            self.settings.sensitivity = float(msg["sensitivity"])   # HeadPointer reads it every frame
        if "dwell_ms" in msg and self.dwell is not None:
            self.dwell.dwell_ms = int(msg["dwell_ms"])

    def on_click(self, payload: dict) -> None:
        self._release()   # os_output clicks; the cursor follows the head again
        if self.dwell is not None:
            self.dwell.disarm(float(payload["x"]), float(payload["y"]))   # nothing more until the cursor moves away

    def set_dwell(self, on: bool) -> None:
        """Dwell clicking on or off (voice "stop clicking" / "start clicking"). Coming back on, it is
        disarmed where the cursor is, so it never clicks the spot you were resting on when you said it."""
        if self.dwell is None:
            self.dwell = DwellClicker(dwell_ms=getattr(self.settings, "dwell_ms", 1000))
        self.dwell.set_enabled(on)
        self._status("info", "Dwell clicking on" if on else "Dwell clicking off")

    def _set_mode(self, mode: str) -> None:
        self.mode = mode
        self.driver.enabled = mode != "paused"
        if mode != "pointer" and self.dwell is not None:
            self.dwell.reset()

    def _hold_at(self, x: float, y: float) -> None:
        with self._lock:
            self.hold = (x, y, self.clock() + HOLD_TIMEOUT_S)
        self.driver.snap(x, y)

    def _release(self) -> None:
        with self._lock:
            held, self.hold = self.hold, None
        if held is not None:
            self.pointer.place(held[0], held[1])   # head mode: carry on from the button just found

    # ------------------------------------------------------ per camera frame
    def process(self, face, now: float) -> None:
        """One camera frame (face is None when no face was found). The camera thread calls this."""
        if face is None:
            if self._scrolling:
                self.scroller.set_rate(0.0)      # no face: coast to a stop, don't run away
            if self._last_face is not None and now - self._last_face > NO_FACE_S and not self._no_face_said:
                self._status("warn", "Can't see your face: face the camera, add light")
                self._no_face_said = True
            return
        if self._no_face_said:
            self._status("info", "Face found again")
            self._no_face_said = False
        self._last_face = now
        feats, pose = self.features.compute(face), self.head.compute(face)
        # for the voice route: words count only while the user's own mouth moves (engine/eyes/mouth.py)
        self.bus.publish(topics.MOUTH, MouthSample(t_ms=face.t_ms, open=float((face.blendshapes or {}).get("jawOpen", 0.0))))

        if self._calib_wanted:
            self._calib_wanted = False
            self._start_calibration(now)
        if self.calib is not None:
            self._calibration_step(feats, pose, now)
            return

        if self.mode == "scroll":
            self._scroll_frame(face.t_ms, pose, now)
            if self.jaw is not None:
                self.jaw.update(face.t_ms, face.blendshapes or {})   # keeps its resting level; no right click here
            self._gestures(face.t_ms, pose, now, self.brows.update(face.t_ms, face.blendshapes or {}),
                           blink=self.blink.update(face.t_ms, face.blendshapes or {}, pose.pitch) if self.blink else None)
            return
        if self._scrolling:
            self._end_scroll()
        point = self.pointer.update(face.t_ms, feats, pose)
        brow = self.brows.update(face.t_ms, face.blendshapes or {})
        mouth = self.jaw.update(face.t_ms, face.blendshapes or {}) if self.jaw is not None else None
        blink = self.blink.update(face.t_ms, face.blendshapes or {}, pose.pitch) if self.blink is not None else None
        with self._lock:
            if self.hold is not None and now > self.hold[2]:
                self.hold = None
            holding = self.hold is not None
        point = self._freeze(point, now, holding)
        self.cursor = (point.x, point.y)
        self._recent_cursor.append((point.t_ms, point.x, point.y))
        self.bus.publish(topics.GAZE, GazePoint(t_ms=face.t_ms, x=point.x, y=point.y, confidence=point.confidence))
        self._ui(messages.GAZE, x=round(point.x, 1), y=round(point.y, 1), confidence=round(point.confidence, 2))

        if not holding and self.mode not in ("menu", "paused"):
            self.driver.set_target(point.x, point.y)

        self._gestures(face.t_ms, pose, now, brow, mouth, blink)
        if self.dwell is not None:
            self._dwell(face.t_ms, point, holding, now)

    def _freeze(self, point: GazePoint, now: float, holding: bool) -> GazePoint:
        """Hold the cursor where it was when a raise (or a held-open mouth) started, until the face relaxes
        (or MAX_FREEZE_S)."""
        raised, lookback_ms = self._face_held()
        if not raised:
            self._freeze_spent = False
        elif holding:
            self._freeze_spent = True   # this raise answers the found target; the hold already keeps the cursor
        if self._frozen is None:
            if raised and not self._freeze_spent and not holding and self.mode in FREEZE_MODES:
                x, y = self._cursor_before(point.t_ms - lookback_ms, point)
                self._frozen = (x, y, now)
            else:
                return point
        x, y, since = self._frozen
        if not raised or holding or now - since > MAX_FREEZE_S:
            self._frozen = None
            self._freeze_spent = raised
            self.pointer.place(x, y)   # carry on from the held spot, no jump
        return GazePoint(t_ms=point.t_ms, x=x, y=y, confidence=point.confidence)

    def _face_held(self) -> tuple[bool, int]:
        """(a click gesture is in progress, how far back it started in ms): the brows, or the mouth."""
        if self.brows.raised:
            return True, FREEZE_LOOKBACK_MS
        if self.blink is not None and self.blink.held:
            return True, self.blink.closed_ms + FREEZE_LOOKBACK_MS
        if self.jaw is not None and self.jaw.held:
            return True, self.jaw.open_ms + FREEZE_LOOKBACK_MS
        return False, 0

    def _cursor_before(self, t_ms: int, point: GazePoint) -> tuple[float, float]:
        """Where the cursor was at t_ms: the newest position at or before it (or the oldest one kept)."""
        for t, x, y in reversed(self._recent_cursor):
            if t <= t_ms:
                return x, y
        return (self._recent_cursor[0][1], self._recent_cursor[0][2]) if self._recent_cursor else (point.x, point.y)

    def _gestures(self, t_ms: int, pose, now: float, brow: Gesture | None = None,
                  mouth: Gesture | None = None, blink: Gesture | None = None) -> None:
        awaiting = self.question_open or self.preview_waiting   # "type <words>" previews in Pointer mode too
        if awaiting and not self._awaiting:
            self.yes_nod.arm()
        self._awaiting = awaiting
        yes = self.yes_nod.update(t_ms, pose.pitch)
        g = self.gestures.update(pose)
        if g is not None and g.type != "nod":
            self.bus.publish(topics.GESTURE, g)          # shake = no, held tilt = scroll
        elif (self.confirm != "brow" and (g is not None or (yes and awaiting))
              and now - self._last_nod > NOD_DEDUPE_S):
            self._last_nod = now
            self.bus.publish(topics.GESTURE, Gesture(t_ms=t_ms, type="nod"))
        if brow is not None and self.confirm != "nod":
            self.bus.publish(topics.GESTURE, brow)
        if blink is not None:
            self.bus.publish(topics.GESTURE, blink)   # a long blink: the same as a raise (yes, or the snap click)
        if mouth is not None:
            self.bus.publish(topics.GESTURE, mouth)   # the mode manager: right click (Pointer, nothing waiting)

    def _scroll_frame(self, t_ms: int, pose, now: float) -> None:
        """Scroll mode, one frame: the cursor stays; the head's tilt sets the speed."""
        if not self._scrolling:
            self._scrolling = True
            self.head_scroll.start()
            self.hold = None
            self._frozen = None
        self._dwell_blocked = True   # after scrolling the cursor rests where it stayed: move before a dwell click
        self.scroller.set_rate(self.head_scroll.update(t_ms, pose.pitch))
        if now - self._scroll_ui_at >= SCROLL_UI_S:
            self._scroll_ui_at = now
            self._ui(messages.SCROLL, rate=round(self.scroller.rate, 1), offset=round(self.head_scroll.offset, 1),
                     ready=self.head_scroll.settled)

    def _end_scroll(self) -> None:
        """Back to pointing: the speed coasts to zero and the cursor carries on from where it stayed."""
        self._scrolling = False
        self.scroller.set_rate(0.0)
        if self.cursor is not None:
            self.pointer.place(*self.cursor)
        self._ui(messages.SCROLL, rate=0.0, offset=0.0, ready=False)

    def _dwell(self, t_ms: int, point: GazePoint, holding: bool, now: float) -> None:
        blocked = (self.mode != "pointer" or holding or self.question_open or self.preview_waiting
                   or now < self.searching_until or self._frozen is not None or not self.dwell.enabled)
        if blocked:
            self._dwell_blocked = True
            self.dwell.reset()
        elif self._dwell_blocked:   # allowed again: not where the cursor already rests, only after a move
            self._dwell_blocked = False
            self.dwell.disarm(point.x, point.y)
        over = self.probe.over_at(point.x, point.y, now) if self.probe is not None else None
        gate = self.probe is not None and getattr(self.settings, "dwell_clickable_only", True)
        at = None if blocked else self.dwell.update(t_ms, point.x, point.y, over=over, gate=gate)
        shown = 1.0 if at is not None else self.dwell.progress
        ready = not blocked and over is not None and over.clickable and self.dwell.armed
        sent = (round(shown, 2), bool(ready))
        if sent != self._dwell_sent:   # the ring: fills while resting, one 0 after it stops or clicks
            self._ui(messages.DWELL, progress=sent[0], ready=sent[1])
            self._dwell_sent = sent
        if at is not None:
            self.bus.publish(topics.DWELL_DONE, GazePoint(t_ms=t_ms, x=at[0], y=at[1], confidence=point.confidence))

    # ------------------------------------------------------------ calibration
    def _start_calibration(self, now: float) -> None:
        self.scroller.set_rate(0.0)
        self._dwell_blocked = True   # after calibrating, the cursor sits on the center: don't click it
        self._frozen = None
        self._recent_cursor.clear()
        with self._lock:
            self.calib = {"t0": now, "gaze": []}
        cx, cy = self.W // 2, self.H // 2
        self._ui(messages.CALIB_TARGET, index=0, x=cx, y=cy, total=1)
        self._status("info", "Calibrating: look at the dot in the middle of the screen")
        self.driver.snap(cx, cy)

    def _calibration_step(self, feats, pose, now: float) -> None:
        with self._lock:
            calib = self.calib
        if calib is None:
            return
        elapsed = now - calib["t0"]
        if elapsed >= CALIB_SETTLE_S:
            g = self.pointer.raw_gaze(feats, pose)
            if g is not None:
                calib["gaze"].append(g)
        if elapsed < CALIB_SETTLE_S + CALIB_COLLECT_S:
            return
        self.head.set_neutral()   # median of the last frames, all spent looking at the dot
        shift_px = 0.0
        if calib["gaze"]:
            mx = statistics.median(p[0] for p in calib["gaze"])
            my = statistics.median(p[1] for p in calib["gaze"])
            self.pointer.gaze_shift = (self.W / 2 - mx, self.H / 2 - my)
            shift_px = (self.pointer.gaze_shift[0] ** 2 + self.pointer.gaze_shift[1] ** 2) ** 0.5
        self.pointer.reset()
        self.gestures.reset()
        self.yes_nod.reset()
        self.brows.reset()   # relearn the resting brow level
        if self.blink is not None:
            self.blink.reset()
        if self.jaw is not None:
            self.jaw.reset()
        with self._lock:
            self.calib = None
        self._ui(messages.CALIB_RESULT, error_px=round(shift_px, 1), ok=True)
        self._status("info", "Calibrated: the screen center is where you face now")

    # ----------------------------------------------------------------- output
    def _ui(self, msg_type: str, **fields) -> None:
        self.bus.publish(topics.TO_UI, messages.make(msg_type, **fields))

    def _status(self, level: str, text: str) -> None:
        self._ui(messages.STATUS, level=level, text=text)
