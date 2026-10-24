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
  paused               the engine leaves the cursor alone
  calibrating          holds on the screen center while you look at the dot (about 1.5 s)
The first face the camera sees starts a calibration, so the head's neutral pose is wherever you face
the screen center. Say "calibrate" (or the UI's calibrate) to redo it after moving your chair.

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

from engine.config import settings as default_settings
from engine.core import messages, topics
from engine.core.types import GazePoint, Gesture, MouthSample
from engine.eyes.cursor_driver import CursorDriver
from engine.eyes.head_gestures import GestureDetector
from engine.eyes.brow_gesture import BrowRaiseDetector
from engine.eyes.look_click import ConfirmNod
from engine.eyes.pointer import Pointer
from engine.eyes.smoothing import DwellDetector

CALIB_SETTLE_S = 0.6      # look at the dot: the eyes and head settle first...
CALIB_COLLECT_S = 0.9     # ...then this long is measured
HOLD_TIMEOUT_S = 10.0     # a found target holds the cursor at most this long (the question expires at 8 s)
NOD_DEDUPE_S = 1.0        # a second nod this soon after one is the same nod
NO_FACE_S = 2.0
CONFIRMS = ("brow", "nod", "both")   # what says yes: eyebrow raise (default), nod, or either           # say so when no face was seen for this long
DWELL_RADIUS_PX = 60.0


class EyesService:
    def __init__(self, bus, settings=default_settings, pointer: str = "head", dwell: bool = False,
                 model_path: str | None = None, os_api=None, camera_factory=None, mesh_factory=None,
                 features=None, head_pose=None, clock=time.monotonic, confirm: str = "brow") -> None:
        if confirm not in CONFIRMS:
            raise ValueError(f"confirm must be one of {CONFIRMS}")
        self.confirm = confirm
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
        self.dwell = DwellDetector(DWELL_RADIUS_PX, getattr(settings, "dwell_ms", 800)) if dwell else None
        self._dwell_armed = True
        self._dwell_at: tuple[float, float] | None = None

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
        self._thread = threading.Thread(target=self._run, name="eyes", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        self.driver.stop()

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
        elif kind == messages.TARGET_FOUND:
            self._hold_at(float(msg["x"]), float(msg["y"]))
        elif kind == messages.TARGET_NONE:
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

        point = self.pointer.update(face.t_ms, feats, pose)
        self.cursor = (point.x, point.y)
        self.bus.publish(topics.GAZE, GazePoint(t_ms=face.t_ms, x=point.x, y=point.y, confidence=point.confidence))
        self._ui(messages.GAZE, x=round(point.x, 1), y=round(point.y, 1), confidence=round(point.confidence, 2))

        with self._lock:
            if self.hold is not None and now > self.hold[2]:
                self.hold = None
            holding = self.hold is not None
        if not holding and self.mode not in ("menu", "paused"):
            self.driver.set_target(point.x, point.y)

        self._gestures(face.t_ms, pose, now, face.blendshapes)
        if self.dwell is not None:
            self._dwell(face.t_ms, point, holding)

    def _gestures(self, t_ms: int, pose, now: float, blendshapes: dict | None = None) -> None:
        awaiting = self.question_open or self.preview_waiting   # "type <words>" previews in Pointer mode too
        if awaiting and not self._awaiting:
            self.yes_nod.arm()
        self._awaiting = awaiting
        yes = self.yes_nod.update(t_ms, pose.pitch)
        brow = self.brows.update(t_ms, blendshapes or {})
        g = self.gestures.update(pose)
        if g is not None and g.type != "nod":
            self.bus.publish(topics.GESTURE, g)          # shake = no, held tilt = scroll
        elif (self.confirm != "brow" and (g is not None or (yes and awaiting))
              and now - self._last_nod > NOD_DEDUPE_S):
            self._last_nod = now
            self.bus.publish(topics.GESTURE, Gesture(t_ms=t_ms, type="nod"))
        if brow is not None and self.confirm != "nod":
            self.bus.publish(topics.GESTURE, brow)

    def _dwell(self, t_ms: int, point: GazePoint, holding: bool) -> None:
        if self.mode != "pointer" or holding or self.question_open:
            self.dwell.reset()
            return
        if not self._dwell_armed:   # after a dwell click, move away before the next one
            if ((point.x - self._dwell_at[0]) ** 2 + (point.y - self._dwell_at[1]) ** 2) ** 0.5 > DWELL_RADIUS_PX:
                self._dwell_armed = True
                self.dwell.reset()
            return
        progress = self.dwell.update(t_ms, point.x, point.y)
        if progress > 0:
            self._ui(messages.DWELL, progress=round(progress, 2))
        if progress >= 1.0:
            self._dwell_armed, self._dwell_at = False, (point.x, point.y)
            self.dwell.reset()
            self.bus.publish(topics.DWELL_DONE, point)

    # ------------------------------------------------------------ calibration
    def _start_calibration(self, now: float) -> None:
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
        with self._lock:
            self.calib = None
        self._ui(messages.CALIB_RESULT, error_px=round(shift_px, 1), ok=True)
        self._status("info", "Calibrated: the screen center is where you face now")

    # ----------------------------------------------------------------- output
    def _ui(self, msg_type: str, **fields) -> None:
        self.bus.publish(topics.TO_UI, messages.make(msg_type, **fields))

    def _status(self, level: str, text: str) -> None:
        self._ui(messages.STATUS, level=level, text=text)
