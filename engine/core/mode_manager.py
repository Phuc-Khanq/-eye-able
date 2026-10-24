"""State machine that decides what each gaze, gesture and command means right now.

Route: Khoi and Khoa (Khoi leads)
Publishes: to the UI (topics.TO_UI): mode, dictation_preview, confirm, target_searching,
           target_found, target_none, gesture (menu only), status
           to the engine: TargetRequest, os.click / os.type / os.key / os.scroll / os.undo,
           eyes.calibrate, voice.delete_data
Subscribes: GazePoint, dwell, Gesture, transcripts, TargetResult, UI messages
APIs: plain Python (enum, threading)

Only this module asks os_input to click or type, so stray signals can't act alone.

What each input means (one table per mode: GESTURES below):
  Pointer    dwell (rest the cursor 1 s) = LEFT click at the cursor; eyebrow raise = yes when a
             question, preview or search is waiting, otherwise the RIGHT click at the cursor;
             tilt left/right = scroll up/down; "click/open/select <name>" = look-and-say,
             "type <words>" = one-shot dictation (the words go to the preview; "yes" types them)
             An eyebrow raise made while the mouth is moving is ignored (talking, not a click).
  Dictation  speech -> preview bubble; eyebrow raise, nod or "yes" inserts, shake or "no" discards;
             "click <name>" still clicks (only "click": "open the door..." stays text)
  Menu       tilts and nods go to the UI's radial menu; shake closes it;
             the UI sends menu_select with the item
  Paused     everything is ignored except "resume" and the UI's set_mode

Preview, then confirm:
  * "send it", look-and-say clicks and "delete my voice data" always ask first
    (a confirm message); only nod / "yes" / the UI's button goes ahead
  * an open question expires after CONFIRM_TIMEOUT_S and counts as "no"
  * while a question is open, nod and shake only answer it

Look-and-say: command "click <name>" -> TargetRequest -> wait for TargetResult ->
target_found to the UI -> nod clicks, shake cancels.

Thread safety: camera, audio and Whisper threads all call in here, so every
handler takes one lock.
"""

from __future__ import annotations

import itertools
import re
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from engine.core import messages, topics
from engine.core.types import GazePoint, Gesture, TargetRequest, TargetResult
from engine.voice import commands
from engine.voice.commands import Command

CONFIRM_TIMEOUT_S = 12.0  # Whisper only answers after the pause, so a spoken "yes" lands 2-4 s late
TARGET_TIMEOUT_S = 6.0   # UI-Venus should answer well inside this
TICK_S = 0.25
REPEAT_CLICK_S = 3.0      # "click Send" again this soon after Send was clicked is a repeat, not a new click


class Mode(Enum):
    POINTER = "pointer"
    DICTATION = "dictation"
    MENU = "menu"
    PAUSED = "paused"


# Gesture -> intent, per mode. Change the controls here, in one place.
GESTURES: dict[Mode, dict[str, str]] = {
    # brow_raise = "gesture_yes": yes if something waits, else the right click (Pointer only)
    Mode.POINTER:   {"tilt_left": "scroll_up", "tilt_right": "scroll_down", "brow_raise": "gesture_yes"},
    Mode.DICTATION: {"nod": "yes", "shake": "no", "brow_raise": "gesture_yes"},
    Mode.MENU:      {"tilt_left": "menu_gesture", "tilt_right": "menu_gesture", "brow_raise": "menu_gesture",
                     "nod": "menu_gesture", "shake": "close_menu"},
    Mode.PAUSED:    {},
}

# Voice commands that still work while paused.
PAUSED_COMMANDS = {"resume"}


@dataclass
class _Question:
    id: str
    prompt: str
    expires_at: float
    on_yes: Callable[[], None]
    on_no: Callable[[], None]
    kind: str = "general"  # "target" questions can also be answered by the UI's target message
    label: str = ""        # target questions: what would be clicked


@dataclass
class _Search:
    label: str
    expires_at: float
    preconfirmed: bool = False  # "click Send, yes": the yes came before UI-Venus answered


class ModeManager:
    def __init__(self, bus, clock: Callable[[], float] = time.monotonic,
                 rewrite_wait_s: float | None = None, mouth=None, settings=None, log=print) -> None:
        """rewrite_wait_s: the dictation clean-up is on (engine/voice/rewrite.py's RewriteWorker is
        running) and "yes" waits at most this long for it. None (the default): off, exactly as before.
        mouth: engine.eyes.mouth.MouthTracker. With it, an eyebrow raise made while the user is talking
        is ignored (see _raised_while_talking). None (the default): no guard, exactly as before."""
        self.bus = bus
        self.clock = clock
        self.mouth = mouth
        self._brow_guard = bool(getattr(settings, "brow_ignore_while_talking", True))
        self._brow_window_ms = float(getattr(settings, "brow_talk_window_ms", 600))
        self._brow_min_moving = float(getattr(settings, "brow_talk_min_moving", 0.3))
        self._brow_warned = float("-inf")   # clock time of the last "tracker failed" warning
        self._log = log                     # console lines for tuning
        self.mode = Mode.POINTER
        self._return_mode = Mode.POINTER   # where Menu and Paused go back to
        self.preview: list[dict] = []      # words waiting for nod / shake
        self.question: _Question | None = None
        self.search: _Search | None = None
        self.last_gaze: GazePoint | None = None
        self._ids = itertools.count(1)
        self._last_click: tuple[str, float] | None = None   # (label, time) of the last look-and-say click
        # Dictation clean-up (see _preview_changed): only the latest request's result counts
        self.rewrite_wait_s = rewrite_wait_s
        self._rw_id = 0
        self._rw_running = False                 # a clean-up of the current preview is still running
        self._clean: list[dict] | None = None    # the accepted clean words for the current preview
        self._use_clean = True                   # "use original" turns it off for this preview only
        self._pending: tuple[float, Callable[[], None] | None] | None = None  # (deadline, then): a waiting insert
        self._lock = threading.RLock()
        self._stop = threading.Event()

        bus.subscribe(topics.GAZE, self.on_gaze)
        bus.subscribe(topics.DWELL_DONE, self.on_dwell)
        bus.subscribe(topics.GESTURE, self.on_gesture)
        bus.subscribe(topics.TRANSCRIPT, self.on_transcript)
        bus.subscribe(topics.TARGET_RESULT, self.on_target_result)
        bus.subscribe(topics.from_ui(messages.SET_MODE), self.on_ui_set_mode)
        bus.subscribe(topics.from_ui(messages.MENU_SELECT), self.on_ui_menu_select)
        bus.subscribe(topics.from_ui(messages.DICTATION), self.on_ui_dictation)
        bus.subscribe(topics.from_ui(messages.TARGET), self.on_ui_target)
        bus.subscribe(topics.from_ui(messages.CONFIRM_ANSWER), self.on_ui_confirm_answer)
        bus.subscribe(topics.REWRITE_RESULT, self.on_rewrite_result)

    def start(self) -> None:
        """Start the timer thread that expires old questions and searches."""
        def loop() -> None:
            while not self._stop.wait(TICK_S):
                self.tick()

        threading.Thread(target=loop, name="mode-manager-tick", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------ inputs
    def on_gaze(self, gaze: GazePoint) -> None:
        with self._lock:
            self.last_gaze = gaze

    def on_dwell(self, gaze: GazePoint) -> None:
        with self._lock:
            self.tick()
            if self.mode == Mode.POINTER and not self.question:
                self._os(topics.OS_CLICK, x=int(gaze.x), y=int(gaze.y))

    def on_gesture(self, gesture: Gesture) -> None:
        with self._lock:
            self.tick()
            if gesture.type == "brow_raise" and (self.question or self.mode != Mode.PAUSED):
                if self._raised_while_talking(gesture.t_ms):
                    return
            if self.question:
                if gesture.type in ("nod", "brow_raise"):
                    self._answer(True)
                elif gesture.type == "shake":
                    self._answer(False)
                return
            intent = GESTURES[self.mode].get(gesture.type)
            if intent == "menu_gesture":
                self._ui(messages.GESTURE, gesture=gesture.type)
            elif intent == "gesture_yes":
                # An eyebrow raise answers whatever is waiting. With nothing waiting it is the RIGHT
                # click in Pointer mode (dwell is the left click), at the cursor: the eyes service
                # freezes the cursor during a raise, so the last gaze is where it froze. A raise while
                # talking never gets here (_raised_while_talking).
                if self.preview:
                    self._insert_preview()
                elif self.search:
                    self.search.preconfirmed = True
                    self._status("info", f"Will click {self.search.label} as soon as it's found")
                elif self.mode == Mode.POINTER:
                    self._right_click()
            elif intent:
                self._do(Command(name=intent))

    def _raised_while_talking(self, t_ms: float) -> bool:
        """An eyebrow raise made while the mouth was moving is talking, not a yes or a click: people
        raise their brows when they talk (about 0.8 a minute in Hung's test). Fails open: no face
        (None) or a broken tracker lets the raise through."""
        if self.mouth is None or not self._brow_guard:
            return False
        try:
            f = self.mouth.moving_fraction(t_ms - self._brow_window_ms, t_ms)
        except Exception as exc:
            now = self.clock()
            if now - self._brow_warned >= 60:
                self._brow_warned = now
                self._log(f"[brow] the mouth tracker failed ({exc}); not ignoring raises")
            return False
        if f is not None and f >= self._brow_min_moving:
            self._log(f"[brow] ignored: you were talking ({f:.2f})")
            return True
        return False

    def on_transcript(self, transcript: dict) -> None:
        """One utterance can hold several things ("Click search box. Yeah."), so it is cut into
        clauses (commands.split_clauses) and each clause is read on its own:
          Pointer    every clause that is a command runs, in order; other clauses are ignored
          Dictation  a whole-utterance command runs as before; otherwise "click <name>" clauses and
                     clauses after a trigger word ("Venus, stop dictation") are commands, and
                     the rest goes to the preview
          Menu       fixed commands only;  Paused  only "resume"
        The same command twice in a row in one utterance runs once."""
        with self._lock:
            self.tick()
            words = transcript.get("words") or [{"text": w, "conf": 1.0} for w in transcript["text"].split()]
            clauses = commands.split_clauses(words)
            if self.mode == Mode.PAUSED:
                for c in clauses:
                    cmd = commands.parse(c.text, allow_targets=False)
                    if cmd and cmd.name in PAUSED_COMMANDS:
                        self._do(cmd)
                        return
                return
            if self.mode == Mode.DICTATION:
                whole = commands.parse(transcript["text"], allow_targets=True,
                                       target_verbs=commands.DICTATION_TARGET_VERBS)
                if (whole and whole.name != "type_text" and len(clauses) <= 1
                        and not (whole.name == "target"
                                 and len(whole.label.split()) > commands.DICTATION_MAX_LABEL_WORDS)):
                    if whole.name in ("yes", "no") and not (self.question or self.preview or self.search):
                        self._add_to_preview(words)   # nothing is waiting: a one-word reply to dictate
                        return
                    self._do(whole)
                    return
            done, last, i = False, None, 0
            while i < len(clauses):
                c = clauses[i]
                cmd = self._clause_command(c)
                if cmd is None:
                    if self.mode == Mode.DICTATION:
                        self._add_to_preview(c.words)
                    i += 1
                    continue
                done = True
                if cmd.name == "type_text":
                    rest = [w for later in clauses[i + 1:] for w in later.words]
                    self._type_text({"words": c.words + rest})
                    return
                if (cmd.name, cmd.label) != last:
                    self._do(cmd)
                last = (cmd.name, cmd.label)
                i += 1
            if not done and self.mode != Mode.DICTATION:
                self._status("info", f'Heard "{transcript["text"]}" (not a command)')

    def _clause_command(self, c) -> Command | None:
        if c.triggered or self.mode == Mode.POINTER:
            cmd = commands.parse(c.text, allow_targets=self.mode in (Mode.POINTER, Mode.DICTATION))
            if cmd and not c.triggered and not c.starts_sentence:
                return None   # cut from mid-sentence: "I said click it" is talk, not a yes
            return cmd
        if self.mode == Mode.DICTATION:   # untriggered: only "click <name>", as its own short sentence
            cmd = commands.parse(c.text, allow_targets=True, target_verbs=commands.DICTATION_TARGET_VERBS)
            if (cmd and cmd.name == "target" and c.starts_sentence
                    and len(cmd.label.split()) <= commands.DICTATION_MAX_LABEL_WORDS):
                return cmd
            return None
        cmd = commands.parse(c.text, allow_targets=False)   # Menu
        return None if cmd and cmd.name == "type_text" else cmd

    def on_target_result(self, result: TargetResult) -> None:
        with self._lock:
            if not self.search or result.label != self.search.label:
                return  # late answer to a search we already gave up on
            preconfirmed, self.search = self.search.preconfirmed, None
            if not result.found:
                self._ui(messages.TARGET_NONE, label=result.label)
                return
            x, y, label = result.x, result.y, result.label
            self._ui(messages.TARGET_FOUND, x=x, y=y, label=label)
            click = lambda: self._click_target(label, x, y)
            if preconfirmed:   # already said yes while UI-Venus was looking
                click()
                return
            self._ask(f"Click {label}?", on_yes=click, kind="target", label=label)

    def tick(self) -> None:
        """Expire an old question or search. The timer thread calls this; so does every input."""
        with self._lock:
            now = self.clock()
            if self.question and now >= self.question.expires_at:
                q, self.question = self.question, None
                self._ui(messages.CONFIRM, id=q.id, prompt=q.prompt, state="expired",
                         expires_in_s=0)
                q.on_no()
            if self.search and now >= self.search.expires_at:
                s, self.search = self.search, None
                self._ui(messages.TARGET_NONE, label=s.label)
            if self._pending and now >= self._pending[0]:
                self._type_now(self._pending[1])   # the clean-up took too long: type what's showing

    # ------------------------------------------------------------- UI messages
    def on_ui_set_mode(self, msg: dict) -> None:
        with self._lock:
            try:
                mode = Mode(msg.get("mode"))
            except ValueError:
                return
            if mode in (Mode.MENU, Mode.PAUSED) and self.mode not in (Mode.MENU, Mode.PAUSED):
                self._return_mode = self.mode
            self._set_mode(mode)

    def on_ui_menu_select(self, msg: dict) -> None:
        with self._lock:
            if self.mode != Mode.MENU:
                return
            self._set_mode(self._return_mode)
            item = msg.get("item", "")
            if item in commands.FIXED_COMMANDS.values():
                self._do(Command(name=item))

    def on_ui_dictation(self, msg: dict) -> None:
        with self._lock:
            action = msg.get("action")
            if action == "insert":
                self._insert_preview()
            elif action == "discard":
                self._discard_preview()
            elif action == "edit_word":
                shown = self._shown_words()
                i = msg.get("index", -1)
                if 0 <= i < len(shown) and msg.get("text"):
                    # the index is into what the UI shows; a fixed word is final, so no new clean-up
                    self.preview = [dict(w) for w in shown]
                    self.preview[i] = {"text": msg["text"], "conf": 1.0}
                    self._preview_changed(clean_up=False)

    def on_ui_target(self, msg: dict) -> None:
        with self._lock:
            if self.question and self.question.kind == "target":
                self._answer(msg.get("action") == "click")
            elif msg.get("action") == "cancel" and self.search:
                self._cancel_search()

    def on_ui_confirm_answer(self, msg: dict) -> None:
        with self._lock:
            if self.question and msg.get("id") == self.question.id:
                self._answer(bool(msg.get("yes")))

    # ----------------------------------------------------------------- intents
    def _do(self, cmd: Command) -> None:
        name, m = cmd.name, self.mode

        if name in ("yes", "no"):
            if self.question:
                self._answer(name == "yes")
            elif self.preview:
                self._insert_preview() if name == "yes" else self._discard_preview()
            elif self.search:
                if name == "yes":
                    self.search.preconfirmed = True
                    self._status("info", f"Will click {self.search.label} as soon as it's found")
                else:
                    self._cancel_search()
            else:
                self._status("info", f'Nothing is waiting for a "{name}" (a question expires after '
                                      f'{CONFIRM_TIMEOUT_S:.0f} s)')
        elif name == "cancel":
            if self.question:
                self._answer(False)
            elif self.search:
                self._cancel_search()
            elif m == Mode.MENU:
                self._set_mode(self._return_mode)
            elif self.preview:
                self._discard_preview()
        elif name == "start_dictation":
            self._set_mode(Mode.DICTATION)
        elif name == "stop_dictation":
            if self.preview:
                self._ask(f'Insert "{self._preview_text()}" before stopping?',
                          on_yes=lambda: self._insert_preview(then=lambda: self._set_mode(Mode.POINTER)),
                          on_no=lambda: (self._discard_preview(), self._set_mode(Mode.POINTER)))
            else:
                self._set_mode(Mode.POINTER)
        elif name == "pause":
            if m != Mode.PAUSED:
                if m != Mode.MENU:
                    self._return_mode = m
                self._set_mode(Mode.PAUSED)
        elif name == "resume":
            if m == Mode.PAUSED:
                self._set_mode(self._return_mode)
        elif name == "menu":
            if m not in (Mode.MENU, Mode.PAUSED):
                self._return_mode = m
                self._set_mode(Mode.MENU)
        elif name == "close_menu":
            self._set_mode(self._return_mode)
        elif name in ("scroll_up", "scroll_down"):
            self._os(topics.OS_SCROLL, direction=name.split("_")[1])
        elif name == "send_it":
            self._ask_send()
        elif name == "undo":
            if self.preview:
                self._discard_preview()
            else:
                self._os(topics.OS_UNDO)
        elif name == "new_line":
            if self.preview:
                self._status("warn", "Nod to insert the preview first")
            else:
                # shift+enter: a plain Enter would SEND in most chat apps
                self._os(topics.OS_KEY, key="shift+enter")
        elif name == "calibrate":
            self.bus.publish(topics.CALIBRATE, {"action": "start"})
        elif name == "delete_voice_data":
            self._ask("Delete your voice recordings and vocabulary?",
                      on_yes=lambda: self.bus.publish(topics.DELETE_VOICE_DATA, {}))
        elif name == "target":
            self._look_and_say(cmd.label)
        elif name == "use_original":
            if self.preview and self._use_clean:
                self._use_clean = False
                self._show_preview()
        elif name == "clean_up":
            if not self.preview:
                return
            if self._clean is None:
                self._status("info", "Still cleaning up..." if self._rw_running
                             else "There's no cleaned-up version of this text")
            elif not self._use_clean:
                self._use_clean = True
                self._show_preview()

    # ------------------------------------------------------------ look-and-say
    def _right_click(self) -> None:
        g = self.last_gaze
        if g is None:
            self._status("info", "Can't right-click yet: the cursor position isn't known")
            return
        self._os(topics.OS_CLICK, x=int(g.x), y=int(g.y), button="right")

    def _click_target(self, label: str, x: int, y: int) -> None:
        self._last_click = (label, self.clock())
        self._os(topics.OS_CLICK, x=x, y=y)

    def _look_and_say(self, label: str) -> None:
        # Saying the same "click <name>" again is impatience, not a new request:
        q, now = self.question, self.clock()
        if q and q.kind == "target" and q.label == label:
            self._answer(True)                       # repeated while "Click X?" is open: that's a yes
            return
        if self.search and self.search.label == label:
            self.search.preconfirmed = True          # repeated while searching: click when found
            return
        if self._last_click and self._last_click[0] == label and now - self._last_click[1] < REPEAT_CLICK_S:
            return                                   # just clicked it: don't click again
        if self.last_gaze is None:
            self._status("warn", f"Look near {label} first, then say it again")
            return
        if self.search:
            self._cancel_search()
        g = self.last_gaze
        self.search = _Search(label, self.clock() + TARGET_TIMEOUT_S)
        self._ui(messages.TARGET_SEARCHING, label=label)
        self.bus.publish(topics.TARGET_REQUEST,
                         TargetRequest(gaze_x=int(g.x), gaze_y=int(g.y), label=label))

    def _cancel_search(self) -> None:
        s, self.search = self.search, None
        if s:
            self._ui(messages.TARGET_NONE, label=s.label)

    # ----------------------------------------------------------------- preview
    def _type_text(self, transcript: dict) -> None:
        """'type <words>' / 'dictate <words>': only the words after the verb go to the preview,
        with their confidence; then "yes" (or a nod) types them, as in Dictation mode."""
        words = list(transcript.get("words") or [])
        for i, w in enumerate(words):
            if commands.normalize(w["text"]) in commands.TYPE_VERBS:
                words = words[i + 1:]
                break
        if not words:
            return
        self._add_to_preview(words)

    def _add_to_preview(self, words: list[dict]) -> None:
        self.preview = self.preview + list(words)  # keep talking: new words join the bubble
        self._preview_changed()

    def _preview_text(self) -> str:
        """What "yes" would type: the clean text if it's showing, else the user's own words."""
        return " ".join(w["text"] for w in self._shown_words())

    def _insert_preview(self, then: Callable[[], None] | None = None) -> str:
        """Type the preview, then run `then` ("send it": press Enter; "stop dictation": leave the mode).
        If the clean-up of this text is still running, don't block (this may be the camera thread):
        the insert waits until the result arrives or rewrite_wait_s passes (tick), whichever is first."""
        if self.preview and self._rw_running and self._use_clean:
            self._pending = (self.clock() + self.rewrite_wait_s, then)
            return self._preview_text()
        return self._type_now(then)

    def _type_now(self, then: Callable[[], None] | None = None) -> str:
        self._pending = None
        text = self._preview_text()
        if text:
            self._os(topics.OS_TYPE, text=text + " ")
            self.preview = []
            self._preview_changed(clean_up=False)          # empty words = hide the bubble
        if then:
            then()
        return text

    def _discard_preview(self) -> None:
        if self.preview:
            self.preview = []
            self._preview_changed(clean_up=False)

    # ------------------------------------------------------------- clean-up
    def _cleaning(self) -> bool:
        return self.rewrite_wait_s is not None

    def _shown_words(self) -> list[dict]:
        if self._cleaning() and self._use_clean and self._clean is not None:
            return self._clean
        return self.preview

    def _preview_changed(self, clean_up: bool = True) -> None:
        """self.preview just changed: the old clean text and any waiting insert no longer apply.
        Ask for a clean-up of the new text (unless the user fixed a word by hand) and show it raw."""
        self._pending = None
        self._clean = None
        self._rw_id += 1                                   # results for older text are ignored
        self._rw_running = False
        if not self.preview:
            self._use_clean = True                         # "use original" lasts for one preview
        elif self._cleaning() and clean_up:
            self._rw_running = True
            self.bus.publish(topics.REWRITE_REQUEST,
                             {"id": self._rw_id, "text": " ".join(w["text"] for w in self.preview)})
        self._show_preview()

    def _show_preview(self) -> None:
        if not self._cleaning():
            self._ui(messages.DICTATION_PREVIEW, words=list(self.preview))   # exactly as before
        elif self._clean is not None and self._use_clean:
            self._ui(messages.DICTATION_PREVIEW, words=list(self._clean),
                     original=" ".join(w["text"] for w in self.preview), cleaned=True)
        else:
            self._ui(messages.DICTATION_PREVIEW, words=list(self.preview), cleaned=False)

    def on_rewrite_result(self, result: dict) -> None:
        with self._lock:
            if result.get("id") != self._rw_id or not self.preview:
                return                                     # for text that has changed since
            self._rw_running = False
            if result.get("used") and str(result.get("clean", "")).strip():
                self._clean = self._clean_words(result["clean"])
                if self._use_clean:
                    self._show_preview()
            if self._pending:
                self._type_now(self._pending[1])

    def _clean_words(self, clean: str) -> list[dict]:
        """Words for the clean text. A word the user also said keeps Whisper's lowest confidence for it,
        so the UI still marks words Whisper wasn't sure of; words the clean-up wrote get 1.0."""
        def key(t: str) -> str:
            return re.sub(r"[^\w']", "", t.lower())

        conf: dict[str, float] = {}
        for w in self.preview:
            k = key(w["text"])
            conf[k] = min(conf.get(k, 1.0), float(w.get("conf", 1.0)))
        return [{"text": t, "conf": conf.get(key(t), 1.0)} for t in clean.split()]

    # ----------------------------------------------------------------- confirm
    def _ask_send(self) -> None:
        text = self._preview_text()
        prompt = f'Insert "{text}" and send?' if text else "Send?"

        def yes() -> None:
            self._insert_preview(then=lambda: self._os(topics.OS_KEY, key="enter"))

        self._ask(prompt, on_yes=yes)

    def _ask(self, prompt: str, on_yes: Callable[[], None],
             on_no: Callable[[], None] = lambda: None, kind: str = "general", label: str = "") -> None:
        if self.question:   # a newer question replaces the old one, which counts as "no"
            self._answer(False)
        q = _Question(f"q{next(self._ids)}", prompt, self.clock() + CONFIRM_TIMEOUT_S,
                      on_yes, on_no, kind, label)
        self.question = q
        self._ui(messages.CONFIRM, id=q.id, prompt=prompt, state="open",
                 expires_in_s=CONFIRM_TIMEOUT_S)

    def _answer(self, yes: bool) -> None:
        q, self.question = self.question, None
        if q is None:
            return
        self._ui(messages.CONFIRM, id=q.id, prompt=q.prompt,
                 state="confirmed" if yes else "cancelled", expires_in_s=0)
        (q.on_yes if yes else q.on_no)()

    # ----------------------------------------------------------------- outputs
    def _set_mode(self, mode: Mode) -> None:
        if mode == self.mode:
            return
        if self.mode == Mode.DICTATION and mode == Mode.POINTER:
            self._discard_preview()   # leaving dictation for good: drop what wasn't confirmed
        self.mode = mode
        self._ui(messages.MODE, mode=mode.value)

    def _ui(self, msg_type: str, **fields) -> None:
        self.bus.publish(topics.TO_UI, messages.make(msg_type, **fields))

    def _os(self, topic: str, **fields) -> None:
        self.bus.publish(topic, fields)

    def _status(self, level: str, text: str) -> None:
        self._ui(messages.STATUS, level=level, text=text)
