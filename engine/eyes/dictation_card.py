"""The dictation card: a see-through floating panel that shows what was heard and what will be typed, so the
user never has to read the terminal. Drawn by the engine itself, so it works without the Electron UI.
Styled after a meeting-assistant overlay: a small pill on top (state, and a live waveform while you speak),
and under it a dark glass panel with your own words in a blue chip on the right, the text that will be typed,
an action row, and a hint field with key caps.

Route: Khang (Nhi's Electron bubble can replace it later: dictation_card false in settings)
Publishes: nothing
Subscribes: topics.TO_UI, read only: listening, mode, dictation_preview, dictation_live, confirm; topics.TRANSCRIPT (what was heard);
            topics.OS_TYPE (so it can say "Typed" rather than "Dropped")
APIs: tkinter Canvas, as a Toplevel on the click ring's Tk thread (engine/eyes/dwell_meter.py), so the engine
      has one Tk thread; Windows ctypes for click-through, never focused, out of the taskbar, and out of
      screenshots (SetWindowDisplayAffinity), so UI-Venus never reads the card as part of the page

States, first match wins:
  confirm    "Question": the engine's yes/no question ("Insert "..." and send?", "Click Send?") in large text,
             the seconds left as a thin bar, and "blink or say yes · say no to cancel" in the hint field. It
             contains the preview's text, so it beats the preview. Stays until the engine closes it.
  preview    "Will type": the words waiting to be typed (unsure words in amber, underlined), your own words in
             the chip, "Type it / Drop it / Use original", and "blink or say yes" in the hint field. Stays until
             it is typed or dropped.
  live       "Hearing": what Whisper hears so far of the sentence being spoken, in grey, about once a second. With
             a preview waiting it follows the preview's words ("Will type"), so you see everything at once; the
             finished words replace it (refined text later replaces those). Dictation mode only.
  done       "Typed" or "Dropped" for a moment after a preview closes; "Done" or "Cancelled" after a question
  heard      for HEARD_S after anything is heard: your words in the chip, so every command gets feedback
  listening  the pill only, with a moving waveform, as soon as the mic hears speech (before Whisper finishes)
  mode       the pill only, in Dictation, Scroll or Paused mode
  hidden     otherwise (moved off screen; never withdrawn, so showing it can't take the keyboard focus)
The pill also shows the waveform whenever the mic hears speech, over any state.
Out of screenshots also means out of OBS: set card_hide_from_capture false while recording the demo video.
"""
from __future__ import annotations

import math
import re
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field

from engine.core import messages, topics

LIVE_STALE_S = 4.0        # live text with no update this long is dropped (the engine sends one about every second)
HEARD_S = 2.5             # your words stay in the chip this long after a command
DONE_S = 1.2              # "Typed" / "Dropped" stays this long
FLASH_S = 1.2             # a command's name ("Volume up") stays in the pill this long
LEAVE_FLASH = {"dictation": "Dictation off", "scroll": "Scroll off", "paused": "Resumed", "menu": "Menu closed"}
LISTEN_HOLD_S = 0.4       # the waveform stays a moment after the mic goes quiet, so it doesn't flicker
SHAKY_CONF = 0.6          # a word under this confidence is drawn in amber, underlined
# Ways to answer (all real commands; tests/test_answers_and_click_fx.py checks). The original card shows "yes" and "no"
TYPE_IT = ("yes", "yeah", "write it")
DROP_IT = ("no", "nope", "don't write it")
YES_TO_QUESTION = ("yes", "yeah", "do it")
NO_TO_QUESTION = ("no", "nope", "cancel")
CHIP_MAX_CHARS = 70
QUOTE_MAX_CHARS = 120     # the dictated text quoted inside a question is shortened to this
PROMPT_MAX_CHARS = 180    # and the whole question to this, so the card never grows past the screen
CONFIRM_GRACE_S = 2.0     # a question whose end message never came is dropped this long after it expires
WAVE_BARS = 14
CHIP_GAP = 24             # px between the gray label and the blue chip

# look
PANEL_W = 560
PAD = 18
R_PANEL = 28              # big rounded corners: a soft floating card, not a box
PILL_H = 38
GAP = 8                   # between the pill and the panel
BOTTOM_GAP = 80           # px above the bottom of the screen (clear of the taskbar)
TOP_GAP = 10              # px below the top of the screen
POLL_MS = 50
KEY = "#ff00fe"           # made transparent: the rounded corners
PANEL = "#1b1c21"
PANEL_EDGE = "#3c3d45"
PILL = "#26272d"
TEXT = "#f4f4f5"
MUTED = "#a1a1aa"
CHIP = "#2563eb"
SHAKY = "#fbbf24"
FIELD = "#24252b"
KEYCAP = "#3a3b42"
FONT = "Segoe UI"
WDA_EXCLUDEFROMCAPTURE = 0x11
OFF_SCREEN = "+-5000+-5000"
MODE_NAMES = {"pointer": "[eye]able", "dictation": "Dictation", "scroll": "Scroll", "paused": "Paused",
              "menu": "Menu"}
MODE_HINTS = {"dictation": "say “dictate off” to stop", "scroll": "tilt your head · say “stop”",
              "paused": "say “resume”"}


@dataclass
class View:
    kind: str                                  # "confirm" | "preview" | "live" | "done" | "command" | "heard" | "listening" |
                                               # "mode" | "hidden" ("command": the pill alone, with the command's name)
    pill: str = ""                             # the pill's text (the mode)
    pill_hint: str = ""
    listening: bool = False                    # the pill shows the waveform
    label: str = ""                            # small gray line at the top of the panel
    chip: str = ""                             # your own words, in the blue chip
    words: list = field(default_factory=list)  # [(text, shaky)]: the text that will be typed; shaky True = unsure
                                               # (amber), "live" = still being heard (grey)
    body: str = ""                             # plain large text (done)
    actions: list = field(default_factory=list)  # ["type", "drop", "original"]
    keys: bool = False                         # the hint field with "yes" / "no" key caps
    hint_yes: str = "to type"                  # after the "yes" key cap
    hint_no: str = "to drop it"                # after the "no" key cap
    time_left: int = 0                         # confirm: whole seconds left (0 = no bar)
    time_total: int = 0                        # confirm: seconds the question was open for

    @property
    def panel(self) -> bool:
        return self.kind in ("confirm", "preview", "live", "done", "heard")


def shorten(text: str, n: int = CHIP_MAX_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def shorten_prompt(prompt: str) -> str:
    """'Insert "<a long dictation>" and send?' -> the quoted text shortened, so the question part stays."""
    prompt = re.sub(r'"(.*)"', lambda m: f'"{shorten(m.group(1), QUOTE_MAX_CHARS)}"', " ".join(prompt.split()),
                    count=1, flags=re.S)
    return shorten(prompt, PROMPT_MAX_CHARS)


@dataclass
class Question:
    id: str
    prompt: str
    opened_at: float
    expires_in_s: float


class CardModel:
    """What the card shows; no Tk, so the tests can drive it."""

    def __init__(self, clock=time.monotonic, heard_outside_dictation: bool = False,
                 command_flash: bool = True) -> None:
        self.clock = clock
        # A command that just ran ("Volume up", "New tab") shows in the pill for FLASH_S, in every mode: the user
        # sees it was heard without the card showing their words. Off: command_flash=False.
        self.command_flash = command_flash
        self._flash, self._flash_at = "", float("-inf")
        # The mic is always on, but what it hears (the waveform, the "Heard" chip) only shows in Dictation mode:
        # in Pointer mode speech is only checked for commands. Questions, a one-shot "dictate <words>" preview
        # and the Scroll / Paused / Menu pill still show in every mode.
        self.heard_outside_dictation = heard_outside_dictation
        self.preview: list[dict] = []
        self.original = ""
        self.mode = "pointer"
        self.heard, self._heard_at = "", float("-inf")
        self.live, self._live_at = "", float("-inf")     # what is being heard right now (dictation_live)
        self._listening_until = float("-inf")
        self.level = 0.0
        self.levels: deque[float] = deque([0.0] * WAVE_BARS, maxlen=WAVE_BARS)
        self._done, self._done_at = "", float("-inf")
        self._last_typed = False
        self.question: Question | None = None

    def on_ui(self, msg: dict) -> None:
        kind = msg.get("type")
        now = self.clock()
        if kind == messages.LISTENING:
            self.level = float(msg.get("level", 0.0) or 0.0)
            if msg.get("speaking"):
                self._listening_until = now + LISTEN_HOLD_S
        elif kind == messages.MODE:
            before, self.mode = self.mode, str(msg.get("mode", "pointer"))
            if self.mode != "dictation":
                self.live = ""
            if self.mode == "pointer" and before in LEAVE_FLASH:
                self._set_flash(LEAVE_FLASH[before], now)   # back home: say what just ended
        elif kind == messages.COMMAND:
            self._set_flash(str(msg.get("label", "")), now)
        elif kind == messages.DICTATION_PREVIEW:
            words = list(msg.get("words") or [])
            if not words and self.preview:
                self._done, self._done_at = ("Typed" if self._last_typed else "Dropped"), now
            self.preview = words
            self.original = str(msg.get("original", "")) if msg.get("cleaned") else ""
            self._last_typed = False
        elif kind == messages.DICTATION_LIVE:
            self.live, self._live_at = " ".join(str(msg.get("text", "")).split()), now
        elif kind == messages.CONFIRM:
            self._on_confirm(msg, now)

    def _live_words(self, now: float) -> list:
        """[(word, "live")] for the text being heard now; none outside Dictation or when it went stale."""
        if not self.live or self.mode != "dictation" or now - self._live_at > LIVE_STALE_S:
            return []
        return [(w, "live") for w in self.live.split()]

    def _set_flash(self, label: str, now: float) -> None:
        if self.command_flash and label.strip():
            self._flash, self._flash_at = shorten(label, 32), now

    def _on_confirm(self, msg: dict, now: float) -> None:
        qid, state = str(msg.get("id", "")), msg.get("state")
        if state == "open":
            self.question = Question(qid, str(msg.get("prompt", "")), now,
                                     max(0.0, float(msg.get("expires_in_s", 0) or 0)))
            return
        if self.question is None or self.question.id != qid:
            return                                       # not the open question: nothing to end
        self.question = None
        if now - self._done_at <= DONE_S and self._done == "Typed":
            return                                       # "Typed" already says it for this moment
        self._done, self._done_at = ("Done" if state == "confirmed" else "Cancelled"), now

    def on_typed(self) -> None:
        """os.type went out: the preview that closes next was typed, not dropped."""
        self._last_typed = True
        if not self.preview:
            self._done, self._done_at = "Typed", self.clock()

    def on_heard(self, text: str) -> None:
        self.heard, self._heard_at = text.strip(), self.clock()

    def tick_wave(self) -> None:
        """Called about 20 times a second by the Tk thread: one new bar for the scrolling waveform."""
        self.levels.append(self.level if self.listening() else 0.0)

    def listening(self) -> bool:
        return self.clock() <= self._listening_until

    def view(self) -> View:
        now = self.clock()
        pill, hint, listening = MODE_NAMES.get(self.mode, "[eye]able"), MODE_HINTS.get(self.mode, ""), self.listening()
        show_heard = self.heard_outside_dictation or self.mode == "dictation"
        listening = listening and show_heard
        if listening:
            pill, hint = "Listening", ""
        base = dict(pill=pill, pill_hint=hint, listening=listening)
        q = self.question
        if q and q.expires_in_s and now > q.opened_at + q.expires_in_s + CONFIRM_GRACE_S:
            self.question = q = None                     # its end message never came: don't look frozen
        if q:
            left = math.ceil(q.opened_at + q.expires_in_s - now) if q.expires_in_s else 0
            return View("confirm", label="Question", body=shorten_prompt(q.prompt), keys=True, hint_yes="",
                        hint_no="to cancel", time_left=max(0, left), time_total=math.ceil(q.expires_in_s),
                        **base)
        if self.preview:
            chip = self.original or (self.heard if now - self._heard_at <= 30 else "")
            return View("preview", label="Will type", chip=shorten(chip),
                        words=[(w["text"], float(w.get("conf", 1.0)) < SHAKY_CONF) for w in self.preview]
                        + self._live_words(now),
                        actions=["type", "drop"] + (["original"] if self.original else []), keys=True, **base)
        live = self._live_words(now)
        if live:
            return View("live", label="Hearing", words=live, **base)
        if now - self._done_at <= DONE_S:
            return View("done", body=self._done, **base)
        if now - self._flash_at <= FLASH_S and self._flash:
            return View("command", pill=self._flash)
        if show_heard and now - self._heard_at <= HEARD_S and self.heard:
            return View("heard", label="Heard", chip=shorten(self.heard), **base)
        if listening:
            return View("listening", **base)
        if self.mode in ("dictation", "scroll", "paused"):
            return View("mode", **base)
        return View("hidden")


class DictationCard:
    def __init__(self, bus, clock=time.monotonic, hide_from_capture: bool = True, opacity: float = 0.82,
                 position: str = "top", heard_outside_dictation: bool = False, command_flash: bool = True) -> None:
        self.bus = bus
        self.model = CardModel(clock, heard_outside_dictation=heard_outside_dictation, command_flash=command_flash)
        self.hide_from_capture = hide_from_capture
        self.opacity = min(1.0, max(0.3, float(opacity)))
        self.position = position
        self._lock = threading.Lock()
        self.page = None   # engine.core.page_mode.PageMode: the demo page draws its own card, so this one hides
        bus.subscribe(topics.TO_UI, self._on_ui)
        bus.subscribe(topics.TRANSCRIPT, self._on_transcript)
        bus.subscribe(topics.OS_TYPE, self._on_type)

    def _on_ui(self, msg: dict) -> None:
        with self._lock:
            self.model.on_ui(msg)

    def _on_transcript(self, t: dict) -> None:
        with self._lock:
            self.model.on_heard(str(t.get("text", "")))

    def _on_type(self, _payload: dict) -> None:
        with self._lock:
            self.model.on_typed()

    def view(self) -> tuple[View, tuple]:
        with self._lock:
            self.model.tick_wave()
            if self.page is not None and self.page.active:
                return View(kind="hidden"), ()
            return self.model.view(), tuple(round(v, 2) for v in self.model.levels)

    # ------------------------------------------------------------- Tk thread (the click ring's)
    def attach(self, root) -> None:
        """Build the card as a Toplevel of the ring's Tk root; call on that thread."""
        import tkinter as tk
        import tkinter.font as tkfont

        win = tk.Toplevel(root)
        win.overrideredirect(True)
        win.attributes("-topmost", True)
        win.configure(bg=KEY)
        if sys.platform == "win32":
            win.attributes("-transparentcolor", KEY)
        try:
            win.attributes("-alpha", self.opacity)
        except tk.TclError:
            pass
        canvas = tk.Canvas(win, width=PANEL_W + 4, height=PILL_H + 4, bg=KEY, highlightthickness=0, bd=0)
        canvas.pack()
        fonts = {
            "pill": tkfont.Font(root=root, family=FONT, size=11, weight="bold"),
            "pill_hint": tkfont.Font(root=root, family=FONT, size=10),
            "label": tkfont.Font(root=root, family=FONT, size=10),
            "chip": tkfont.Font(root=root, family=FONT, size=11, weight="bold"),
            "body": tkfont.Font(root=root, family=FONT, size=16),
            "shaky": tkfont.Font(root=root, family=FONT, size=16, underline=True),
            "action": tkfont.Font(root=root, family=FONT, size=11),
            "hint": tkfont.Font(root=root, family=FONT, size=10),
            "key": tkfont.Font(root=root, family=FONT, size=10, weight="bold"),
        }
        # Never withdrawn and shown again (that can take the focus): it stays mapped and moves off screen.
        win.geometry(OFF_SCREEN)
        win.update_idletasks()
        hwnd = _setup_window(win, self.hide_from_capture)
        shown = [None]

        def tick():
            try:
                v, levels = self.view()
                key = (v, levels if v.listening else None)
                if key != shown[0]:
                    shown[0] = key
                    if v.kind == "hidden":
                        win.geometry(OFF_SCREEN)
                    else:
                        h = draw(canvas, v, levels, fonts)
                        canvas.config(width=PANEL_W + 4, height=h)
                        sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
                        x = (sw - PANEL_W) // 2
                        y = TOP_GAP if self.position == "top" else sh - h - BOTTOM_GAP
                        win.geometry(f"{PANEL_W + 4}x{h}+{x}+{y}")
                        if hwnd:
                            win.update_idletasks()   # move it first: a SetWindowPos while Tk's move is still
                            _keep_on_top(hwnd)       # pending loses the move (the card stayed off screen)
            except Exception:
                pass
            win.after(POLL_MS, tick)

        win.after(POLL_MS, tick)


# ----------------------------------------------------------------- drawing (Tk thread)
def round_rect(c, x1, y1, x2, y2, r, **kw):
    r = max(1, min(r, (x2 - x1) / 2, (y2 - y1) / 2))
    pts = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
           x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
    return c.create_polygon(pts, smooth=True, **kw)


def fit(text: str, font, max_w: float) -> str:
    """The text, cut with an ellipsis so it fits max_w on one line (a chip never wraps)."""
    if font.measure(text) <= max_w:
        return text
    while text and font.measure(text + "\u2026") > max_w:
        text = text[:-1]
    return text.rstrip() + "\u2026"


MAX_PREVIEW_LINES = 9      # a longer passage shows its last lines (a card taller than the screen lost its buttons)


def clip_lines(lines, max_lines: int = MAX_PREVIEW_LINES):
    """(the last max_lines lines, how many words were left out before them)."""
    if len(lines) <= max_lines:
        return lines, 0
    return lines[-max_lines:], sum(len(line) for line in lines[:-max_lines])


def layout_words(words, font, shaky_font, max_w):
    """[(text, shaky)] -> [[(text, shaky, x)]] lines that fit max_w."""
    lines, line, x = [], [], 0
    space = font.measure(" ")
    for text, shaky in words:
        w = (shaky_font if shaky is True else font).measure(text)
        if line and x + w > max_w:
            lines.append(line)
            line, x = [], 0
        line.append((text, shaky, x))
        x += w + space
    if line:
        lines.append(line)
    return lines


def draw(c, v: View, levels, fonts) -> int:
    """Draw the whole card; returns its height."""
    c.delete("all")
    x0, w = 2, PANEL_W
    # --- pill
    pill_w = 52 + fonts["pill"].measure(v.pill)
    if v.listening:
        pill_w += WAVE_BARS * 6 + 12
    elif v.pill_hint:
        pill_w += 14 + fonts["pill_hint"].measure(v.pill_hint)
    px = x0 + (w - pill_w) // 2
    round_rect(c, px, 2, px + pill_w, 2 + PILL_H, PILL_H // 2, fill=PILL, outline=PANEL_EDGE)
    cx, cy = px + 24, 2 + PILL_H // 2
    c.create_oval(cx - 10, cy - 7, cx + 10, cy + 7, outline=TEXT, width=2)       # an eye
    c.create_oval(cx - 3, cy - 3, cx + 3, cy + 3, fill=TEXT, outline=TEXT)
    tx = px + 44
    if v.listening:
        for i, lv in enumerate(levels):
            bh = 4 + min(1.0, lv * 2.5) * 22
            bx = tx + i * 6
            c.create_rectangle(bx, cy - bh / 2, bx + 3, cy + bh / 2, fill=TEXT, outline="")
        tx += WAVE_BARS * 6 + 10
    c.create_text(tx, cy, text=v.pill, fill=TEXT, font=fonts["pill"], anchor="w")
    if v.pill_hint and not v.listening:
        c.create_text(tx + fonts["pill"].measure(v.pill) + 14, cy, text=v.pill_hint, fill=MUTED,
                      font=fonts["pill_hint"], anchor="w")
    if not v.panel:
        return PILL_H + 6
    # --- panel contents first (to know its height), then the panel behind them
    if v.kind == "heard":       # a short line: a panel just wide enough, centered under the pill
        w = min(PANEL_W, 2 * PAD + fonts["label"].measure(v.label) + CHIP_GAP + fonts["chip"].measure(v.chip) + 28)
        x0 = 2 + (PANEL_W - w) // 2
    elif v.kind == "done":
        w = max(240, 2 * PAD + fonts["body"].measure(v.body))
        x0 = 2 + (PANEL_W - w) // 2
    top = 2 + PILL_H + GAP
    y = top + PAD
    inner = w - 2 * PAD
    items_start = len(c.find_all())
    row_h = 0
    if v.label:
        c.create_text(x0 + PAD, y + 14, text=v.label, fill=MUTED, font=fonts["label"], anchor="w")
        row_h = 28
    if v.time_total:            # confirm: the seconds left, and a thin bar that empties
        c.create_text(x0 + w - PAD, y + 14, text=f"{v.time_left} s", fill=MUTED, font=fonts["label"], anchor="e")
        bx1, bx2 = x0 + PAD + fonts["label"].measure(v.label) + 14, x0 + w - PAD - fonts["label"].measure("00 s") - 10
        c.create_rectangle(bx1, y + 13, bx2, y + 15, fill=PANEL_EDGE, outline="")
        share = min(1.0, v.time_left / v.time_total)
        if share > 0:
            c.create_rectangle(bx1, y + 13, bx1 + (bx2 - bx1) * share, y + 15, fill=CHIP, outline="")
    if v.chip:
        room = inner - (fonts["label"].measure(v.label) + CHIP_GAP if v.label else 0)   # beside the label
        chip = fit(v.chip, fonts["chip"], room - 28)
        cw = fonts["chip"].measure(chip) + 28
        cx2 = x0 + w - PAD
        round_rect(c, cx2 - cw, y, cx2, y + 28, 14, fill=CHIP, outline=CHIP)
        c.create_text(cx2 - cw + 14, y + 14, text=chip, fill="#ffffff", font=fonts["chip"], anchor="w")
        row_h = 28
    y += row_h + (10 if row_h else 0)
    if v.words:
        line_h = fonts["body"].metrics("linespace") + 4
        lines, hidden = clip_lines(layout_words(v.words, fonts["body"], fonts["shaky"], inner))
        if hidden:
            c.create_text(x0 + PAD, y, text=f"\u2026 {hidden} earlier words", fill=MUTED, font=fonts["hint"],
                          anchor="nw")
            y += fonts["hint"].metrics("linespace") + 4
        for line in lines:
            for text, shaky, wx in line:
                c.create_text(x0 + PAD + wx, y, text=text,
                              fill=SHAKY if shaky is True else MUTED if shaky == "live" else TEXT,
                              font=fonts["shaky" if shaky is True else "body"], anchor="nw")
            y += line_h
        y += 10
    if v.body:                  # wraps to the panel; a long one adds lines (the model keeps it short)
        item = c.create_text(x0 + PAD, y, text=v.body, fill=TEXT, font=fonts["body"], anchor="nw", width=inner)
        box = c.bbox(item)
        y += max(fonts["body"].metrics("linespace"), (box[3] - box[1]) if box else 0) + 8
    if v.actions:
        ax = x0 + PAD
        ay = y + 14
        names = {"type": "Type it", "drop": "Drop it", "original": "Use original"}
        for i, a in enumerate(v.actions):
            if i:
                c.create_text(ax, ay, text="·", fill=MUTED, font=fonts["action"], anchor="w")
                ax += 18
            _icon(c, a, ax, ay)
            ax += 20
            c.create_text(ax, ay, text=names[a], fill=TEXT, font=fonts["action"], anchor="w")
            ax += fonts["action"].measure(names[a]) + 14
        y += 34
    if v.keys:
        fy = y + 8
        fh = 40
        field_start = len(c.find_all())
        sx, sy = x0 + PAD + 16, fy + fh // 2
        hints = (("Blink or say", False), ("yes", True), (v.hint_yes, False),
                 ("·  say", False), ("no", True), (v.hint_no, False))
        for text, is_key in (h for h in hints if h[0]):
            if is_key:
                kw = fonts["key"].measure(text) + 16
                round_rect(c, sx, sy - 11, sx + kw, sy + 11, 8, fill=KEYCAP, outline=PANEL_EDGE)
                c.create_text(sx + kw // 2, sy, text=text, fill=TEXT, font=fonts["key"])
                sx += kw + 8
            else:
                c.create_text(sx, sy, text=text, fill=MUTED, font=fonts["hint"], anchor="w")
                sx += fonts["hint"].measure(text) + 8
        field = round_rect(c, x0 + PAD, fy, x0 + w - PAD, fy + fh, fh // 2, fill=FIELD, outline=PANEL_EDGE)
        c.tag_lower(field, c.find_all()[field_start])   # behind its own text and key caps
        y = fy + fh
    bottom = y + PAD
    panel = round_rect(c, x0, top, x0 + w, bottom, R_PANEL, fill=PANEL, outline=PANEL_EDGE)
    first = c.find_all()[items_start] if len(c.find_all()) > items_start else None
    if first is not None:
        c.tag_lower(panel, first)
    return bottom + 2


def _icon(c, kind: str, x: float, y: float) -> None:
    if kind == "type":
        c.create_line(x, y, x + 4, y + 5, x + 12, y - 5, fill=TEXT, width=2)
    elif kind == "drop":
        c.create_line(x + 1, y - 5, x + 11, y + 5, fill=TEXT, width=2)
        c.create_line(x + 1, y + 5, x + 11, y - 5, fill=TEXT, width=2)
    else:
        c.create_arc(x, y - 6, x + 12, y + 6, start=30, extent=270, style="arc", outline=TEXT, width=2)
        c.create_line(x + 9, y - 8, x + 11, y - 4, x + 7, y - 3, fill=TEXT, width=2)


# ----------------------------------------------------------------- Windows window styles
def _keep_on_top(hwnd: int) -> None:
    from engine.eyes.dwell_meter import _keep_on_top as on_top

    on_top(hwnd)


def _setup_window(win, hide_from_capture: bool) -> int:
    """Windows: click-through, never focused, out of the taskbar, and (optionally) out of screenshots."""
    if sys.platform != "win32":
        return 0
    import ctypes

    from engine.eyes.dwell_meter import _click_through

    hwnd = _click_through(win)
    if hide_from_capture:
        try:
            ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
        except Exception:
            pass
    return hwnd
