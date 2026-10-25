"""Forgiving voice commands.

Turns a Whisper transcript into a Command. Fixed commands are fuzzy-matched
as whole phrases; "click/open/select <name>" keeps the name for UI-Venus.
The mode manager decides what a Command means in the current mode.

Forgiving means:
  * filler words are dropped: "um, scroll down please" -> "scroll down"
  * small mishearings still match: "sent it", "scrol down", "Klick compose"
  * each command has a few ways to say it: "scratch that" = "undo"
Safe means:
  * the WHOLE utterance must be the command: "I'll send it tomorrow" is not "send it"
  * short phrases ("yes", "no", "menu") must match exactly, so "go" never becomes "no"
  * target commands only count when the mode manager allows them: all three verbs in
    Pointer mode, only "click" in Dictation mode, so "click Send" always clicks while
    dictating "Open the door when you get home" stays text
One-shot typing: "type <words>" or "dictate <words>" in Pointer or Dictation mode puts just
those words in the preview (Command "type_text"); the mode manager then asks to type them.

Mixed speech: Whisper hands over everything said before a pause as one utterance, e.g.
"Click search box. Yeah." or "Yes. Yes. Click window." split_clauses() cuts it at sentence
punctuation, before every "click", and at a trigger word, and the mode manager reads each
clause on its own. Repeats collapse ("yes yes" = "yes").
Trigger words ("Venus, scroll down", "computer stop dictation"): whatever follows one is always
read as a command, in any mode, even in the middle of dictated text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import fuzz, process

# Phrase the user says -> command name the mode manager handles.
FIXED_COMMANDS: dict[str, str] = {
    "start dictation": "start_dictation",
    "dictation": "start_dictation",
    "dictate": "start_dictation",        # "dictate" alone; "dictate <words>" stays the one-shot
    "start typing": "start_dictation",
    "dictate on": "start_dictation",
    "dictation on": "start_dictation",
    "stop dictation": "stop_dictation",
    "stop typing": "stop_dictation",
    "done typing": "stop_dictation",
    "dictate off": "stop_dictation",
    "dictation off": "stop_dictation",
    "dictation of": "stop_dictation",    # how Whisper writes "dictation off" (Khang's live test, Oct 24)
    "dictate of": "stop_dictation",
    "stop dictating": "stop_dictation",
    "end dictation": "stop_dictation",
    "exit dictation": "stop_dictation",
    "close dictation": "stop_dictation",
    "turn off dictation": "stop_dictation",
    # greeting / goodbye (Khang, Oct 24 night): "Hi Dizzy" starts dictation, "Bye" / "Bye Dizzy" / "Bye bye" ends it.
    # "hey" is dropped as a filler word, so "Hey Dizzy" reaches us as "dizzy".
    "hi dizzy": "start_dictation",
    "hi dizzie": "start_dictation",
    "hi dizzi": "start_dictation",
    "high dizzy": "start_dictation",
    "dizzy": "start_dictation",
    "bye": "stop_dictation",
    "bye dizzy": "stop_dictation",
    "bye dizzie": "stop_dictation",
    "bye bye": "stop_dictation",
    # quit: stops the whole program, said alone (exact, never fuzzy: see EXACT_ONLY)
    "quit": "quit_app",
    "quit app": "quit_app",
    "quit program": "quit_app",
    "quit eyeable": "quit_app",
    "quit eye able": "quit_app",        # how Whisper often writes the name
    "quit i able": "quit_app",
    "quit dizzy": "quit_app",
    "exit program": "quit_app",
    "exit app": "quit_app",
    "shut down": "quit_app",
    "pause": "pause",
    # How Whisper writes a short "pause" (Khang's live test, Oct 24: "pot"). Said alone they mean nothing else
    "paws": "pause",
    "pot": "pause",
    "poss": "pause",
    "pots": "pause",
    "paw": "pause",
    "paused": "pause",
    "pause it": "pause",
    "stop listening": "pause",
    "pause listening": "pause",
    "take a break": "pause",       # longer ways to say it: easier to hear than one short word
    "go to sleep": "pause",
    "resume": "resume",
    "resumed": "resume",
    "resuming": "resume",
    "unpause": "resume",
    "wake up": "resume",
    "start listening": "resume",
    "continue": "resume",
    "menu": "menu",
    "open menu": "menu",
    "show menu": "menu",
    "cancel": "cancel",
    "never mind": "cancel",
    "scroll up": "scroll_up",
    "scroll down": "scroll_down",
    "send it": "send_it",
    "send": "send_it",
    "send message": "send_it",
    "scratch that": "undo",
    "undo": "undo",
    "undo that": "undo",
    "new line": "new_line",
    "next line": "new_line",
    # Answers. The card lists a few of each. Many ways to say them, so a misheard one is never a dead end
    # (Khang was stuck for 30 s when "no" kept coming out as "know"). "Okay" and "sure" are NOT yes: said in
    # passing, they confirmed "send" and "delete my voice data"
    "yes": "yes",
    "yeah": "yes",
    "yep": "yes",
    "yup": "yes",
    "yes please": "yes",
    "confirm": "yes",
    "do it": "yes",
    "go ahead": "yes",
    "write it": "yes",
    "write that": "yes",
    "type it": "yes",              # before this, "type it" was the one-shot "type <words>": it typed "it"
    "type that": "yes",
    "insert it": "yes",
    "put it in": "yes",
    "keep it": "yes",
    "yes write it": "yes",
    "yes type it": "yes",
    "no": "no",
    "nope": "no",
    "nah": "no",
    "no thanks": "no",
    "no thank you": "no",
    "don't write it": "no",
    "don't write that": "no",
    "don't type it": "no",
    "don't type that": "no",
    "do not write it": "no",
    "do not type it": "no",
    "drop it": "no",               # the card's own words for it
    "scrap it": "no",
    "scrap that": "no",
    "throw it away": "no",
    "discard": "no",
    "discard it": "no",
    "not that": "no",
    "no don't": "no",
    # Answering "Click Send?" (the mode manager reads these as yes when something waits; without them,
    # "click it" started a new search for a button named "It"). With nothing waiting: the snap click,
    # the left click at the cursor, onto the nearest button (UI-Venus)
    "click": "click_here",
    "click it": "click_here",
    "click that": "click_here",
    "click this": "click_here",
    "click here": "click_here",
    "click there": "click_here",
    "right click": "right_click",
    "go back": "go_back",
    "escape": "escape",
    "press escape": "escape",
    "calibrate": "calibrate",
    # Recenter (Khang, Oct 24 night): the head pointer drifts after a while; this puts the cursor in the middle
    # and takes "looking at it" as the new straight ahead. Whisper writes "Re-center." as "re center", which
    # used to fuzzy-match "enter" and press Enter.
    "recenter": "calibrate",
    "re center": "calibrate",
    "recentre": "calibrate",
    "re centre": "calibrate",
    "recenter cursor": "calibrate",
    "recenter the cursor": "calibrate",
    "center": "calibrate",
    "center cursor": "calibrate",
    "center the cursor": "calibrate",
    "reset cursor": "calibrate",
    "reset the cursor": "calibrate",
    "cursor center": "calibrate",
    "delete my voice data": "delete_voice_data",
    # Scroll mode: tilt the head to scroll until "stop" (exact only: "scroll" used to fuzzy-match "scroll up")
    "scroll": "scroll_mode",
    "scroll mode": "scroll_mode",
    "start scrolling": "scroll_mode",
    # Dwell (resting the cursor) is the left click: these turn it off / on
    "stop clicking": "dwell_off",
    "start clicking": "dwell_on",
    # Dictation clean-up: switch the preview between the cleaned text and the user's own words
    "use original": "use_original",
    "use the original": "use_original",
    "clean it up": "clean_up",
    # System controls (engine/eyes/os_output.py carries them out: media keys for volume, the laptop
    # screen's brightness through Windows)
    "volume up": "volume_up",
    "turn it up": "volume_up",
    "louder": "volume_up",
    "volume down": "volume_down",
    "turn it down": "volume_down",
    "quieter": "volume_down",
    "softer": "volume_down",
    "mute": "mute",
    "unmute": "mute",          # the mute key toggles
    "brightness up": "brightness_up",
    "brighter": "brightness_up",
    "brightness down": "brightness_down",
    "dimmer": "brightness_down",
    "darker": "brightness_down",
    # Keyboard shortcuts (KEY_SHORTCUTS below says which keys)
    "copy": "key_copy",
    "copy that": "key_copy",
    "paste": "key_paste",
    "paste it": "key_paste",
    "cut": "key_cut",
    "select all": "key_select_all",
    "save": "key_save",
    "save it": "key_save",
    "find": "key_find",
    "new tab": "key_new_tab",
    "new tap": "key_new_tab",            # how Whisper writes it (Khang's live test, Oct 24)
    "open new tab": "key_new_tab",
    "new window": "key_new_window",
    "open new window": "key_new_window",
    "close tab": "key_close_tab",
    "reopen tab": "key_reopen_tab",
    "next tab": "key_next_tab",
    "previous tab": "key_previous_tab",
    "last tab": "key_previous_tab",
    "refresh": "key_refresh",
    "reload": "key_refresh",
    "refresh page": "key_refresh",
    "go forward": "key_forward",
    "switch window": "key_switch_window",
    "switch windows": "key_switch_window",
    "minimize": "key_minimize",
    "minimize window": "key_minimize",
    "maximize": "key_maximize",
    "maximize window": "key_maximize",
    "show desktop": "key_show_desktop",
    "zoom in": "key_zoom_in",
    "zoom out": "key_zoom_out",
    "page up": "key_page_up",
    "page down": "key_page_down",
    "go to top": "key_top",
    "go to the top": "key_top",
    "go to bottom": "key_bottom",
    "go to the bottom": "key_bottom",
    "press enter": "key_enter",
    "enter": "key_enter",
    "press tab": "key_tab",
    "backspace": "key_backspace",
    "delete that": "key_backspace",
    # Asks first: Alt+F4 can close the wrong thing (the mode manager asks "Close this window?")
    "close window": "close_window",
    "close this window": "close_window",
    "close app": "close_window",
}

# Shortcut commands -> the keys os_input.combo presses (Windows; "cmd" is the Windows key)
KEY_SHORTCUTS: dict[str, str] = {
    "key_copy": "ctrl+c",
    "key_paste": "ctrl+v",
    "key_cut": "ctrl+x",
    "key_select_all": "ctrl+a",
    "key_save": "ctrl+s",
    "key_find": "ctrl+f",
    "key_new_tab": "ctrl+t",           # a new tab in Chrome, Edge, Firefox and Windows 11's File Explorer
    "key_new_window": "ctrl+n",        # a new WINDOW in browsers (Ctrl+N is not "new tab" there)
    "key_close_tab": "ctrl+w",
    "key_reopen_tab": "ctrl+shift+t",
    "key_next_tab": "ctrl+tab",
    "key_previous_tab": "ctrl+shift+tab",
    "key_refresh": "f5",
    "key_forward": "alt+right",
    "key_switch_window": "alt+tab",
    "key_minimize": "cmd+down",
    "key_maximize": "cmd+up",
    "key_show_desktop": "cmd+d",
    "key_zoom_in": "ctrl+=",
    "key_zoom_out": "ctrl+-",
    "key_page_up": "page_up",
    "key_page_down": "page_down",
    "key_top": "ctrl+home",
    "key_bottom": "ctrl+end",
    "key_enter": "enter",
    "key_tab": "tab",
    "key_backspace": "backspace",
}

# What the card flashes when a command runs ("Volume up"), so the user sees it was heard. Commands with their own
# feedback (questions and dictation: the card's panel; modes: the pill) aren't here; the spoken "click" and "right
# click" flash "Click" / "Right click" from the mode manager, and every click flashes where it lands (click_feedback).
COMMAND_LABELS: dict[str, str] = {
    "volume_up": "Volume up", "volume_down": "Volume down", "mute": "Mute",
    "brightness_up": "Brightness up", "brightness_down": "Brightness down",
    "key_copy": "Copy", "key_paste": "Paste", "key_cut": "Cut", "key_select_all": "Select all", "key_save": "Save",
    "key_find": "Find", "key_new_tab": "New tab", "key_new_window": "New window", "key_close_tab": "Close tab", "key_reopen_tab": "Reopen tab",
    "key_next_tab": "Next tab", "key_previous_tab": "Previous tab", "key_refresh": "Refresh",
    "key_forward": "Forward", "key_switch_window": "Switch window", "key_minimize": "Minimize",
    "key_maximize": "Maximize", "key_show_desktop": "Show desktop", "key_zoom_in": "Zoom in",
    "key_zoom_out": "Zoom out", "key_page_up": "Page up", "key_page_down": "Page down", "key_top": "Top",
    "key_bottom": "Bottom", "key_enter": "Enter", "key_tab": "Tab", "key_backspace": "Backspace",
    "scroll_up": "Scroll up", "scroll_down": "Scroll down", "go_back": "Back", "escape": "Escape",
    "undo": "Undo", "new_line": "New line", "quit_app": "Quitting",
}

# System commands -> the os.system action (os_output: volume by media keys, brightness through Windows)
SYSTEM_COMMANDS = {"volume_up", "volume_down", "mute", "brightness_up", "brightness_down"}

# The command list (python -m engine.voice.command_list prints it). EVERY command name above must have a row here:
# tests/test_command_list.py fails otherwise, so adding a command means writing down what it does.
# name -> (category, what it does). The list shows every phrase from FIXED_COMMANDS next to it.
COMMAND_INFO: dict[str, tuple[str, str]] = {
    "start_dictation": ("Dictation", "Starts dictation: what you say next shows on the card and is typed after you say yes. "
                                     "Also \"hi Dizzy\""),
    "stop_dictation": ("Dictation", "Ends dictation (asks first if words are waiting). Also \"bye\", \"bye Dizzy\", \"bye bye\". "
                                    "While dictating, \"close\", \"off\" and \"turn off\" also work"),
    "quit_app": ("Setup", "Quits EYEable completely: the engine, camera, mic and card all stop. Say \"quit\" alone; works "
                          "in every mode, even paused"),
    "send_it": ("Dictation", "Types the preview and presses Enter, after asking \"send?\""),
    "undo": ("Dictation", "Drops the preview, or presses Ctrl+Z when none is showing"),
    "new_line": ("Dictation", "Starts a new line in the preview"),
    "use_original": ("Dictation", "Types your own words instead of the cleaned-up text"),
    "clean_up": ("Dictation", "Types the cleaned-up text again"),
    "yes": ("Answers", "Says yes to a waiting question, preview or search (types the preview). While one waits, "
                       "\"yea\", \"ya\" and \"yah\" count too"),
    "no": ("Answers", "Says no to a waiting question, preview or search (drops the preview). While one waits, "
                      "\"know\", \"don't\" and \"do not\" count too"),
    "cancel": ("Answers", "Cancels a question or search, closes the menu, or drops the preview"),
    "click_here": ("Click and mouse", "Left click at the cursor, snapped onto the nearest button or link by UI-Venus, for a "
                                      "spot the cursor can't quite reach (a long blink clicks the exact pixel instead; "
                                      "answers a question when one waits)"),
    "right_click": ("Click and mouse", "Right click at the cursor"),
    "dwell_on": ("Click and mouse", "Turns on dwell clicking (rest the cursor to click)"),
    "dwell_off": ("Click and mouse", "Turns off dwell clicking"),
    "scroll_up": ("Scroll", "Scrolls up a little and stays in the current mode"),
    "scroll_down": ("Scroll", "Scrolls down a little and stays in the current mode"),
    "scroll_mode": ("Scroll", "Scroll mode: tilt your head chin up or down to scroll; say \"stop\" or \"done\" to leave"),
    "pause": ("Modes", "Pauses the engine: it ignores everything except \"resume\". \"Take a break\" is easier to hear "
                       "than \"pause\""),
    "resume": ("Modes", "Wakes the engine up again"),
    "menu": ("Modes", "Opens the menu (no menu screen yet; say \"cancel\" to close it)"),
    "calibrate": ("Setup", "Recenter: the cursor jumps to the middle; look at it for a second and it becomes straight "
                           "ahead. Also \"recenter\", \"center\", \"reset cursor\". Use it when the cursor drifts"),
    "delete_voice_data": ("Setup", "Deletes the saved voice data (asks first)"),
    "volume_up": ("Volume and screen", "Volume up 10%"),
    "volume_down": ("Volume and screen", "Volume down 10%"),
    "mute": ("Volume and screen", "Mutes or unmutes"),
    "brightness_up": ("Volume and screen", "Screen brightness up 10% (laptop screen only)"),
    "brightness_down": ("Volume and screen", "Screen brightness down 10% (laptop screen only, never below 10%)"),
    "key_copy": ("Editing", "Ctrl+C"),
    "key_paste": ("Editing", "Ctrl+V"),
    "key_cut": ("Editing", "Ctrl+X"),
    "key_select_all": ("Editing", "Ctrl+A"),
    "key_save": ("Editing", "Ctrl+S"),
    "key_find": ("Editing", "Ctrl+F"),
    "key_new_tab": ("Browser and tabs", "Ctrl+T: a new tab in Chrome, Edge, Firefox and Windows 11's File Explorer"),
    "key_new_window": ("Browser and tabs", "Ctrl+N: a new window"),
    "key_close_tab": ("Browser and tabs", "Ctrl+W: closes the current tab"),
    "key_reopen_tab": ("Browser and tabs", "Ctrl+Shift+T: reopens the tab you just closed"),
    "key_next_tab": ("Browser and tabs", "Ctrl+Tab"),
    "key_previous_tab": ("Browser and tabs", "Ctrl+Shift+Tab"),
    "key_refresh": ("Browser and tabs", "F5"),
    "go_back": ("Browser and tabs", "Alt+Left: back"),
    "key_forward": ("Browser and tabs", "Alt+Right: forward"),
    "key_zoom_in": ("Browser and tabs", "Ctrl+="),
    "key_zoom_out": ("Browser and tabs", "Ctrl+-"),
    "key_switch_window": ("Windows", "Alt+Tab"),
    "key_minimize": ("Windows", "Windows key + Down"),
    "key_maximize": ("Windows", "Windows key + Up"),
    "key_show_desktop": ("Windows", "Windows key + D"),
    "close_window": ("Windows", "Alt+F4, after asking \"Close this window?\""),
    "escape": ("Keys and moving around", "Esc"),
    "key_page_up": ("Keys and moving around", "Page Up"),
    "key_page_down": ("Keys and moving around", "Page Down"),
    "key_top": ("Keys and moving around", "Ctrl+Home: top of the page or document"),
    "key_bottom": ("Keys and moving around", "Ctrl+End: bottom of the page or document"),
    "key_enter": ("Keys and moving around", "Enter"),
    "key_tab": ("Keys and moving around", "Tab"),
    "key_backspace": ("Keys and moving around", "Backspace"),
}

# Said alone while dictating, these end Dictation mode (the mode manager checks; anywhere else they are words
# or nothing). Longer ways to say it ("dictate off", "stop dictation") are fixed commands and work everywhere.
DICTATION_EXIT = {"close", "off", "turn off", "turn it off", "exit", "stop", "done", "close it", "that's all",
                  "thats all", "i'm done", "im done"}

# Answers read as the answer ONLY while something waits for one (a question, a preview, a look-and-say search):
# how Whisper writes a short "no" or "yes" ("Know.", "Now."), and replies that are ordinary words anywhere else
# ("Don't."). With nothing waiting they are just words, and "I know" never matches (the whole utterance must be one).
# Before Oct 24 night this only worked for a question, so "Know." while a dictation preview waited was typed into it.
ANSWER_MISHEARINGS: dict[str, str] = {
    "know": "no", "noh": "no", "naw": "no", "nop": "no", "gnaw": "no",      # not "now": "now!" can mean "do it now"
    "don't": "no", "dont": "no", "do not": "no",
    "yea": "yes", "yah": "yes", "ya": "yes", "yas": "yes", "yess": "yes", "yesh": "yes", "ye": "yes",
}


def answer(text: str) -> str | None:
    """"yes" / "no" if the whole utterance is one of ANSWER_MISHEARINGS, else None (the mode manager asks this
    only while something waits for an answer)."""
    return ANSWER_MISHEARINGS.get(_collapse(normalize(text)))

# Verbs that start a look-and-say target request.
TARGET_VERBS = ("click", "open", "select")
DICTATION_TARGET_VERBS = ("click",)   # while dictating, only "click <name>" is a command...
DICTATION_MAX_LABEL_WORDS = 3          # ...as its own sentence with a short name ("click send", not
                                       # "I'll click the link later" or "click save before you close it")
TRIGGER_WORDS = ("venus", "computer")  # "Venus, click send": what follows is always a command
TYPE_VERBS = ("type", "dictate")      # "type see you at noon": one-shot dictation

# Minimum fuzzy scores (0-100). Kept high so chatter doesn't trigger commands.
FIXED_CUTOFF = 80
VERB_CUTOFF = 75
EXACT_ONLY_BELOW = 6       # phrases shorter than this (chars) must match exactly
# ...and these always: fuzzy matching made "I used the original" and "cleaning up" commands
EXACT_ONLY = {"use original", "use the original", "clean it up", "stop clicking", "start clicking",
              "scroll", "scroll mode", "start scrolling",
              # one word or close pairs: never fuzzy ("new tab" and "next tab", "louder" from "lower")
              "new tab", "new window", "next tab", "last tab", "louder", "quieter", "softer", "brighter", "dimmer", "darker",
              "refresh", "reload", "minimize", "maximize", "backspace", "go forward", "dictate on", "dictate off",
              "dictation on", "dictation off",
              # greeting, goodbye and quit: exact, so only these words ever start dictation or stop the program
              "hi dizzy", "hi dizzie", "hi dizzi", "high dizzy", "bye dizzy", "bye dizzie", "bye bye", "quit app",
              "quit program", "quit eyeable", "quit eye able", "quit i able", "quit dizzy", "exit program", "exit app", "shut down",
              # pause / resume and the answers: exact, so "passed" never pauses and "note that" never says no
              "paused", "pause it", "pause listening", "take a break", "go to sleep", "resumed", "resuming",
              "unpause", "continue", "yes please", "write that", "type that", "insert it", "put it in", "keep it",
              "no thanks", "drop it", "scrap it", "scrap that", "discard", "discard it", "not that", "no don't"}
SCROLL_WORDS = ("scroll", "scrolling")
NOT_A_SCROLL_REQUEST = {"stop", "done", "don't", "dont", "no", "not", "never", "quit", "end"}
MAX_COMMAND_WORDS = 5      # longer utterances are never fixed commands
MAX_LABEL_WORDS = 5        # "click <up to 5 words>"

FILLER_WORDS = {"um", "uh", "umm", "uhh", "er", "hmm", "please", "ok", "okay", "hey", "so"}
FILLER_PREFIXES = ("can you ", "could you ", "would you ")
LABEL_LEADING_WORDS = {"on", "the", "a", "an"}  # "click on the Send button" -> "Send Button"
POINTING_WORDS = {"it", "this", "that", "here", "there"}  # "click this video", "click it baby": the snap click


@dataclass(frozen=True)
class Command:
    name: str  # a FIXED_COMMANDS value, or "target"
    label: str = ""  # target name for "target", e.g. "Send"
    score: float = 100.0  # fuzzy match score 0-100


def normalize(text: str) -> str:
    """Lowercase, drop punctuation Whisper adds and filler words, collapse spaces."""
    text = re.sub(r"[^\w\s']", " ", text.lower().replace("’", "'"))
    text = " ".join(text.split())
    for prefix in FILLER_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix):]
    return " ".join(w for w in text.split() if w not in FILLER_WORDS)


_SHORT = {p for p in FIXED_COMMANDS if len(p) < EXACT_ONLY_BELOW}
_LONG = [p for p in FIXED_COMMANDS if len(p) >= EXACT_ONLY_BELOW and p not in EXACT_ONLY]


def _fixed(clean: str) -> Command | None:
    if clean in FIXED_COMMANDS:
        return Command(name=FIXED_COMMANDS[clean])
    if len(clean) < EXACT_ONLY_BELOW or len(clean.split()) > MAX_COMMAND_WORDS:
        return None
    match = process.extractOne(clean, _LONG, scorer=fuzz.ratio, score_cutoff=FIXED_CUTOFF)
    if match is None:
        return None
    phrase, score, _ = match
    return Command(name=FIXED_COMMANDS[phrase], score=score)


def _target(clean: str, verbs=TARGET_VERBS) -> Command | None:
    first, _, rest = clean.partition(" ")
    if not rest:
        return None
    verb = process.extractOne(first, verbs, scorer=fuzz.ratio, score_cutoff=VERB_CUTOFF)
    if verb is None:
        return None
    words = rest.split()
    while words and words[0] in LABEL_LEADING_WORDS:
        words.pop(0)
    if not words or len(words) > MAX_LABEL_WORDS:
        return None
    if words[0] in POINTING_WORDS:
        return Command(name="click_here", score=verb[1])   # the one under the cursor, not a button named "It"
    return Command(name="target", label=" ".join(words).title(), score=verb[1])


@dataclass
class Clause:
    text: str
    words: list[dict]          # Whisper words ({"text", "conf"}) in this clause
    triggered: bool = False    # came right after a trigger word: always read as a command
    starts_sentence: bool = True   # starts the utterance or follows sentence punctuation (not cut mid-sentence)


_END = re.compile(r"[.!?,;:]+[\"')\]]*$")


def split_clauses(words: list[dict]) -> list[Clause]:
    """Cut one utterance into clauses: after sentence punctuation, before every "click" that has real
    words before it (fillers like "um" don't count), and at a trigger word (dropped; the clause after
    it is triggered). Each clause knows if it starts a sentence: a "click" cut out of the middle of
    one ("I'll click the link later") doesn't."""
    clauses: list[Clause] = []
    cur: list[dict] = []
    triggered = pending = False
    starts = True
    trigger: dict | None = None          # the trigger word waiting for its clause
    glued = False                        # the clause before it was cut only by that trigger word
    trigger_of: dict[int, tuple[dict, bool]] = {}

    def close(after_punct: bool) -> bool:
        nonlocal cur, triggered, starts
        closed = bool(cur)
        if cur:
            clauses.append(Clause(" ".join(w["text"].strip() for w in cur), cur, triggered, starts))
            starts = after_punct
        cur, triggered = [], False
        return closed

    for w in words:
        raw = str(w.get("text", "")).strip()
        bare = re.sub(r"[^\w']", "", raw.lower())
        if not bare:
            continue
        if bare in TRIGGER_WORDS:
            glued = close(after_punct=False)
            pending, trigger = True, w
            continue
        if (bare == "click" and cur and normalize(" ".join(str(x.get("text", "")) for x in cur))
                and re.sub(r"[^\w']", "", str(cur[-1].get("text", "")).lower()) != "right"):   # "right click" stays one
            close(after_punct=False)
        if not cur and pending:
            triggered, pending = True, False
            trigger_of[len(clauses)] = (trigger, glued)
        cur.append(w)
        if _END.search(raw):
            close(after_punct=True)
    close(after_punct=False)
    if pending:                          # a trigger word at the very end: nothing follows it
        trigger_of[len(clauses)] = (trigger, glued)
        clauses.append(Clause("", [], True, starts))
    return _untrigger(clauses, trigger_of)


def _untrigger(clauses: list[Clause], trigger_of: dict[int, tuple[dict, bool]]) -> list[Clause]:
    """A trigger word only counts when a command follows it ("Venus, scroll down"). Otherwise it is
    just a word ("my computer is slow today"): put it back in its place in the text."""
    out: list[Clause] = []
    for i, c in enumerate(clauses):
        if not c.triggered or (c.words and parse(c.text) is not None):
            out.append(c)
            continue
        word, glued = trigger_of[i]
        words, starts = [word] + c.words, c.starts_sentence
        if glued and out:                # "my | computer | is slow today" was one clause
            prev = out.pop()
            words, starts = prev.words + words, prev.starts_sentence
        out.append(Clause(" ".join(w["text"].strip() for w in words), words, False, starts))
    return out


def wants_scroll(text: str) -> bool:
    """A short request that ends in "scroll" / "scrolling": "help me scroll", "I want to scroll".
    Not "stop scrolling", "don't scroll", "scroll down" (a one-shot) or a longer sentence about it."""
    words = normalize(text).split()
    return (2 <= len(words) <= 4 and words[-1] in SCROLL_WORDS and words[0] not in NOT_A_SCROLL_REQUEST
            and not {"up", "down"} & set(words))


def _collapse(clean: str) -> str:
    """'yes yes yes' -> 'yes': people repeat themselves when nothing seems to happen."""
    out: list[str] = []
    for w in clean.split():
        if not out or out[-1] != w:
            out.append(w)
    return " ".join(out)


def _type(clean: str) -> Command | None:
    first, _, rest = clean.partition(" ")
    if rest.split(" ", 1)[0] == "of":      # "type of food", "dictate of course": ordinary speech
        return None
    return Command(name="type_text", label=rest) if first in TYPE_VERBS and rest else None


def parse(text: str, allow_targets: bool = True, target_verbs=TARGET_VERBS) -> Command | None:
    """Return the best Command for a transcript, or None if nothing matches.

    allow_targets=False (Dictation mode) only matches fixed commands.
    Fixed commands win over targets, so "open menu" opens the menu.
    """
    clean = _collapse(normalize(text))
    if not clean:
        # "Okay." / "sure" alone are not yes: said in passing, they confirmed "send" and
        # "delete my voice data". "okay" stays a filler word, so "okay, click send" still works.
        return None
    return _fixed(clean) or (_target(clean, target_verbs) if allow_targets else None) or _type(clean)


if __name__ == "__main__":
    for sample in ["Click Send.", "click sen", "Klick compose", "Open the first result",
                   "click on the send button", "Start dictation.", "paws", "scroll down",
                   "um, scroll down please", "Send it!", "sent it", "Scratch that.",
                   "I'll send it tomorrow", "go", "hello there"]:
        print(f"{sample!r:28} -> {parse(sample)}")
