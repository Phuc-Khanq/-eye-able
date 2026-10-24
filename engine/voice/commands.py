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
  * target commands only count when the mode manager allows them (Pointer mode),
    so dictating "Open the door when you get home" stays text
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import fuzz, process

# Phrase the user says -> command name the mode manager handles.
FIXED_COMMANDS: dict[str, str] = {
    "start dictation": "start_dictation",
    "dictation": "start_dictation",
    "start typing": "start_dictation",
    "stop dictation": "stop_dictation",
    "stop typing": "stop_dictation",
    "done typing": "stop_dictation",
    "pause": "pause",
    "paws": "pause",  # common Whisper mishearing
    "stop listening": "pause",
    "resume": "resume",
    "wake up": "resume",
    "start listening": "resume",
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
    "yes": "yes",
    "yeah": "yes",
    "yep": "yes",
    "confirm": "yes",
    "no": "no",
    "nope": "no",
    "calibrate": "calibrate",
    "delete my voice data": "delete_voice_data",
}

# Verbs that start a look-and-say target request.
TARGET_VERBS = ("click", "open", "select")

# Minimum fuzzy scores (0-100). Kept high so chatter doesn't trigger commands.
FIXED_CUTOFF = 80
VERB_CUTOFF = 75
EXACT_ONLY_BELOW = 6       # phrases shorter than this (chars) must match exactly
MAX_COMMAND_WORDS = 5      # longer utterances are never fixed commands
MAX_LABEL_WORDS = 5        # "click <up to 5 words>"

FILLER_WORDS = {"um", "uh", "umm", "uhh", "er", "hmm", "please", "ok", "okay", "hey", "so"}
FILLER_PREFIXES = ("can you ", "could you ", "would you ")
LABEL_LEADING_WORDS = {"on", "the", "a", "an"}  # "click on the Send button" -> "Send Button"


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
_LONG = [p for p in FIXED_COMMANDS if len(p) >= EXACT_ONLY_BELOW]


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


def _target(clean: str) -> Command | None:
    first, _, rest = clean.partition(" ")
    if not rest:
        return None
    verb = process.extractOne(first, TARGET_VERBS, scorer=fuzz.ratio, score_cutoff=VERB_CUTOFF)
    if verb is None:
        return None
    words = rest.split()
    while words and words[0] in LABEL_LEADING_WORDS:
        words.pop(0)
    if not words or len(words) > MAX_LABEL_WORDS:
        return None
    return Command(name="target", label=" ".join(words).title(), score=verb[1])


def parse(text: str, allow_targets: bool = True) -> Command | None:
    """Return the best Command for a transcript, or None if nothing matches.

    allow_targets=False (Dictation mode) only matches fixed commands.
    Fixed commands win over targets, so "open menu" opens the menu.
    """
    clean = normalize(text)
    if not clean:
        return None
    return _fixed(clean) or (_target(clean) if allow_targets else None)


if __name__ == "__main__":
    for sample in ["Click Send.", "click sen", "Klick compose", "Open the first result",
                   "click on the send button", "Start dictation.", "paws", "scroll down",
                   "um, scroll down please", "Send it!", "sent it", "Scratch that.",
                   "I'll send it tomorrow", "go", "hello there"]:
        print(f"{sample!r:28} -> {parse(sample)}")
