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
    "do it": "yes",
    "go ahead": "yes",
    # Answering "Click Send?": without these, "click it" started a new search for a button named "It"
    "click": "yes",
    "click it": "yes",
    "click that": "yes",
    "no": "no",
    "nope": "no",
    "calibrate": "calibrate",
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
}

# Answers Whisper mishears, read as the answer only while a question is open ("Click Send?" -> "Know."):
# anywhere else "know" is just a word, and "I know" never matches
ANSWER_MISHEARINGS: dict[str, str] = {"know": "no"}

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
              "scroll", "scroll mode", "start scrolling"}
SCROLL_WORDS = ("scroll", "scrolling")
NOT_A_SCROLL_REQUEST = {"stop", "done", "don't", "dont", "no", "not", "never", "quit", "end"}
MAX_COMMAND_WORDS = 5      # longer utterances are never fixed commands
MAX_LABEL_WORDS = 5        # "click <up to 5 words>"
IT_EXTRA_WORDS = 2         # "click it" + up to 2 words ("now", "again", "right now") is a yes

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
    if verb[0] == "click" and words[0] == "it" and len(words) <= 1 + IT_EXTRA_WORDS:
        return Command(name="yes", score=verb[1])   # "click it now" / "click it again": yes, never a button "It"
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
        if bare == "click" and cur and normalize(" ".join(str(x.get("text", "")) for x in cur)):
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
