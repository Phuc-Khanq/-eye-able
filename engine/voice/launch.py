"""Opening apps and sites by voice ("tier 2" OS control): which spoken phrases start something.

Route: Khoi and Khoa (Khoa leads)
Publishes: nothing itself. The mode manager (ModeManager._launch_payload) calls match() and publishes
           topics.OS_LAUNCH with the result; engine/eyes/os_output.py starts it
Subscribes: nothing
APIs: re, urllib.parse (standard library only)

    match("open Chrome", settings)            -> {"app": "chrome"}
    match("go to YouTube dot com", settings)  -> {"url": "https://youtube.com"}
    match("search for cats", settings)        -> {"url": "https://www.google.com/search?q=cats"}
    match("open compose", settings)           -> None   (not on the list: look-and-say keeps it)

Short list, strict matching:
  * "open / start / launch <app>" only for a name in settings.launch_apps (or settings.launch_aliases).
    Any other name returns None, so "open compose" stays look-and-say (UI-Venus finds a button).
  * "go to <site>": one clean host name ("youtube.com", "YouTube dot com"), optionally with a path, or a
    one-word name from settings.launch_sites ("youtube" -> youtube.com). Any other word with no dot is
    not a site: "go to sleep" is None. Only ever https://.
  * "search for <words>" / "search <words>": settings.launch_search_url with the words url-encoded.
  * The WHOLE utterance must be the command, like commands.py: "I want to open Chrome tomorrow" is None.
  * Fixed commands always win: "open menu", "start dictation", "go back" are never launches.
Filler words and Whisper's capitals and punctuation are ignored ("Um, open Chrome please.").
"""

from __future__ import annotations

import re
from urllib.parse import quote_plus

from engine.voice import commands

VERBS = ("open", "start", "launch")
APP_LEADING_WORDS = {"the"}            # "open the settings"
# A host name: labels of letters, digits and inner hyphens, joined by dots, then an optional path.
_LABEL = r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?"
_HOST = re.compile(rf"^{_LABEL}(?:\.{_LABEL})*(?:/[a-z0-9\-._~/%?=&+#]*)?$")
_TRIGGER = re.compile(rf"^\W*(?:{'|'.join(commands.TRIGGER_WORDS)})\b\W*", re.IGNORECASE)

# Example phrases (the message to Khoi and the tests use them).
PHRASES = [
    "open chrome", "open google chrome", "launch edge", "start notepad", "open file explorer",
    "open settings", "open mail", "go to youtube.com", "go to youtube dot com", "go to wikipedia.org",
    "search for cats", "search for how to bake bread",
]


def strip_trigger(text: str) -> tuple[str, bool]:
    """("go to youtube.com", True) for "Venus, go to youtube.com". commands.split_clauses only takes the
    trigger word off a clause commands.parse knows ("Venus, open Chrome"), so "go to" and "search"
    clauses still start with it."""
    m = _TRIGGER.match(text)
    return (text[m.end():], True) if m else (text, False)


def _words(text: str) -> list[str]:
    """Lowercase words with fillers and Whisper's punctuation gone; dots and slashes kept for sites."""
    text = text.lower().replace("’", "'").strip()
    text = re.sub(r"\s+dot\s+", ".", f" {text} ").strip()          # "youtube dot com" -> "youtube.com"
    text = re.sub(r"[^\w\s'./+#-]", " ", text)                   # commas, ?, !, quotes, colons ("C++" stays)
    words = [w.strip(".'-/") for w in text.split()]              # "chrome." -> "chrome"
    text = " ".join(w for w in words if w)
    for prefix in commands.FILLER_PREFIXES:                      # "can you open chrome"
        if text.startswith(prefix):
            text = text[len(prefix):]
    return [w for w in text.split() if w not in commands.FILLER_WORDS]


def _app(rest: list[str], settings) -> dict | None:
    while rest and rest[0] in APP_LEADING_WORDS:
        rest = rest[1:]
    name = " ".join(rest)
    apps = getattr(settings, "launch_apps", {}) or {}
    name = (getattr(settings, "launch_aliases", {}) or {}).get(name, name)
    return {"app": name} if name in apps else None


def _site(rest: list[str], settings) -> dict | None:
    if len(rest) != 1 or not _HOST.match(rest[0]):
        return None
    host, _, path = rest[0].partition("/")
    if "." not in host:                                          # "go to youtube" -> youtube.com
        host = str((getattr(settings, "launch_sites", {}) or {}).get(host, ""))
        if not _HOST.match(host) or "." not in host:
            return None                                          # "go to sleep": not a site
    return {"url": f"https://{host}" + (f"/{path}" if path else "")}


def _search(rest: list[str], settings) -> dict | None:
    if rest and rest[0] == "for":
        rest = rest[1:]
    if not rest:
        return None
    url = str(getattr(settings, "launch_search_url", "")).replace("{q}", quote_plus(" ".join(rest)))
    return {"url": url} if url.startswith("https://") else None


def match(text: str, settings=None) -> dict | None:
    """The os.launch payload for a spoken phrase, or None (not for us: fall through to look-and-say).

    {"app": <a key of settings.launch_apps>} or {"url": <an https:// URL>}. Pure: starts nothing.
    """
    if settings is None:
        from engine.config import settings
    if not text or not text.strip():
        return None
    cmd = commands.parse(text, allow_targets=False)
    if cmd is not None and cmd.name != "type_text":
        return None                                              # "open menu", "start dictation", "go back"
    words = _words(text)
    if len(words) < 2:
        return None
    first, rest = words[0], words[1:]
    if first in VERBS:
        return _app(rest, settings)
    if first == "go" and rest[0] == "to":
        return _site(rest[1:], settings)
    if first == "search":
        return _search(rest, settings)
    return None
