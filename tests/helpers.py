"""Test helpers: a synchronous stand-in for Khoa's EventBus, plus fake Whisper output."""
from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace


class FakeBus:
    """Same subscribe/publish as engine.core.events.EventBus, but runs handlers right away
    and records everything so tests can check what was published."""

    def __init__(self) -> None:
        self._handlers = defaultdict(list)
        self.log: list[tuple[str, object]] = []

    def subscribe(self, topic, handler) -> None:
        self._handlers[topic].append(handler)

    def publish(self, topic, payload) -> None:
        self.log.append((topic, payload))
        for handler in list(self._handlers[topic]):
            handler(payload)

    def of(self, topic) -> list:
        return [p for t, p in self.log if t == topic]

    def ui(self, msg_type: str) -> list[dict]:
        from engine.core import topics

        return [p for p in self.of(topics.TO_UI) if p["type"] == msg_type]


class FakeWhisperModel:
    """Looks like faster_whisper.WhisperModel. The "audio" is the text to return.
    A word ending in "?" comes back with low confidence: "meet at nine? today"."""

    def transcribe(self, audio, **kwargs):
        words = []
        for token in str(audio).split():
            conf = 0.3 if token.endswith("?") and len(token) > 1 else 0.95
            words.append(SimpleNamespace(word=" " + token.rstrip("?"), probability=conf))
        seg = SimpleNamespace(words=words, no_speech_prob=0.01)
        return iter([seg] if words else []), SimpleNamespace(language="en")
