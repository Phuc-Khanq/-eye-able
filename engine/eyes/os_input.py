"""Moves the real cursor, clicks, scrolls and types. Only the mode manager calls click and type.

Route: Khang and Hung (Hung leads)
Publishes: nothing
Subscribes: cursor moves from smoothing; click and type requests from mode_manager
APIs: pynput mouse and keyboard; ctypes SetProcessDpiAwarenessContext so pixels match at 125% and 150%
"""

def make_dpi_aware() -> None:
    """Call once at startup, before any screenshot or cursor move."""
    raise NotImplementedError


def move(x: int, y: int) -> None:
    raise NotImplementedError


def click(x: int, y: int) -> None:
    raise NotImplementedError


def type_text(text: str) -> None:
    raise NotImplementedError
