"""Moves the real cursor, clicks, scrolls and types. Only the mode manager calls click and type.

Route: Khang and Hung (Hung leads)
Publishes: nothing
Subscribes: cursor moves from smoothing; click and type requests from mode_manager
APIs: pynput mouse and keyboard; ctypes SetProcessDpiAwarenessContext so pixels match at 125% and 150%

All coordinates are physical screen pixels, the same pixels mss screenshots use, once
make_dpi_aware() has run. pynput is imported on first use so the engine still starts on
machines without a desktop session.
"""
import sys
import time

_mouse = None
_keyboard = None
_dpi_aware = False
_screen: tuple[int, int] | None = None

# DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 is the handle value -4.
PER_MONITOR_AWARE_V2 = -4
CLICK_SETTLE_S = 0.03   # let the app see the cursor arrive before the press


def make_dpi_aware() -> None:
    """Call once at startup, before any screenshot or cursor move."""
    global _dpi_aware
    if _dpi_aware or sys.platform != "win32":
        _dpi_aware = True
        return
    import ctypes

    user32 = ctypes.windll.user32
    try:
        ok = user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(PER_MONITOR_AWARE_V2))
    except AttributeError:  # older than Windows 10 1703
        ok = False
    if not ok:
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
        except (AttributeError, OSError):
            user32.SetProcessDPIAware()
    _dpi_aware = True


def screen_size() -> tuple[int, int]:
    """Primary screen size in physical pixels, read once."""
    global _screen
    if _screen is None:
        _screen = _read_screen_size()
    return _screen


def _read_screen_size() -> tuple[int, int]:
    make_dpi_aware()
    if sys.platform == "win32":
        import ctypes

        user32 = ctypes.windll.user32
        return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))  # SM_CXSCREEN, SM_CYSCREEN
    try:
        import mss

        with mss.mss() as sct:
            mon = sct.monitors[1]
            return int(mon["width"]), int(mon["height"])
    except Exception:
        return 1920, 1080


def _get_mouse():
    global _mouse
    if _mouse is None:
        make_dpi_aware()
        from pynput.mouse import Controller

        _mouse = Controller()
    return _mouse


def _get_keyboard():
    global _keyboard
    if _keyboard is None:
        from pynput.keyboard import Controller

        _keyboard = Controller()
    return _keyboard


def move(x: int, y: int) -> None:
    w, h = screen_size()
    _get_mouse().position = (min(max(int(round(x)), 0), w - 1), min(max(int(round(y)), 0), h - 1))


def position() -> tuple[int, int]:
    x, y = _get_mouse().position
    return int(x), int(y)


def click(x: int, y: int, button: str = "left", count: int = 1) -> None:
    """Move to (x, y) and click. button is "left" or "right"; count=2 double-clicks."""
    from pynput.mouse import Button

    move(x, y)
    time.sleep(CLICK_SETTLE_S)
    _get_mouse().click(Button.right if button == "right" else Button.left, count)


def scroll(dy: int, dx: int = 0) -> None:
    """Scroll by notches; dy > 0 scrolls up, like a mouse wheel."""
    _get_mouse().scroll(dx, dy)


def type_text(text: str) -> None:
    _get_keyboard().type(text)


def press(key: str) -> None:
    """Press and release a named key, e.g. "enter", "backspace", "tab", "esc"."""
    from pynput.keyboard import Key

    _get_keyboard().tap(getattr(Key, key))
