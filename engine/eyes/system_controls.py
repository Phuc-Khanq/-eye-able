"""Volume and screen brightness by voice ("volume up", "mute", "brightness down").

Route: Khang and Hung (Khang leads)
Publishes: nothing; os_output calls these on its worker thread and shows a status line
APIs: os_input.combo with pynput's media keys (volume); Windows WMI through PowerShell (brightness)

Volume: Windows' own volume keys, so the on-screen volume bar appears as with a keyboard. One key press is 2%,
so "volume up" presses VOLUME_PRESSES times (10%). "mute" toggles.
Brightness: WmiMonitorBrightness, which only a laptop's built-in screen supports (external monitors say no, and
the status line says so). BRIGHTNESS_STEP per command, never under BRIGHTNESS_MIN so the screen can't go black.
PowerShell runs without a window or profile; the script is fixed text plus one integer, never user words.
"""
from __future__ import annotations

import subprocess
import sys

VOLUME_PRESSES = 5        # 5 x 2% = 10% per "volume up"
BRIGHTNESS_STEP = 10      # percent per "brightness up/down"
BRIGHTNESS_MIN = 10       # never darker than this
CREATE_NO_WINDOW = 0x08000000

VOLUME_KEYS = {"volume_up": "media_volume_up", "volume_down": "media_volume_down", "mute": "media_volume_mute"}

_BRIGHTNESS_PS = (
    "$ErrorActionPreference='Stop';"
    "$b=(Get-CimInstance -Namespace root/WMI -ClassName WmiMonitorBrightness | Select-Object -First 1)"
    ".CurrentBrightness;"
    "$n=[Math]::Max({lo},[Math]::Min(100,[int]$b+({delta})));"
    "Get-CimInstance -Namespace root/WMI -ClassName WmiMonitorBrightnessMethods | Select-Object -First 1 | "
    "Invoke-CimMethod -MethodName WmiSetBrightness -Arguments @{{Timeout=1;Brightness=[byte]$n}} | Out-Null;"
    "$n"
)


class BrightnessUnsupported(RuntimeError):
    """No built-in screen that Windows lets us dim (an external monitor, or not Windows)."""


def volume(action: str, combo, presses: int = VOLUME_PRESSES) -> str:
    """Press the volume key for action ("volume_up" | "volume_down" | "mute"); returns the status text."""
    key = VOLUME_KEYS[action]
    for _ in range(1 if action == "mute" else presses):
        combo(key)
    return {"volume_up": "Volume up", "volume_down": "Volume down", "mute": "Mute on/off"}[action]


def brightness(delta: int, run=subprocess.run, platform: str = sys.platform) -> int:
    """Change the built-in screen's brightness by delta percent; returns the new level (BRIGHTNESS_MIN to 100)."""
    if platform != "win32":
        raise BrightnessUnsupported("brightness by voice works on Windows only")
    script = _BRIGHTNESS_PS.format(lo=BRIGHTNESS_MIN, delta=int(delta))
    kw = {"creationflags": CREATE_NO_WINDOW} if platform == "win32" and sys.platform == "win32" else {}
    try:
        out = run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                  capture_output=True, text=True, timeout=10, **kw)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BrightnessUnsupported(f"PowerShell didn't answer: {exc}") from exc
    lines = [ln.strip() for ln in (out.stdout or "").splitlines() if ln.strip()]
    if out.returncode != 0 or not lines or not lines[-1].isdigit():
        raise BrightnessUnsupported("this screen's brightness can't be changed (an external monitor?)")
    return int(lines[-1])


def run_action(action: str, os_api, run=subprocess.run, platform: str = sys.platform) -> str:
    """Carry out an os.system action; returns the status text for the UI."""
    if action in VOLUME_KEYS:
        return volume(action, os_api.combo)
    if action in ("brightness_up", "brightness_down"):
        level = brightness(BRIGHTNESS_STEP if action == "brightness_up" else -BRIGHTNESS_STEP, run, platform)
        return f"Brightness {level}%"
    raise ValueError(f"unknown system action {action!r}")
