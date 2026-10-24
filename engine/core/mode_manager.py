"""State machine that decides what each gaze, gesture and command means right now.

Route: Khoi and Khoa (Khoi leads)
Publishes: mode, target_searching, TargetRequest, click requests to os_input, dictation inserts
Subscribes: GazePoint, Gesture, commands, TargetResult, UI messages
APIs: plain Python (enum)
"""

from enum import Enum


class Mode(Enum):
    POINTER = "pointer"
    DICTATION = "dictation"
    MENU = "menu"
    PAUSED = "paused"


class ModeManager:
    """Only this module asks os_input to click or type, so stray signals can't act alone.

    Look-and-say: command "click <name>" -> TargetRequest -> wait for TargetResult ->
    target_found to the UI -> nod clicks, shake cancels.
    """

    def __init__(self, bus) -> None:
        self.bus = bus
        self.mode = Mode.POINTER
