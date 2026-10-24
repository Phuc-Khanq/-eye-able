"""Message names and fields between the engine and the UI (JSON over WebSocket).
This is the contract with the front end. Keep ui/src/renderer/shared in sync.
Every message is a JSON object: {"type": <name>, ...fields}.
"""

# Engine -> UI
MODE = "mode"                            # mode: "pointer" | "dictation" | "menu" | "paused"
GAZE = "gaze"                            # x, y (screen px), confidence 0-1
GESTURE = "gesture"                      # type: "nod" | "shake" | "tilt_left" | "tilt_right"
DWELL = "dwell"                          # progress 0-1
TARGET_SEARCHING = "target_searching"    # label
TARGET_FOUND = "target_found"            # x, y (screen px), label
TARGET_NONE = "target_none"              # label
CALIB_TARGET = "calib_target"            # index, x, y, total
CALIB_RESULT = "calib_result"            # error_px, ok
LISTENING = "listening"                  # level 0-1, speaking true/false
DICTATION_PREVIEW = "dictation_preview"  # words: [{"text": str, "conf": float}]
STATUS = "status"                        # level: "info" | "warn" | "error", text

# Engine -> UI, only while the console is open
DEBUG_FRAME = "debug_frame"              # jpeg (base64), 640 px wide
DEBUG_FACE = "debug_face"                # eye, iris and head-pose points
DEBUG_FEATURES = "debug_features"        # h, v, lid per eye
TARGET_DEBUG = "target_debug"            # crop (base64 JPEG), x, y, ms

# UI -> engine
SET_MODE = "set_mode"                    # mode
CALIBRATE = "calibrate"                  # action: "start" | "cancel"
MENU_SELECT = "menu_select"              # item
SETTINGS = "settings"                    # pause_seconds, dwell_ms, sensitivity
DICTATION = "dictation"                  # action: "insert" | "discard" | "edit_word"
TARGET = "target"                        # action: "click" | "cancel"
CONSOLE = "console"                      # open: true/false


def make(msg_type: str, **fields) -> dict:
    """Build a message dict, e.g. make(TARGET_FOUND, x=812, y=440, label="Send")."""
    return {"type": msg_type, **fields}
