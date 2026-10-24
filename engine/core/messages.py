"""Message names and fields between the engine and the UI (JSON over WebSocket).
This is the contract with the front end. Keep ui/src/renderer/shared in sync.
Every message is a JSON object: {"type": <name>, ...fields}.
"""

# Engine -> UI
MODE = "mode"                            # mode: "pointer" | "dictation" | "menu" | "paused" | "scroll"
GAZE = "gaze"                            # x, y (screen px), confidence 0-1
GESTURE = "gesture"                      # gesture: "nod" | "shake" | "tilt_left" | "tilt_right" | "brow_raise" | "mouth_open" | "blink" (a long blink; not "type": that key is the message name)
DWELL = "dwell"                          # progress 0-1, ready true/false (resting on something clickable
                                         # and dwell armed: show "this clicks" before it fills). Sent when
                                         # either changes. ready is new; an old UI can ignore it
SCROLL = "scroll"                        # rate (notches/s, positive = up), offset (head tilt in degrees, chin down positive), ready (false while it learns "straight ahead", the first 0.4 s). About 10 a second in scroll mode
TARGET_SEARCHING = "target_searching"    # label; snap: true for a click with no name (label "here"), which
                                         # clicks as soon as it's found, no question
TARGET_FOUND = "target_found"            # x, y (screen px), label; snap: true as above
TARGET_NONE = "target_none"              # label; snap: true as above (it then clicks at the cursor)
CALIB_TARGET = "calib_target"            # index, x, y, total
CALIB_RESULT = "calib_result"            # error_px, ok
LISTENING = "listening"                  # level 0-1, speaking true/false
DICTATION_PREVIEW = "dictation_preview"  # words: [{"text": str, "conf": float}]; [] hides the bubble.
                                         # With the clean-up on, also cleaned: true|false, and when true
                                         # original: the user's own words (words then holds the cleaned text)
STATUS = "status"                        # level: "info" | "warn" | "error", text
CONFIRM = "confirm"                      # id, prompt, state: "open" | "confirmed" | "cancelled" | "expired", expires_in_s

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
CONFIRM_ANSWER = "confirm_answer"        # id, yes: true/false (a button, same as nod / shake)


def make(msg_type: str, **fields) -> dict:
    """Build a message dict, e.g. make(TARGET_FOUND, x=812, y=440, label="Send")."""
    if "type" in fields:
        raise ValueError("'type' is the message name; give the field another name")
    return {"type": msg_type, **fields}
