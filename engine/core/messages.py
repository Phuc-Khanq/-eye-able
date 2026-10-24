"""Message names and fields between the engine and the UI (JSON over WebSocket).
This is the contract with the front end. Keep ui/src/renderer/shared in sync.
Every message is a JSON object: {"type": <name>, ...fields}.
"""

# Engine -> UI. Fields as the engine really sends them; tests/test_recordings.py checks every message
# in tests/recordings/all_messages.jsonl against FIELDS below, so keep the two in sync.
MODE = "mode"                            # mode: "pointer" | "dictation" | "menu" | "paused" | "scroll". On every change,
                                         # and once when a UI connects
GAZE = "gaze"                            # x, y (screen px, 0.1 px), confidence 0-1. Every camera frame with a face
GESTURE = "gesture"                      # gesture: "nod" | "tilt_left" | "tilt_right" | "brow_raise" | "blink" (a long
                                         # blink). Only in menu mode (the UI moves / picks in the menu); a shake there
                                         # closes it (a mode message). "shake" and "mouth_open" aren't sent yet.
                                         # Not "type": that key is the message name
DWELL = "dwell"                          # progress 0-1 (2 decimals), ready true/false (resting on something clickable and
                                         # dwell armed: show "this clicks" before it fills). Sent when either changes:
                                         # fills while the cursor rests, 1.0 at the click, then 0. Dwell is off by default
SCROLL = "scroll"                        # rate (notches/s, positive = up, 0.1), offset (head tilt in degrees, chin down
                                         # positive), ready (false while it learns "straight ahead", the first 0.4 s).
                                         # About 10 a second in scroll mode, and one rate=0, offset=0, ready=false at the end
TARGET_SEARCHING = "target_searching"    # label ("Search Bar"): UI-Venus is looking. snap: true (only then) for a click
                                         # with no name (a raise or "click"; label "here"): it clicks as soon as it's
                                         # found, no question
TARGET_FOUND = "target_found"            # x, y (screen px), label; snap: true as above. Without snap a confirm
                                         # "Click <label>?" follows
TARGET_NONE = "target_none"              # label: not found, or the search timed out (6 s); snap: true as above (it then
                                         # clicks at the cursor)
CALIB_TARGET = "calib_target"            # index, x, y (screen px), total. One-look calibration: index 0, total 1, the center
CALIB_RESULT = "calib_result"            # error_px (how far the gaze was shifted, 0.1 px), ok (false = cancelled)
LISTENING = "listening"                  # level 0-1, speaking true/false. About 10 a second while the mic is on
DICTATION_PREVIEW = "dictation_preview"  # words: [{"text": str, "conf": float}]; [] hides the bubble.
                                         # With the clean-up on, also cleaned: true|false; when true, words holds the
                                         # cleaned text and original the user's own words (a string)
STATUS = "status"                        # level: "info" | "warn" | "error", text (show it, then let it fade)
CONFIRM = "confirm"                      # id ("q1"), prompt, state: "open" | "confirmed" | "cancelled" | "expired",
                                         # expires_in_s (12.0 when open, 0 otherwise). Same id from open to the end

# Engine -> UI, only while the console is open. Not sent by the engine yet (names reserved)
DEBUG_FRAME = "debug_frame"              # jpeg (base64), 640 px wide
DEBUG_FACE = "debug_face"                # eye, iris and head-pose points
DEBUG_FEATURES = "debug_features"        # h, v, lid per eye
TARGET_DEBUG = "target_debug"            # crop (base64 JPEG), x, y, ms

# The fields of every Engine -> UI message the engine sends: required, then optional
FIELDS: dict[str, tuple[set[str], set[str]]] = {
    MODE: ({"mode"}, set()),
    GAZE: ({"x", "y", "confidence"}, set()),
    GESTURE: ({"gesture"}, set()),
    DWELL: ({"progress", "ready"}, set()),
    SCROLL: ({"rate", "offset", "ready"}, set()),
    TARGET_SEARCHING: ({"label"}, {"snap"}),
    TARGET_FOUND: ({"x", "y", "label"}, {"snap"}),
    TARGET_NONE: ({"label"}, {"snap"}),
    CALIB_TARGET: ({"index", "x", "y", "total"}, set()),
    CALIB_RESULT: ({"error_px", "ok"}, set()),
    LISTENING: ({"level", "speaking"}, set()),
    DICTATION_PREVIEW: ({"words"}, {"cleaned", "original"}),
    STATUS: ({"level", "text"}, set()),
    CONFIRM: ({"id", "prompt", "state", "expires_in_s"}, set()),
}

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
