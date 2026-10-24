"""Event-bus topic names inside the engine.

messages.py is the contract with the UI (JSON over WebSocket). This file is the
contract between engine modules on the in-process bus. Payload types come from
types.py. Tell the team before renaming a topic.

Route: Khoi and Khoa
"""

# Eyes route -> mode manager
GAZE = "eyes.gaze"                  # GazePoint, every frame
GESTURE = "eyes.gesture"            # Gesture (nod, shake, tilt_left, tilt_right, brow_raise)
DWELL_DONE = "eyes.dwell_done"      # GazePoint where a dwell finished (the cursor rested dwell_ms)
CALIBRATE = "eyes.calibrate"        # {"action": "start" | "cancel"}
MOUTH = "eyes.mouth"                # MouthSample, every frame with a face (engine/eyes/mouth.py reads it)

# Voice route
UTTERANCE = "voice.utterance"       # numpy float32 array, 16 kHz mono (audio_in -> dictation)
SPEECH_SPAN = "voice.speech_span"   # {"start_ms": float, "end_ms": float} (audio_in -> dictation), published
                                    # right before each UTTERANCE on the same thread: time.monotonic() ms of
                                    # the utterance audio's first sample (pre-roll included) and of its end
TRANSCRIPT = "voice.transcript"     # {"text": str, "words": [{"text", "conf"}], "ms": float}
DELETE_VOICE_DATA = "voice.delete_data"  # {} (mode manager -> voice_data, after a confirm)

# Dictation clean-up (mode manager <-> engine/voice/rewrite.py RewriteWorker)
REWRITE_REQUEST = "dictation.rewrite_request"  # {"id": int, "text": str}, each time the preview text changes
REWRITE_RESULT = "dictation.rewrite_result"    # {"id": int, "clean": str, "used": bool, "ms": float}; clean is
                                               # the original text when used is False

# Look-and-say (mode manager <-> screen_target)
TARGET_REQUEST = "target.request"   # TargetRequest
TARGET_RESULT = "target.result"     # TargetResult

# Mode manager -> os_input. Only the mode manager publishes these.
OS_CLICK = "os.click"               # {"x": int, "y": int}
OS_TYPE = "os.type"                 # {"text": str}
OS_KEY = "os.key"                   # {"key": "enter" | "backspace"}
OS_SCROLL = "os.scroll"             # {"direction": "up" | "down"}
OS_UNDO = "os.undo"                 # {} (Ctrl+Z)

# Engine <-> UI. events.serve_websocket sends every TO_UI payload (a messages.make()
# dict) to the UI, and publishes each message from the UI on from_ui(its "type").
TO_UI = "ui.out"


def from_ui(msg_type: str) -> str:
    """Topic for a UI -> engine message, e.g. from_ui(messages.SET_MODE) == "ui.set_mode"."""
    return f"ui.{msg_type}"
