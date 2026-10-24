"""Settings with their defaults. The console's settings panel changes these at runtime.

Each laptop can override any setting in config.local.json at the repo root
(git ignores it). Example for a laptop without an NVIDIA GPU:
    {"whisper_device": "cpu", "whisper_model": "small",
     "whisper_compute_type": "int8", "use_ui_venus": false}
"""
import json
from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass
class Settings:
    # Camera
    # Defaults are the demo PC's webcam; run `python -m tests.camera_probe` to check a camera.
    camera_index: int = 0
    camera_backend: str = "dshow"  # "dshow", "msmf" or "any"
    camera_fourcc: str = "MJPG"    # "" lets the driver choose
    frame_width: int = 1280
    frame_height: int = 720
    fps: int = 30
    preview_mirror: bool = True    # mirror only what people see; MediaPipe gets the unmirrored frame
    face_model_path: str = "models/face_landmarker.task"
    blink_threshold: float = 0.5

    # Pointing and clicking
    # Left click: an eyebrow raise (or "click") snaps onto the nearest button, link or box: at once when the
    # cursor is already on one (the hand cursor or UI Automation), else UI-Venus looks near the cursor.
    brow_click: str = "left"       # "left": a raise with nothing waiting is the snap left click; "right": the old right click
    snap_trust_cursor: bool = True  # already on a link or button: click there without asking UI-Venus
    snap_max_px: int = 150         # UI-Venus may move a snap click at most this far
    blink_click: bool = True       # a long, deliberate blink (0.4 to 1.2 s, both eyes) = the same click as an eyebrow
                                   # raise (on since Khang's live test, Oct 24; --no-blink turns it off)
    blink_rise: float = 0.20       # the blink click: how far over your open level the eyes must close; and how
    blink_closed_abs: float = 0.55  # shut they must get (tests.blink_record + tests.blink_replay suggest yours)
    blink_min_ms: int = 400        # the blink click: shut at least this long (a normal blink is shorter; raise it to
                                   # 500 if tired, slow blinks click by accident)
    blink_max_ms: int = 1600       # ...and not longer than this (longer is resting the eyes). Was 1200: deliberate
                                   # closes of 1.2 to 1.6 s were ignored in Khang's live test
    blink_left_gain: float = 1.0   # multiply each eye's blink score before the lower of the two counts. A weak eye
    blink_right_gain: float = 1.0  # (Khang's right eye scores 0.43 to 0.62 when shut) gets a gain above 1;
                                   # tests.blink_replay prints the suggestion. A wink still never clicks.
    mouth_click: bool = False      # open the mouth and close it again (0.3 to 1.5 s) = right click; off: it fired
                                   # while talking in the live test (Oct 24). Right click is "right click" by voice
    dwell_ms: int = 1000           # with --dwell: rest the cursor this long = left click (off by default)
    dwell_clickable_only: bool = True   # Windows: dwell only on links, buttons, tabs... (engine/eyes/clickable.py);
                                        # false for canvas apps and games, which show neither
    dwell_uia: bool = True         # also ask Windows UI Automation (native buttons, text boxes; pip install uiautomation)
    dictation_card: bool = True    # the box at the bottom of the screen: what was heard and what will be typed
                                   # (engine/eyes/dictation_card.py); false once Nhi's Electron bubble shows it
    card_opacity: float = 0.82     # 0.3 to 1: lower = more see-through
    card_position: str = "top"     # "top" or "bottom" center of the screen
    card_hide_from_capture: bool = True   # keep the card out of screenshots so UI-Venus never reads it; this also
                                          # hides it from OBS, so set false while recording the demo video
    dwell_meter: bool = True       # the on-screen ring at the cursor (engine/eyes/dwell_meter.py): the snap click's
                                   # spinner and flash, and the dwell fill; no Electron UI needed
    sensitivity: float = 1.0
    gpu_log_every_s: float = 10.0   # --gpu-log: print the GPU memory this often (engine/gpu_log.py)
    gpu_warn_mb: int = 14000        # ...and warn above this (the 5080 has 16 GB; the demo needs room to spare)

    # UI-Venus
    use_ui_venus: bool = True
    ui_venus_model: str = "inclusionAI/UI-Venus-1.5-2B"
    ui_venus_device: str = "cuda"
    crop_width: int = 1280
    crop_height: int = 720

    # Voice
    sample_rate: int = 16000
    pause_seconds: float = 1.5  # user setting, 1 to 4
    whisper_model: str = "large-v3-turbo"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "float16"  # use "int8" on CPU
    # Only the user's voice: keep a word only if the camera saw the mouth moving as it was said
    mouth_gate: bool = True
    mouth_gate_mode: str = "utterance"  # "utterance": keep or drop the whole utterance; "word": each word on its own
    mouth_min_moving: float = 0.3      # share of face frames with a moving mouth; lower it for weak movement
    mouth_pad_ms: float = 150.0        # widen each word's time window by this much on both sides
    mouth_status_every_s: float = 10.0  # "Ignored speech while your mouth wasn't moving", at most this often
    # An eyebrow raise while the mouth is moving is talking, not a yes or a right click
    brow_ignore_while_talking: bool = True
    brow_talk_window_ms: float = 600.0  # look this far back from the raise
    brow_talk_min_moving: float = 0.3   # share of face frames with a moving mouth that counts as talking
    scroll_idle_s: float = 10.0         # scroll mode ends by itself after this long with no scrolling
    # Dictation clean-up: a small local model tidies the preview before it's typed (engine/voice/rewrite.py)
    rewrite_dictation: bool = True
    rewrite_model: str = "Qwen/Qwen2.5-1.5B-Instruct"   # Apache 2.0; about 3.1 GB, downloaded once
    rewrite_device: str = "cuda"       # "cpu" works but is slow (seconds per sentence)
    rewrite_wait_s: float = 1.5        # "yes" waits at most this long for the clean-up, then types what's showing
    rewrite_max_s: float = 3.0         # a clean-up that takes longer is cut off and not used

    # Opening apps and sites by voice (engine/voice/launch.py; engine/eyes/os_output.py starts them)
    # "open Chrome" starts an app only if its name is here; any other name stays look-and-say.
    launch_apps: dict = field(default_factory=lambda: {
        "chrome": "chrome.exe",            # os.startfile finds these through Windows App Paths
        "edge": "msedge.exe",
        "notepad": "notepad.exe",
        "file explorer": "explorer.exe",
        "settings": "ms-settings:",        # Windows Settings
        "mail": "mailto:",                 # the default mail app
    })
    launch_aliases: dict = field(default_factory=lambda: {   # other ways to say a launch_apps name
        "google chrome": "chrome",
        "microsoft edge": "edge",
        "explorer": "file explorer",
        "files": "file explorer",
        "my files": "file explorer",
        "email": "mail",
        "outlook": "mail",
    })
    launch_sites: dict = field(default_factory=lambda: {   # "go to YouTube": one-word sites that need no ".com";
        "youtube": "youtube.com",                          # any other word with no dot is not a site
        "google": "google.com",                            # ("go to sleep" opens nothing)
        "gmail": "mail.google.com",
        "wikipedia": "wikipedia.org",
        "facebook": "facebook.com",
        "netflix": "netflix.com",
    })
    launch_search_url: str = "https://www.google.com/search?q={q}"   # "search for <words>"; {q} = the words

    # Engine <-> UI
    ws_host: str = "127.0.0.1"
    ws_port: int = 8765


def load() -> Settings:
    s = Settings()
    local = Path(__file__).resolve().parent.parent / "config.local.json"
    if local.exists():
        known = {f.name for f in fields(Settings)}
        for key, value in json.loads(local.read_text(encoding="utf-8")).items():
            if key in known:
                setattr(s, key, value)
            else:
                print(f"config.local.json: unknown setting '{key}' ignored")
    return s


settings = load()
