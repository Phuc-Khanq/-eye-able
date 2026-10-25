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
    # Left click: a long blink clicks the exact pixel under the cursor (Oct 24 night: snapping moved blink clicks
    # onto the wrong button). The voice "click" snaps onto the nearest button, link or box, for spots the cursor
    # can't quite reach: at once when the cursor is already on one (the hand cursor or UI Automation), else UI-Venus
    # looks near the cursor.
    # An eyebrow raise is NOT a click any more (Oct 24 evening): it answers a waiting question, preview or search
    # ("yes") and leaves Scroll mode. Put "left" here to bring the raise-click back.
    brow_click: str = "off"        # "off": a raise with nothing waiting does nothing; "left": left click; "right": right click
    snap_gesture_clicks: bool = False   # true: a blink click snaps with UI-Venus too, like the voice "click" (the old way)
    snap_trust_cursor: bool = True  # already on a link or button: click there without asking UI-Venus
    snap_max_px: int = 150         # UI-Venus may move a snap click at most this far
    blink_click: bool = True       # a long, deliberate blink (both eyes) is the left click (on since Khang's live
                                   # test, Oct 24; --no-blink turns it off)
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
    card_heard_outside_dictation: bool = False  # the card shows what the mic hears only in Dictation mode
                                                # (true: also in Pointer mode, as before Oct 24 evening)
    card_command_flash: bool = True   # the card's pill shows a command's name ("Volume up") for a moment when it runs
    dwell_meter: bool = True       # the on-screen dwell ring at the cursor (engine/eyes/dwell_meter.py), only with
                                   # dwell clicking on; clicks no longer draw a ring (click_flash below)
    click_flash: bool = True       # a bright dot pops and fades where each click lands (engine/eyes/click_feedback.py)
    click_sound: bool = True       # ...and a short tick (a lower one for a right click)
    click_sound_volume: float = 0.5   # 0 to 1
    sensitivity: float = 1.0
    pointer_mode: str = "head"        # head: the head alone moves the cursor (default again, Oct 24 night: the eye pull
                                      # was too hard to control). head-eyes: the eyes also pull it toward where you look
                                      # while the head turns that way (engine/eyes/pointer.py; needs
                                      # calibration/gaze_model.json, else it is "head"). "head": the head alone
    eye_pull: float = 0.5             # head-eyes: how hard the eyes pull (0 = never, 0.5 = half as hard, 1.5 = harder)
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
    whisper_hotwords: str = ""     # words Whisper should lean toward, e.g. "pause yes no": try them with
                                   # python -m engine.voice.command_bench --hotwords "..." before setting them here
    # Only the user's voice: keep a word only if the camera saw the mouth moving as it was said
    mouth_gate: bool = True
    mouth_gate_mode: str = "utterance"  # "utterance": keep or drop the whole utterance; "word": each word on its own
    mouth_min_moving: float = 0.3      # share of face frames with a moving mouth; lower it for weak movement
    mouth_endpoint: bool = True        # also end a clip when your mouth has been still for pause_seconds (a noisy room:
                                       # the VAD never hears a pause); needs the camera to see your face
    mouth_pad_ms: float = 150.0        # widen each word's time window by this much on both sides
    mouth_status_every_s: float = 10.0  # "Ignored speech while your mouth wasn't moving", at most this often
    # An eyebrow raise while the mouth is moving is talking, not a yes or a right click
    brow_ignore_while_talking: bool = True
    brow_talk_window_ms: float = 600.0  # look this far back from the raise
    brow_talk_min_moving: float = 0.3   # share of face frames with a moving mouth that counts as talking
    scroll_idle_s: float = 10.0         # scroll mode ends by itself after this long with no scrolling
    # Dictation clean-up: a small local model tidies the preview before it's typed (engine/voice/rewrite.py)
    rewrite_dictation: bool = True
    rewrite_model: str = "Qwen/Qwen2.5-3B-Instruct"     # Apache 2.0; about 6 GB, downloaded once (was 1.5B until
                                                        # Oct 24 night: too small to fix words from context)
    rewrite_device: str = "cuda"       # "cpu" works but is slow (seconds per sentence)
    rewrite_wait_s: float = 1.5        # "yes" waits at most this long for the clean-up, then types what's showing
    rewrite_max_s: float = 3.0         # a clean-up that takes longer is cut off and not used
    # Whole-passage refine (engine/voice/rewrite.py): once you stop talking for refine_after_s, the whole passage is
    # rewritten to read well and words Whisper was unsure of are fixed from context. Only with the clean-up on.
    refine_dictation: bool = True
    refine_after_s: float = 3.0        # quiet this long after the last words: start refining
    refine_min_words: int = 8          # shorter passages only get the quick clean-up
    refine_wait_s: float = 6.0         # "yes" while it refines waits at most this long, then types what's showing
    refine_max_s: float = 15.0         # a refine that takes longer is cut off and not used

    # Live preview (engine/voice/audio_in.py + dictation.py): while you are still speaking, the card shows what Whisper
    # hears so far, about once a second, then the finished words replace it. Dictation mode only; never typed.
    live_preview: bool = True
    live_every_s: float = 1.0          # how often the audio so far is transcribed
    live_min_s: float = 0.8            # not before this much speech (a shorter clip is mostly noise)

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
