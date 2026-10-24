"""Starts the engine: threads for camera and vision, audio, Whisper and UI-Venus,
plus the asyncio loop for the WebSocket server. Run with: python -m engine.main
"""
from engine.config import settings


def main() -> None:
    # Planned start order:
    # 1. core.events: event bus + WebSocket server on settings.ws_host:ws_port
    # 2. core.mode_manager: subscribes to gaze, gestures and commands
    # 3. eyes: camera -> face_mesh -> eye_features / head_pose -> calibration -> smoothing -> os_input
    #    plus head_gestures and the screen_target worker (UI-Venus)
    # 4. voice: audio_in -> dictation -> commands
    print(f"Diseyeble engine skeleton. WebSocket will run on ws://{settings.ws_host}:{settings.ws_port}")
    print("Modules aren't built yet; see the build plan.")


if __name__ == "__main__":
    main()
