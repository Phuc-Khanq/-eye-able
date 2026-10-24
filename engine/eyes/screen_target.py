"""UI-Venus 1.5: crop the screen around the gaze point, find the named element, return screen pixels.

Route: Khang and Hung (Khang leads)
Publishes: TargetResult, target_debug (console open only)
Subscribes: TargetRequest
APIs: mss (screen crop), transformers>=4.57, torch (CUDA 12.8), inclusionAI/UI-Venus-1.5-2B. Runs in its own worker thread. Prompt and coordinate format: confirm in tests/target_bench.py.
"""

class ScreenTarget:
    def __init__(self, model_id: str) -> None:
        raise NotImplementedError

    def find(self, request) -> "TargetResult":
        """[-1,-1] from the model means not found: return found=False, never guess."""
        raise NotImplementedError
