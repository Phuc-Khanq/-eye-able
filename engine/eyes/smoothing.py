"""One Euro filter on the gaze point, and dwell detection.

Route: Khang and Hung (Khang leads)
Publishes: GazePoint (gaze), dwell
Subscribes: raw gaze predictions
APIs: own code (One Euro filter, about 30 lines)
"""

class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.01) -> None:
        raise NotImplementedError

    def __call__(self, t_s: float, value: float) -> float:
        raise NotImplementedError
