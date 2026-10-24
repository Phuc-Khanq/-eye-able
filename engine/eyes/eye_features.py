"""Turns eye landmarks into h, v and lid, each divided by eye width.

Route: Khang and Hung (Khang leads)
Publishes: EyeFeatures
Subscribes: FaceFrame
APIs: NumPy; blendshapes eyeBlinkLeft / eyeBlinkRight for blink skipping
"""

# Corner A (image left), corner B (image right), iris center, upper lid, lower lid
RIGHT_EYE = {"a": 33, "b": 133, "iris": 468, "upper": 159, "lower": 145}
LEFT_EYE = {"a": 362, "b": 263, "iris": 473, "upper": 386, "lower": 374}


def compute(face) -> "EyeFeatures":
    """Convert to pixels first, compute h, v, lid per eye, average weighted by eye width.
    Mark the frame unusable when either blink score is above settings.blink_threshold."""
    raise NotImplementedError
