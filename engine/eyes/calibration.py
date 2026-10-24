"""9-point calibration: median features per dot, then a regression to screen x, y.

Route: Khang and Hung (Khang leads)
Publishes: calib_target, calib_result, the fitted gaze model
Subscribes: EyeFeatures, HeadPose, calibrate messages
APIs: scikit-learn Ridge with PolynomialFeatures; saved as JSON (numbers only, no images)
"""

class GazeModel:
    def fit(self, samples, targets) -> float:
        """Fit on calibration samples; return the average error in pixels."""
        raise NotImplementedError

    def predict(self, eyes, head) -> tuple[float, float]:
        raise NotImplementedError
