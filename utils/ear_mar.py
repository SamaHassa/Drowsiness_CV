import numpy as np
from collections import deque


EAR_THRESHOLD   = 0.25   
MAR_THRESHOLD   = 0.65   
SMOOTH_WINDOW   = 5      


def euclidean(p1, p2) -> float:
    """Euclidean distance between two (x, y) points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))


def eye_aspect_ratio(eye_points: list) -> float:
    p1, p2, p3, p4, p5, p6 = eye_points

    vertical_1 = euclidean(p2, p6)
    vertical_2 = euclidean(p3, p5)
    horizontal = euclidean(p1, p4)

    if horizontal == 0:
        return 0.0

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return round(ear, 4)


def mouth_aspect_ratio(mouth_points: list) -> float:
    p1, p2, p3, p4, p5, p6, p7, p8 = mouth_points

    vertical_1 = euclidean(p3, p4)
    vertical_2 = euclidean(p7, p8)
    horizontal = euclidean(p1, p2)

    if horizontal == 0:
        return 0.0

    mar = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return round(mar, 4)


class RatioSmoother:


    def __init__(self, window: int = SMOOTH_WINDOW):
        self._buf = deque(maxlen=window)

    def update(self, value: float) -> float:
        self._buf.append(value)
        return sum(self._buf) / len(self._buf)

    def reset(self):
        self._buf.clear()

    @property
    def is_full(self) -> bool:
        return len(self._buf) == self._buf.maxlen


def compute_ratios(left_eye, right_eye, mouth):
    if left_eye is None or right_eye is None:
        return None, None
    ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
    mar = mouth_aspect_ratio(mouth) if mouth else 0.0
    return round(ear, 4), round(mar, 4)