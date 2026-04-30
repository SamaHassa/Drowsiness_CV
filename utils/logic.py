import time
from enum import Enum

EAR_THRESHOLD  = 0.20   
MAR_THRESHOLD  = 0.60   
CLOSED_FRAMES  = 3     
YAWN_FRAMES    = 3     


class AlertLevel(Enum):
    NORMAL     = "NORMAL"
    YAWNING    = "YAWNING"
    DROWSY     = "DROWSY"
    HIGH_ALERT = "HIGH ALERT"


class DrowsinessLogic:

    def __init__(self):
        self._closed_counter = 0
        self._yawn_counter   = 0
        self._fps_clock      = []

    def update(self, ear=None, mar=None, eye_prob=None, yawn_prob=None):
        #EAR
        eye_closed = False
        if ear is not None:
            if ear < EAR_THRESHOLD:
                self._closed_counter += 1
            else:
                self._closed_counter = 0      # immediate reset
            eye_closed = self._closed_counter >= CLOSED_FRAMES

        #  MAR  
        yawning = False
        if mar is not None:
            if mar > MAR_THRESHOLD:
                self._yawn_counter += 1
            else:
                self._yawn_counter = 0        # immediate reset
            yawning = self._yawn_counter >= YAWN_FRAMES

        #  DeciSion 
        if eye_closed and yawning:
            level = AlertLevel.HIGH_ALERT
        elif eye_closed:
            level = AlertLevel.DROWSY
        elif yawning:
            level = AlertLevel.YAWNING
        else:
            level = AlertLevel.NORMAL

        signals = {
            "ear":            ear,
            "mar":            mar,
            "closed_counter": self._closed_counter,
            "yawn_counter":   self._yawn_counter,
            "eye_closed":     eye_closed,
            "yawning":        yawning,
            "eye_prob":       eye_prob,
            "yawn_prob":      yawn_prob,
            "cnn_eye":        False,
            "cnn_yawn":       False,
        }

        return level, signals

    def reset(self):
        self._closed_counter = 0
        self._yawn_counter   = 0

    def tick(self):
        now = time.time()
        self._fps_clock.append(now)
        self._fps_clock = [t for t in self._fps_clock if now - t < 1.0]
        return len(self._fps_clock)