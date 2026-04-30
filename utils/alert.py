import threading
import os
import sys
import time

ALARM_PATH      = os.path.join(os.path.dirname(__file__), "..", "assets", "alarm.wav")
REPEAT_INTERVAL = 2.0   


def _beep():
    def _run():
        try:
            if sys.platform == "win32":
                import winsound
                winsound.Beep(1500, 600)  
            else:
                print("\a", end="", flush=True)
        except Exception as e:
            print(f"[Alert] {e}")
    threading.Thread(target=_run, daemon=True).start()


class AlertSystem:
    def __init__(self):
        self._last_played = 0.0   # timestamp of last beep

    def update(self, condition: bool):
        if not condition:
            self._last_played = 0.0
            return

        now = time.time()
        if now - self._last_played >= REPEAT_INTERVAL:
            _beep()
            self._last_played = now

    @property
    def is_active(self) -> bool:
        return (time.time() - self._last_played) < REPEAT_INTERVAL