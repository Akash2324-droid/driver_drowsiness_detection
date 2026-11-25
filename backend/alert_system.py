"""
Module 9: Alert System Module
Manages on-screen visual alerts and host audio warnings.
Ensures audible alerts are non-blocking and do not spawn duplicate audio processes.
Stops alerts upon recovery of sustained alertness or manual reset.
Strictly based on the project synopsis requirements.
"""

import sys
import time
import threading
from typing import Optional
from config import settings


class AlertManager:
    """
    Controls audible and visual alarm states.
    Employs a dedicated background thread for host sound generation to avoid UI lag.
    """
    def __init__(self,
                 enabled: bool = settings.AUDIO_ALERT_ENABLED,
                 frequency: int = settings.AUDIO_FREQUENCY_HZ,
                 duration_ms: int = settings.AUDIO_DURATION_MS):
        
        self.enabled = enabled
        self.frequency = frequency
        self.duration_ms = duration_ms

        self._alert_active = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def set_alert_state(self, is_drowsy: bool):
        """
        Activates or deactivates the alarm based on temporal drowsiness confirmation.
        """
        with self._lock:
            if is_drowsy:
                if not self._alert_active:
                    self._alert_active = True
                    self._start_audio_loop()
            else:
                if self._alert_active:
                    self._alert_active = False
                    self._stop_audio_loop()

    def _start_audio_loop(self):
        """Starts the dedicated audio worker thread if not already running."""
        if not self.enabled:
            return

        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._audio_worker, daemon=True)
            self._thread.start()

    def _stop_audio_loop(self):
        """Signals the audio thread to stop."""
        self._stop_event.set()

    def _audio_worker(self):
        """
        Background worker that plays an audible alert until stopped or state returns to alert.
        Uses native winsound on Windows without spawning external processes.
        """
        while not self._stop_event.is_set() and self._alert_active:
            try:
                if sys.platform == "win32":
                    import winsound
                    # Play tone (frequency, duration in ms)
                    winsound.Beep(self.frequency, self.duration_ms)
                else:
                    # ANSI bell fallback on non-Windows systems
                    sys.stdout.write("\a")
                    sys.stdout.flush()
                    time.sleep(self.duration_ms / 1000.0)

                # Brief pause between alarm beeps
                time.sleep(0.1)

            except Exception:
                # If audio device is unavailable or muted, proceed gracefully
                time.sleep(0.3)

    def is_active(self) -> bool:
        """Returns True if the alert is currently active."""
        return self._alert_active

    def reset(self):
        """Immediately stops the alarm and resets state."""
        with self._lock:
            self._alert_active = False
            self._stop_event.set()

    def stop(self):
        """Releases alert resources."""
        self.reset()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.5)
            self._thread = None

    def __del__(self):
        self.stop()
