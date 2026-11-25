"""
Module 10: Event and Operational Logging System
Maintains local CSV logging using Pandas for offline benchmarking and evaluation.
Tracks timestamps, predictions, confidence, visual metrics, FPS, and latency.
Strictly based on the project synopsis requirements.
"""

import time
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
from config import settings


class EventLogger:
    """
    Asynchronously buffers and logs system execution metrics to local disk using Pandas.
    """
    def __init__(self, log_file: Optional[Path] = None, flush_interval: int = 30):
        self.log_file = log_file if log_file is not None else settings.DEFAULT_LOG_FILE
        self.flush_interval = flush_interval  # Flush to disk every N events
        self.buffer: List[Dict[str, Any]] = []

        # Ensure directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Initialize CSV header if file does not exist
        if not self.log_file.exists():
            self._write_empty_header()

    def _write_empty_header(self):
        """Creates empty CSV with column structure."""
        columns = [
            "timestamp", "iso_time", "driver_state", "confidence",
            "ear", "mar", "blink_rate", "eye_closure_duration",
            "head_pitch", "head_roll", "warning_score",
            "is_drowsy_event", "is_warning_event", "is_alert_active", "is_tracking_lost",
            "fps", "latency_ms", "experiment_flag"
        ]
        df = pd.DataFrame(columns=columns)
        df.to_csv(self.log_file, index=False)

    def log_frame(self,
                  state: str,
                  confidence: float,
                  features: Dict[str, Any],
                  warning_score: float,
                  is_alert_active: bool,
                  fps: float,
                  latency_ms: float,
                  experiment_flag: str = "live_session"):
        """
        Appends a frame's metrics to the logging buffer.
        """
        now = time.time()
        record = {
            "timestamp": now,
            "iso_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            "driver_state": state,
            "confidence": round(float(confidence), 4),
            "ear": features.get("ear", 0.0),
            "mar": features.get("mar", 0.0),
            "blink_rate": features.get("blink_rate", 0.0),
            "eye_closure_duration": features.get("eye_closure_duration", 0),
            "head_pitch": features.get("head_pitch", 0.0),
            "head_roll": features.get("head_roll", 0.0),
            "warning_score": round(float(warning_score), 2),
            "is_drowsy_event": 1 if state == settings.STATE_DROWSY else 0,
            "is_warning_event": 1 if state == settings.STATE_WARNING else 0,
            "is_alert_active": 1 if is_alert_active else 0,
            "is_tracking_lost": 1 if state == settings.STATE_TRACKING_LOST else 0,
            "fps": round(float(fps), 1),
            "latency_ms": round(float(latency_ms), 2),
            "experiment_flag": experiment_flag
        }

        self.buffer.append(record)

        if len(self.buffer) >= self.flush_interval:
            self.flush()

    def flush(self):
        """Writes buffered events to disk."""
        if not self.buffer:
            return

        try:
            df = pd.DataFrame(self.buffer)
            df.to_csv(self.log_file, mode="a", header=not self.log_file.exists(), index=False)
            self.buffer.clear()
        except Exception:
            pass

    def load_logs(self) -> pd.DataFrame:
        """Loads and returns all logged records as a Pandas DataFrame."""
        self.flush()
        if self.log_file.exists():
            return pd.read_csv(self.log_file)
        return pd.DataFrame()

    def clear_logs(self):
        """Clears local log history."""
        self.buffer.clear()
        if self.log_file.exists():
            self._write_empty_header()

    def __del__(self):
        self.flush()
