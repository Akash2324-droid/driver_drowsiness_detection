"""
Module 5.3: Blink Rate and Blink Duration Tracking
Tracks:
- Blink frequency (blinks per minute / rolling window)
- Eye-closure duration (consecutive frames)
- Repeated blinking
- Prolonged eye closure (microsleeps)
Distinguishes normal short blinks from sustained eye closure.
Strictly based on the project synopsis requirements.
"""

import time
from collections import deque
from typing import Dict, Any
from config import settings


class BlinkTracker:
    """
    Monitors eye closure state over time to calculate blink frequency and duration.
    """
    def __init__(self,
                 ear_threshold: float = settings.EAR_DROWSY_THRESHOLD,
                 normal_blink_max_frames: int = settings.NORMAL_BLINK_MAX_FRAMES,
                 history_window_seconds: float = 60.0):
        
        self.ear_threshold = ear_threshold
        self.normal_blink_max_frames = normal_blink_max_frames
        self.history_window_seconds = history_window_seconds

        self.consecutive_closed_frames = 0
        self.total_blinks = 0
        self.blink_timestamps = deque()  # Timestamps of completed blinks
        self.is_currently_closed = False
        self.last_blink_duration_frames = 0

    def update(self, ear: float, timestamp: float = None) -> Dict[str, Any]:
        """
        Updates the tracker with the current frame's EAR measurement.

        Returns:
            Dict containing:
            - consecutive_closed_frames: int
            - is_closed: bool
            - is_prolonged_closure: bool
            - is_normal_blink: bool
            - blink_rate_per_min: float
            - total_blinks: int
            - last_blink_duration: int
        """
        if timestamp is None:
            timestamp = time.time()

        is_normal_blink = False

        if ear < self.ear_threshold:
            self.consecutive_closed_frames += 1
            self.is_currently_closed = True
        else:
            if self.is_currently_closed:
                # Eye just opened: evaluate the completed closure event
                if 1 <= self.consecutive_closed_frames <= self.normal_blink_max_frames:
                    # Valid normal blink event
                    self.total_blinks += 1
                    self.blink_timestamps.append(timestamp)
                    is_normal_blink = True

                self.last_blink_duration_frames = self.consecutive_closed_frames
                self.consecutive_closed_frames = 0
                self.is_currently_closed = False

        # Clean timestamps older than history_window_seconds
        cutoff = timestamp - self.history_window_seconds
        while self.blink_timestamps and self.blink_timestamps[0] < cutoff:
            self.blink_timestamps.popleft()

        # Compute blink rate (normalized to blinks per minute)
        recent_blinks = len(self.blink_timestamps)
        # If tracking for less than a minute, extrapolate or report current count
        blink_rate_per_min = float(recent_blinks) * (60.0 / self.history_window_seconds)

        is_prolonged_closure = (self.consecutive_closed_frames >= settings.PROLONGED_CLOSURE_MIN_FRAMES)

        return {
            "consecutive_closed_frames": self.consecutive_closed_frames,
            "is_closed": self.is_currently_closed,
            "is_prolonged_closure": is_prolonged_closure,
            "is_normal_blink": is_normal_blink,
            "blink_rate_per_min": round(blink_rate_per_min, 1),
            "total_blinks": self.total_blinks,
            "last_blink_duration_frames": self.last_blink_duration_frames
        }

    def reset(self):
        """Resets the blink tracker state."""
        self.consecutive_closed_frames = 0
        self.total_blinks = 0
        self.blink_timestamps.clear()
        self.is_currently_closed = False
        self.last_blink_duration_frames = 0
