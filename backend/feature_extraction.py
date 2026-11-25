"""
Module 5: Feature Engineering Module
Aggregates EAR, MAR, Blink dynamics, and Head Pose into a standardized feature vector.
Maintains a rolling window of recent frames to calculate statistical temporal metrics
(EAR rolling mean, EAR rolling std, MAR rolling mean).
Strictly based on the project synopsis requirements.
"""

import numpy as np
from collections import deque
from typing import Dict, Any, List, Optional
from config import settings
from backend.ear import compute_ear
from backend.mar import compute_mar
from backend.blink_detection import BlinkTracker
from backend.head_pose import HeadPoseEstimator


class FeatureExtractor:
    """
    Extracts geometric and statistical features from facial landmarks.
    Produces frame-level feature vectors for the ML classifier.
    """
    def __init__(self, window_size: int = settings.SLIDING_WINDOW_SIZE):
        self.window_size = window_size
        self.blink_tracker = BlinkTracker()
        self.head_pose_estimator = HeadPoseEstimator()

        self.ear_history = deque(maxlen=window_size)
        self.mar_history = deque(maxlen=window_size)

    def extract_features(self, landmarks_2d: Optional[np.ndarray], timestamp: float = None) -> Dict[str, Any]:
        """
        Extracts all metrics from 2D landmarks.

        Returns a dictionary of:
            - features: Dict of individual metrics
            - feature_vector: np.ndarray shape (1, 9) ordered according to settings.FEATURE_NAMES
            - tracking_lost: bool
        """
        if landmarks_2d is None or len(landmarks_2d) < 468:
            return {
                "features": {
                    "ear": 0.0,
                    "mar": 0.0,
                    "blink_rate": 0.0,
                    "eye_closure_duration": 0,
                    "head_pitch": 0.0,
                    "head_roll": 0.0,
                    "ear_rolling_mean": 0.0,
                    "ear_rolling_std": 0.0,
                    "mar_rolling_mean": 0.0
                },
                "feature_vector": None,
                "tracking_lost": True
            }

        # 1. EAR
        avg_ear, left_ear, right_ear = compute_ear(landmarks_2d)

        # 2. MAR
        mar = compute_mar(landmarks_2d)

        # 3. Blink dynamics
        blink_info = self.blink_tracker.update(avg_ear, timestamp=timestamp)
        blink_rate = blink_info["blink_rate_per_min"]
        closure_duration = blink_info["consecutive_closed_frames"]

        # 4. Head pose
        pose_info = self.head_pose_estimator.estimate_pose(landmarks_2d)
        pitch = pose_info["pitch"]
        roll = pose_info["roll"]

        # 5. Temporal rolling statistics
        self.ear_history.append(avg_ear)
        self.mar_history.append(mar)

        ear_mean = float(np.mean(self.ear_history)) if self.ear_history else avg_ear
        ear_std = float(np.std(self.ear_history)) if len(self.ear_history) > 1 else 0.0
        mar_mean = float(np.mean(self.mar_history)) if self.mar_history else mar

        features = {
            "ear": round(avg_ear, 4),
            "left_ear": round(left_ear, 4),
            "right_ear": round(right_ear, 4),
            "mar": round(mar, 4),
            "blink_rate": round(blink_rate, 2),
            "eye_closure_duration": int(closure_duration),
            "head_pitch": round(pitch, 2),
            "head_roll": round(roll, 2),
            "head_yaw": round(pose_info.get("yaw", 0.0), 2),
            "is_nodding_down": pose_info.get("is_nodding_down", False),
            "is_prolonged_closure": blink_info.get("is_prolonged_closure", False),
            "ear_rolling_mean": round(ear_mean, 4),
            "ear_rolling_std": round(ear_std, 4),
            "mar_rolling_mean": round(mar_mean, 4)
        }

        # Vector ordered according to settings.FEATURE_NAMES
        vector = np.array([[
            features["ear"],
            features["mar"],
            features["blink_rate"],
            features["eye_closure_duration"],
            features["head_pitch"],
            features["head_roll"],
            features["ear_rolling_mean"],
            features["ear_rolling_std"],
            features["mar_rolling_mean"]
        ]], dtype=np.float32)

        return {
            "features": features,
            "feature_vector": vector,
            "tracking_lost": False
        }

    def reset(self):
        """Resets the history and trackers."""
        self.ear_history.clear()
        self.mar_history.clear()
        self.blink_tracker.reset()
