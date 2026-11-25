"""
Module: Face Detection
Wraps face isolation and bounding box verification.
Handles tracking loss conditions without falsely incrementing fatigue.
Strictly based on the project synopsis requirements.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from backend.landmark_detection import FaceLandmarkDetector


class FaceDetector:
    """
    Dedicated Face Detection interface for detecting face region and tracking status.
    """
    def __init__(self, detector: Optional[FaceLandmarkDetector] = None):
        self.detector = detector if detector is not None else FaceLandmarkDetector()

    def detect_face(self, rgb_frame: np.ndarray) -> Dict[str, Any]:
        """
        Detects if face is present and extracts bounding box and landmarks.
        Returns:
            Dict containing face_detected (bool), bounding_box, landmarks_2d, landmarks_3d.
        """
        detection_result = self.detector.detect(rgb_frame)
        return detection_result
