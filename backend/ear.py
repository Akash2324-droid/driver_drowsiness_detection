"""
Module 5.1: Eye Aspect Ratio (EAR) Calculation
Calculates the geometric Eye Aspect Ratio to measure eye openness and monitor eye closure.
Formula:
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
Strictly based on the project synopsis requirements.
"""

import numpy as np
from typing import Tuple, Optional
from config import settings


def calculate_single_eye_ear(eye_landmarks: np.ndarray) -> float:
    """
    Computes EAR for a single eye given 6 landmark points [p1, p2, p3, p4, p5, p6].
    p1: outer corner, p4: inner corner
    p2, p6: upper and lower outer vertical points
    p3, p5: upper and lower inner vertical points
    """
    if eye_landmarks is None or len(eye_landmarks) < 6:
        return 0.0

    # Euclidean distance vertical pairs
    # ||p2 - p6||
    v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    # ||p3 - p5||
    v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    # Horizontal eye width: ||p1 - p4||
    h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])

    if h < 1e-6:
        return 0.0

    ear = (v1 + v2) / (2.0 * h)
    return float(ear)


def compute_ear(landmarks_2d: np.ndarray) -> Tuple[float, float, float]:
    """
    Computes left EAR, right EAR, and average EAR from 2D facial landmarks.
    
    Returns:
        (avg_ear, left_ear, right_ear)
    """
    if landmarks_2d is None or len(landmarks_2d) < 468:
        return 0.0, 0.0, 0.0

    left_pts = landmarks_2d[settings.LEFT_EYE_LANDMARKS]
    right_pts = landmarks_2d[settings.RIGHT_EYE_LANDMARKS]

    left_ear = calculate_single_eye_ear(left_pts)
    right_ear = calculate_single_eye_ear(right_pts)

    avg_ear = (left_ear + right_ear) / 2.0
    return float(avg_ear), float(left_ear), float(right_ear)
