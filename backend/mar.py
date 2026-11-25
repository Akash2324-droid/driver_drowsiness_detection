"""
Module 5.2: Mouth Aspect Ratio (MAR) Calculation
Calculates the geometric Mouth Aspect Ratio to quantify mouth opening and identify yawning patterns.
Formula:
    MAR = (||top_outer - bottom_outer|| + ||top_inner - bottom_inner||) / (2 * ||corner_left - corner_right||)
Strictly based on the project synopsis requirements.
"""

import numpy as np
from typing import Optional
from config import settings


def compute_mar(landmarks_2d: np.ndarray) -> float:
    """
    Computes the Mouth Aspect Ratio (MAR) from facial landmarks.

    Landmarks used:
    - Left corner: 61
    - Right corner: 291
    - Top outer: 13, Bottom outer: 14
    - Top inner: 82, Bottom inner: 312

    Returns:
        mar: float value representing mouth openness.
    """
    if landmarks_2d is None or len(landmarks_2d) < 468:
        return 0.0

    p_left = landmarks_2d[settings.MOUTH_CORNER_LEFT]
    p_right = landmarks_2d[settings.MOUTH_CORNER_RIGHT]
    p_top_outer = landmarks_2d[settings.MOUTH_TOP_OUTER]
    p_bottom_outer = landmarks_2d[settings.MOUTH_BOTTOM_OUTER]
    p_top_inner = landmarks_2d[settings.MOUTH_TOP_INNER]
    p_bottom_inner = landmarks_2d[settings.MOUTH_BOTTOM_INNER]

    # Vertical mouth opening distances
    v1 = np.linalg.norm(p_top_outer - p_bottom_outer)
    v2 = np.linalg.norm(p_top_inner - p_bottom_inner)

    # Horizontal mouth width
    h = np.linalg.norm(p_left - p_right)

    if h < 1e-6:
        return 0.0

    mar = (v1 + v2) / (2.0 * h)
    return float(mar)
