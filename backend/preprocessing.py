"""
Module 2: Frame Pre-processing Module
Resizes frames to a standardized resolution, applies color conversions (BGR -> RGB),
and performs optional brightness normalization.
Strictly based on the project synopsis requirements.
"""

import cv2
import numpy as np
from typing import Tuple
from config import settings


class FramePreprocessor:
    """
    Standardizes incoming video frames for detection and landmark extraction.
    """
    def __init__(self, target_width: int = settings.STANDARDIZED_WIDTH,
                 target_height: int = settings.STANDARDIZED_HEIGHT,
                 enable_brightness_norm: bool = settings.ENABLE_BRIGHTNESS_NORMALIZATION):
        self.target_width = target_width
        self.target_height = target_height
        self.enable_brightness_norm = enable_brightness_norm

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocesses a raw video frame:
        1. Resizes to standardized resolution.
        2. Applies optional contrast/brightness normalization (CLAHE in LAB color space).
        3. Converts BGR to RGB for MediaPipe processing.

        Returns:
            (processed_bgr, rgb_frame)
        """
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty or None.")

        # Standardize resolution
        h, w = frame.shape[:2]
        if w != self.target_width or h != self.target_height:
            resized_bgr = cv2.resize(frame, (self.target_width, self.target_height), interpolation=cv2.INTER_LINEAR)
        else:
            resized_bgr = frame.copy()

        # Optional brightness normalization (CLAHE on L-channel)
        if self.enable_brightness_norm:
            lab = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l_channel)
            merged_lab = cv2.merge((cl, a_channel, b_channel))
            normalized_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
        else:
            normalized_bgr = resized_bgr

        # MediaPipe expects RGB format
        rgb_frame = cv2.cvtColor(normalized_bgr, cv2.COLOR_BGR2RGB)

        return normalized_bgr, rgb_frame
