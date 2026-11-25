"""
Module 3 & 4: Face Landmark Detection using MediaPipe Face Landmarker.
Extracts 2D and 3D facial landmarks for eyes, mouth, contours, and head orientation.
Strictly based on the project synopsis requirements.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from config import settings


class FaceLandmarkDetector:
    """
    Detects driver facial landmarks using MediaPipe Face Landmarker.
    Provides pixel coordinates (2D) and normalized coordinates (3D).
    """
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path if model_path is not None else settings.FACE_LANDMARKER_MODEL_PATH
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe Face Landmarker model asset not found at: {self.model_path}. "
                f"Please ensure face_landmarker.task is downloaded in models/."
            )

        base_options = python.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
            num_faces=settings.MAX_NUM_FACES,
            min_face_detection_confidence=settings.MIN_DETECTION_CONFIDENCE,
            min_face_presence_confidence=settings.MIN_TRACKING_CONFIDENCE,
            min_tracking_confidence=settings.MIN_TRACKING_CONFIDENCE
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def detect(self, rgb_frame: np.ndarray) -> Dict[str, Any]:
        """
        Detects facial landmarks on an RGB image.

        Returns:
            Dict containing:
            - face_detected: bool
            - landmarks_2d: np.ndarray of shape (N, 2) in integer pixel coordinates
            - landmarks_3d: np.ndarray of shape (N, 3) in float coordinates (x, y, z)
            - bounding_box: Tuple (x, y, w, h)
            - facial_transformation_matrix: Optional[np.ndarray] (4x4 matrix if available)
        """
        if rgb_frame is None or rgb_frame.size == 0:
            return {
                "face_detected": False,
                "landmarks_2d": None,
                "landmarks_3d": None,
                "bounding_box": None,
                "facial_transformation_matrix": None
            }

        height, width = rgb_frame.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.detector.detect(mp_image)

        if not detection_result.face_landmarks or len(detection_result.face_landmarks) == 0:
            return {
                "face_detected": False,
                "landmarks_2d": None,
                "landmarks_3d": None,
                "bounding_box": None,
                "facial_transformation_matrix": None
            }

        # Primary face landmarks
        first_face = detection_result.face_landmarks[0]
        landmarks_2d = []
        landmarks_3d = []

        for lm in first_face:
            px = int(lm.x * width)
            py = int(lm.y * height)
            landmarks_2d.append([px, py])
            landmarks_3d.append([lm.x, lm.y, lm.z])

        landmarks_2d_arr = np.array(landmarks_2d, dtype=np.int32)
        landmarks_3d_arr = np.array(landmarks_3d, dtype=np.float32)

        # Bounding box calculation
        x_min = int(np.min(landmarks_2d_arr[:, 0]))
        y_min = int(np.min(landmarks_2d_arr[:, 1]))
        x_max = int(np.max(landmarks_2d_arr[:, 0]))
        y_max = int(np.max(landmarks_2d_arr[:, 1]))

        x_min = max(0, x_min)
        y_min = max(0, y_min)
        box_w = min(width - x_min, max(1, x_max - x_min))
        box_h = min(height - y_min, max(1, y_max - y_min))

        matrix = None
        if detection_result.facial_transformation_matrixes and len(detection_result.facial_transformation_matrixes) > 0:
            matrix = np.array(detection_result.facial_transformation_matrixes[0], dtype=np.float32)

        return {
            "face_detected": True,
            "landmarks_2d": landmarks_2d_arr,
            "landmarks_3d": landmarks_3d_arr,
            "bounding_box": (x_min, y_min, box_w, box_h),
            "facial_transformation_matrix": matrix
        }

    def close(self):
        if hasattr(self, "detector") and self.detector is not None:
            self.detector.close()
            self.detector = None

    def __del__(self):
        self.close()
