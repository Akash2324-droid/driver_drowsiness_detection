"""
Module 5.4: Head Pose Dynamics Estimation
Tracks facial landmarks to estimate 3D head orientation (pitch, roll, yaw).
Identifies downward nodding and tilt patterns as supplementary fatigue indicators.
Uses cv2.solvePnP with standard 3D anthropometric facial model keypoints.
Strictly based on the project synopsis requirements.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
from config import settings


class HeadPoseEstimator:
    """
    Estimates 3D head orientation angles (Pitch, Yaw, Roll) using Perspective-n-Point (PnP).
    """
    def __init__(self, frame_width: int = settings.STANDARDIZED_WIDTH,
                 frame_height: int = settings.STANDARDIZED_HEIGHT):
        self.frame_width = frame_width
        self.frame_height = frame_height

        # Standard 3D anthropometric facial points (in mm, centered roughly at nose bridge)
        # 1: Nose tip, 152: Chin, 33: Left eye outer corner, 263: Right eye outer corner,
        # 61: Left mouth corner, 291: Right mouth corner
        self.model_points = np.array([
            [0.0, 0.0, 0.0],          # Nose tip (index 1)
            [0.0, -330.0, -65.0],     # Chin (index 152)
            [-225.0, 170.0, -135.0],  # Left eye outer corner (index 33)
            [225.0, 170.0, -135.0],   # Right eye outer corner (index 263)
            [-150.0, -150.0, -125.0], # Left mouth corner (index 61)
            [150.0, -150.0, -125.0]   # Right mouth corner (index 291)
        ], dtype=np.float64)

        # Camera intrinsic matrix approximation
        focal_length = self.frame_width
        center = (self.frame_width / 2.0, self.frame_height / 2.0)
        self.camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)  # Assume no lens distortion

    def estimate_pose(self, landmarks_2d: np.ndarray) -> Dict[str, Any]:
        """
        Estimates Pitch, Yaw, and Roll angles from 2D landmarks.

        Returns:
            Dict containing:
            - pitch: float (degrees, negative is looking down)
            - yaw: float (degrees, left/right turning)
            - roll: float (degrees, head tilt)
            - is_nodding_down: bool
            - success: bool
        """
        if landmarks_2d is None or len(landmarks_2d) < 468:
            return {
                "pitch": 0.0,
                "yaw": 0.0,
                "roll": 0.0,
                "is_nodding_down": False,
                "success": False
            }

        image_points = np.array([
            landmarks_2d[1],    # Nose tip
            landmarks_2d[152],  # Chin
            landmarks_2d[33],   # Left eye outer corner
            landmarks_2d[263],  # Right eye outer corner
            landmarks_2d[61],   # Left mouth corner
            landmarks_2d[291]   # Right mouth corner
        ], dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            self.model_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return {
                "pitch": 0.0,
                "yaw": 0.0,
                "roll": 0.0,
                "is_nodding_down": False,
                "success": False
            }

        rmat, _ = cv2.Rodrigues(rvec)

        # Decompose rotation matrix to Euler angles using RQDecomp3x3
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        pitch = float(angles[0])
        yaw = float(angles[1])
        roll = float(angles[2])

        is_nodding_down = (pitch < settings.DOWNWARD_PITCH_THRESHOLD)

        return {
            "pitch": round(pitch, 2),
            "yaw": round(yaw, 2),
            "roll": round(roll, 2),
            "is_nodding_down": is_nodding_down,
            "success": True
        }
