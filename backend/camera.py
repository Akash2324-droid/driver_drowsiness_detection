"""
Module 1: Video Acquisition Module
Handles continuous video stream capture from the webcam, camera availability checks,
and resource management without unnecessary frame buffering.
Strictly based on the project synopsis requirements.
"""

import cv2
import time
from typing import Tuple, Optional
from config import settings


class CameraManager:
    """
    Manages webcam capture, health checks, and frame acquisition.
    """
    def __init__(self, camera_index: int = settings.CAMERA_INDEX,
                 width: int = settings.FRAME_WIDTH,
                 height: int = settings.FRAME_HEIGHT):
        self.camera_index = camera_index
        self.requested_width = width
        self.requested_height = height
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.actual_width = 0
        self.actual_height = 0

    def start(self) -> Tuple[bool, str]:
        """
        Initializes and opens the video capture device.
        Returns: (success: bool, message: str)
        """
        if self.is_running and self.cap is not None and self.cap.isOpened():
            return True, "Camera is already running."

        try:
            # On Windows, cv2.CAP_DSHOW provides fast, reliable camera access
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                # Fallback to default backend if CAP_DSHOW fails
                self.cap = cv2.VideoCapture(self.camera_index)

            if not self.cap.isOpened():
                self.is_running = False
                return False, f"Failed to access webcam at index {self.camera_index}. Check camera connection and privacy permissions."

            # Minimize frame buffering to eliminate processing lag
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, settings.CAMERA_BUFFER_SIZE)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)
            self.cap.set(cv2.CAP_PROP_FPS, settings.TARGET_FPS)

            self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.is_running = True

            return True, f"Camera initialized successfully ({self.actual_width}x{self.actual_height} @ index {self.camera_index})."

        except Exception as e:
            self.is_running = False
            return False, f"Exception occurred while starting camera: {str(e)}"

    def get_frame(self) -> Tuple[bool, Optional[cv2.Mat]]:
        """
        Captures a single frame from the camera.
        Returns: (success: bool, frame: Optional[np.ndarray])
        """
        if not self.is_running or self.cap is None or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        if not ret or frame is None or frame.size == 0:
            return False, None

        return True, frame

    def stop(self) -> None:
        """
        Stops camera capture and safely releases hardware resources.
        """
        self.is_running = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def __del__(self):
        self.stop()
