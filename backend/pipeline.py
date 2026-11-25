"""
Backend Pipeline Orchestrator
Integrates camera acquisition, preprocessing, landmark detection, feature extraction,
ML inference, temporal logic, alerts, and event logging into a unified pipeline.
Keeps core computational logic strictly in the backend, exposing a clean interface to the frontend.
Strictly based on the project synopsis requirements.
"""

import cv2
import time
import numpy as np
from typing import Dict, Any, Optional, Tuple
from config import settings
from backend.preprocessing import FramePreprocessor
from backend.landmark_detection import FaceLandmarkDetector
from backend.feature_extraction import FeatureExtractor
from backend.model_inference import ModelInferenceEngine
from backend.temporal_logic import TemporalDecisionEngine
from backend.alert_system import AlertManager
from backend.logging_system import EventLogger


class DrowsinessDetectionPipeline:
    """
    Unified backend execution pipeline for frame-level driver state monitoring.
    """
    def __init__(self):
        self.preprocessor = FramePreprocessor()
        self.detector = FaceLandmarkDetector()
        self.feature_extractor = FeatureExtractor()
        self.inference_engine = ModelInferenceEngine()
        self.temporal_engine = TemporalDecisionEngine()
        self.alert_manager = AlertManager()
        self.logger = EventLogger()

        self.last_frame_time = time.time()
        self.frame_count = 0
        self.current_fps = 0.0

    def process_frame(self, raw_frame: np.ndarray) -> Dict[str, Any]:
        """
        Executes end-to-end analysis on an incoming video frame.

        Sequence:
        Camera Frame -> Preprocessing -> Landmark Detection -> Feature Extraction
        -> ML State Classification -> Temporal Persistence Filtering
        -> Alert System -> Event Logging -> UI Overlay Drawing.

        Returns:
            Dict containing:
            - annotated_frame: np.ndarray (BGR)
            - state: "ALERT", "WARNING", "DROWSY", or "TRACKING_LOST"
            - warning_score: float
            - is_alert_active: bool
            - ml_prediction: Dict
            - features: Dict
            - fps: float
            - latency_ms: float
            - trigger_reason: str
        """
        start_time = time.time()

        # 1. Preprocessing
        normalized_bgr, rgb_frame = self.preprocessor.process(raw_frame)

        # 2. Landmark Detection
        detection_res = self.detector.detect(rgb_frame)
        face_detected = detection_res["face_detected"]
        landmarks_2d = detection_res["landmarks_2d"]
        bounding_box = detection_res["bounding_box"]

        # 3. Feature Engineering
        feature_res = self.feature_extractor.extract_features(landmarks_2d, timestamp=start_time)
        features = feature_res["features"]
        feature_vector = feature_res["feature_vector"]
        tracking_lost = not face_detected

        # 4. ML State Classification
        if not tracking_lost and feature_vector is not None:
            ml_pred = self.inference_engine.predict(feature_vector)
        else:
            ml_pred = {
                "is_drowsy": False,
                "state": settings.STATE_TRACKING_LOST if tracking_lost else settings.STATE_ALERT,
                "confidence": 0.0,
                "proba_drowsy": 0.0,
                "proba_alert": 0.0,
                "model_operational": self.inference_engine.is_loaded
            }

        # 5. Temporal Decision Engine
        temporal_res = self.temporal_engine.evaluate(
            features=features,
            ml_prediction=ml_pred,
            tracking_lost=tracking_lost
        )
        state = temporal_res["state"]
        warning_score = temporal_res["warning_score"]
        is_alert_active = temporal_res["is_alert_active"]
        trigger_reason = temporal_res["trigger_reason"]

        # 6. Alert Triggering
        self.alert_manager.set_alert_state(is_alert_active)

        # 7. Operational Metrics (FPS & Latency)
        now = time.time()
        latency_ms = (now - start_time) * 1000.0
        time_diff = now - self.last_frame_time
        if time_diff > 0:
            self.current_fps = 0.9 * self.current_fps + 0.1 * (1.0 / time_diff) if self.current_fps > 0 else (1.0 / time_diff)
        self.last_frame_time = now
        self.frame_count += 1

        # 8. Local Event Logging
        self.logger.log_frame(
            state=state,
            confidence=ml_pred.get("confidence", 0.0),
            features=features,
            warning_score=warning_score,
            is_alert_active=is_alert_active,
            fps=self.current_fps,
            latency_ms=latency_ms
        )

        # 9. Visual Annotation
        annotated = self._draw_visual_overlays(
            normalized_bgr,
            landmarks_2d=landmarks_2d,
            bounding_box=bounding_box,
            state=state,
            warning_score=warning_score,
            is_alert_active=is_alert_active,
            features=features,
            ml_pred=ml_pred,
            fps=self.current_fps,
            latency_ms=latency_ms
        )

        return {
            "annotated_frame": annotated,
            "state": state,
            "warning_score": warning_score,
            "is_alert_active": is_alert_active,
            "ml_prediction": ml_pred,
            "features": features,
            "fps": round(self.current_fps, 1),
            "latency_ms": round(latency_ms, 1),
            "trigger_reason": trigger_reason,
            "tracking_lost": tracking_lost,
            "face_detected": face_detected
        }

    def _draw_visual_overlays(self, frame: np.ndarray,
                              landmarks_2d: Optional[np.ndarray],
                              bounding_box: Optional[Tuple[int, int, int, int]],
                              state: str,
                              warning_score: float,
                              is_alert_active: bool,
                              features: Dict[str, Any],
                              ml_pred: Dict[str, Any],
                              fps: float,
                              latency_ms: float) -> np.ndarray:
        """
        Draws professional academic monitoring overlays on the video frame.
        """
        canvas = frame.copy()
        h, w = canvas.shape[:2]

        # Colors (BGR)
        color_alert = (0, 200, 0)       # Green
        color_warning = (0, 165, 255)   # Amber / Orange
        color_drowsy = (0, 0, 255)      # Red
        color_tracking = (200, 100, 0)  # Blue/Grey

        if state == settings.STATE_ALERT:
            state_color = color_alert
        elif state == settings.STATE_WARNING:
            state_color = color_warning
        elif state == settings.STATE_DROWSY:
            state_color = color_drowsy
        else:
            state_color = color_tracking

        # 1. Face Bounding Box & Key Landmark Meshes
        if bounding_box is not None and landmarks_2d is not None:
            bx, by, bw, bh = bounding_box
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), state_color, 2)

            # Draw Eyes Points
            for idx in settings.LEFT_EYE_LANDMARKS + settings.RIGHT_EYE_LANDMARKS:
                pt = landmarks_2d[idx]
                cv2.circle(canvas, tuple(pt), 2, (0, 255, 255), -1)

            # Draw Mouth Points
            for idx in settings.MOUTH_LANDMARKS:
                pt = landmarks_2d[idx]
                cv2.circle(canvas, tuple(pt), 2, (255, 0, 255), -1)

        # 2. Prominent Top Status Banner
        cv2.rectangle(canvas, (0, 0), (w, 55), (20, 20, 20), -1)
        cv2.line(canvas, (0, 55), (w, 55), state_color, 3)

        # State text
        status_text = f"STATE: {state}"
        if is_alert_active:
            status_text += " [AUDIO ALARM ACTIVE]"
        cv2.putText(canvas, status_text, (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, state_color, 2)

        # FPS & Latency
        perf_text = f"FPS: {fps:.1f} | Latency: {latency_ms:.0f}ms"
        cv2.putText(canvas, perf_text, (w - 240, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

        # 3. Bottom HUD for Metrics
        cv2.rectangle(canvas, (0, h - 50), (w, h), (20, 20, 20), -1)
        cv2.line(canvas, (0, h - 50), (w, h - 50), (80, 80, 80), 1)

        ear_val = features.get("ear", 0.0)
        mar_val = features.get("mar", 0.0)
        pitch_val = features.get("head_pitch", 0.0)
        blinks_val = features.get("blink_rate", 0.0)

        metrics_text = f"EAR: {ear_val:.3f} | MAR: {mar_val:.3f} | Score: {warning_score:.1f}/{settings.DROWSY_SCORE_THRESHOLD} | Blinks: {blinks_val:.0f}/min | Pitch: {pitch_val:.1f} deg"
        cv2.putText(canvas, metrics_text, (15, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 1)

        # 4. If DROWSY: Draw high-contrast red warning border
        if state == settings.STATE_DROWSY:
            cv2.rectangle(canvas, (0, 0), (w, h), (0, 0, 255), 8)
            cv2.putText(canvas, "! DROWSINESS DETECTED - WAKE UP !", (w // 2 - 250, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        elif state == settings.STATE_TRACKING_LOST:
            cv2.putText(canvas, "TRACKING LOST - PLEASE FACE CAMERA", (w // 2 - 240, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        return canvas

    def reset_session(self):
        """Resets all trackers, temporal scores, and alert alarms."""
        self.feature_extractor.reset()
        self.temporal_engine.reset()
        self.alert_manager.reset()

    def close(self):
        """Releases all resources."""
        self.alert_manager.stop()
        self.detector.close()
        self.logger.flush()

    def __del__(self):
        self.close()
