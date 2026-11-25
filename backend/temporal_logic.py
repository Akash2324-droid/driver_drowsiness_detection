"""
Module 8: Temporal Decision Engine
Maintains a sliding temporal window of recent frames and feature metrics.
Smooths high-frequency noise and implements warning score accumulation.
Distinguishes short normal blinks and isolated yawns from sustained drowsiness.
Properly handles tracking loss without falsely elevating fatigue scores.
Recognized states: ALERT, WARNING, DROWSY, TRACKING_LOST.
Strictly based on the project synopsis requirements.
"""

from collections import deque
from typing import Dict, Any, Optional
from config import settings


class TemporalDecisionEngine:
    """
    Temporal persistence filter and state machine for driver alertness monitoring.
    """
    def __init__(self,
                 window_size: int = settings.SLIDING_WINDOW_SIZE,
                 warning_threshold: float = settings.WARNING_SCORE_THRESHOLD,
                 drowsy_threshold: float = settings.DROWSY_SCORE_THRESHOLD,
                 max_score: float = settings.MAX_WARNING_SCORE):
        
        self.window_size = window_size
        self.warning_threshold = warning_threshold
        self.drowsy_threshold = drowsy_threshold
        self.max_score = max_score

        self.warning_score = 0.0
        self.current_state = settings.STATE_ALERT
        self.recent_predictions = deque(maxlen=window_size)
        self.consecutive_alert_frames = 0
        self.consecutive_drowsy_frames = 0

    def evaluate(self,
                 features: Dict[str, Any],
                 ml_prediction: Dict[str, Any],
                 tracking_lost: bool) -> Dict[str, Any]:
        """
        Evaluates the current frame within temporal context.

        Args:
            features: Dictionary containing ear, mar, blink_rate, eye_closure_duration, is_nodding_down.
            ml_prediction: Dictionary containing is_drowsy, confidence, proba_drowsy.
            tracking_lost: Whether the driver's face could not be detected.

        Returns:
            Dict containing:
            - state: "ALERT", "WARNING", "DROWSY", or "TRACKING_LOST"
            - warning_score: float
            - is_alert_active: bool (True only if state == DROWSY)
            - trigger_reason: str
        """
        # Scenario 1: Tracking Lost (Camera obscured or face missing)
        # As per DOCX: Display tracking warning; hold fatigue score; do NOT increment fatigue score.
        if tracking_lost:
            self.current_state = settings.STATE_TRACKING_LOST
            return {
                "state": settings.STATE_TRACKING_LOST,
                "warning_score": round(self.warning_score, 2),
                "is_alert_active": False,
                "trigger_reason": "Driver face missing or tracking lost."
            }

        ear = features.get("ear", 0.3)
        mar = features.get("mar", 0.3)
        eye_closure_duration = features.get("eye_closure_duration", 0)
        is_nodding = features.get("is_nodding_down", False)
        ml_drowsy = ml_prediction.get("is_drowsy", False)

        self.recent_predictions.append(1 if ml_drowsy else 0)

        # Evaluate Fatigue Indicators
        # A) Prolonged Eye Closure (not a short normal blink)
        is_prolonged_closure = (eye_closure_duration >= settings.PROLONGED_CLOSURE_MIN_FRAMES)
        # B) Yawning
        is_yawning = (mar >= settings.MAR_YAWN_THRESHOLD)
        # C) Compounded evidence (Yawning + Eye closure)
        is_compounded = (is_prolonged_closure and is_yawning)

        trigger_reason = "Normal alertness."

        # Temporal Score Adjustment
        if is_compounded:
            # Compounded fatigue evidence: rapidly increment warning score
            self.warning_score += settings.SCORE_INCREMENT_COMPOUND
            self.consecutive_alert_frames = 0
            self.consecutive_drowsy_frames += 1
            trigger_reason = "Compounded fatigue: sustained eye closure and yawning."

        elif is_prolonged_closure:
            # Strong indicator of drowsiness
            self.warning_score += settings.SCORE_INCREMENT_EYE_CLOSED
            self.consecutive_alert_frames = 0
            self.consecutive_drowsy_frames += 1
            trigger_reason = f"Prolonged eye closure ({eye_closure_duration} frames)."

        elif ml_drowsy and (is_yawning or is_nodding):
            # ML flagged fatigue with supportive behavioral movement
            self.warning_score += settings.SCORE_INCREMENT_YAWN
            self.consecutive_alert_frames = 0
            self.consecutive_drowsy_frames += 1
            trigger_reason = "Fatigue pattern detected (ML prediction + yawning/head nodding)."

        elif ml_drowsy and sum(self.recent_predictions) > (self.window_size * 0.6):
            # Persistent drowsy predictions across sliding window
            self.warning_score += 1.0
            self.consecutive_alert_frames = 0
            self.consecutive_drowsy_frames += 1
            trigger_reason = "Persistent frame-level fatigue predictions."

        elif is_yawning and not is_prolonged_closure:
            # Isolated yawn alone: slight score increment but does not directly push to alert
            self.warning_score = min(self.warning_score + 0.3, self.warning_threshold - 1.0)
            self.consecutive_alert_frames = 0
            trigger_reason = "Isolated yawning observed (monitoring)."

        else:
            # Normal alertness (normal open eyes or brief normal blink)
            self.consecutive_alert_frames += 1
            self.consecutive_drowsy_frames = 0
            # Gradual score recovery
            self.warning_score = max(0.0, self.warning_score - settings.SCORE_DECREMENT_ALERT)
            trigger_reason = "Driver alert and focused."

        # Clamp score to limits [0.0, max_score]
        self.warning_score = min(self.max_score, max(0.0, self.warning_score))

        # Determine Driver State
        if self.warning_score >= self.drowsy_threshold:
            self.current_state = settings.STATE_DROWSY
            is_alert_active = True
        elif self.warning_score >= self.warning_threshold:
            self.current_state = settings.STATE_WARNING
            is_alert_active = False
        else:
            self.current_state = settings.STATE_ALERT
            is_alert_active = False

        return {
            "state": self.current_state,
            "warning_score": round(self.warning_score, 2),
            "is_alert_active": is_alert_active,
            "trigger_reason": trigger_reason
        }

    def reset(self):
        """Manually resets the decision engine."""
        self.warning_score = 0.0
        self.current_state = settings.STATE_ALERT
        self.recent_predictions.clear()
        self.consecutive_alert_frames = 0
        self.consecutive_drowsy_frames = 0
