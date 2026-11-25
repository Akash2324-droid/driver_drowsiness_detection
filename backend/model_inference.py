"""
Module 7: Real-Time Machine Learning Inference Engine
Loads the pre-trained classifier and scaler once during initialization.
Produces frame-level state predictions (Alert vs. Drowsy) and confidence probabilities.
Strictly based on the project synopsis requirements.
"""

import os
import joblib
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from config import settings


class ModelInferenceEngine:
    """
    Executes frame-level inference with pre-trained models.
    """
    def __init__(self, model_path: Optional[Path] = None, scaler_path: Optional[Path] = None):
        self.model_path = model_path if model_path is not None else settings.DEFAULT_MODEL_PATH
        self.scaler_path = scaler_path if scaler_path is not None else settings.SCALER_PATH
        self.model = None
        self.scaler = None
        self.is_loaded = False
        self.load_error_message = ""

        self._load_artifacts()

    def _load_artifacts(self):
        """Loads model and scaler files into memory."""
        try:
            if not self.model_path.exists():
                self.is_loaded = False
                self.load_error_message = (
                    f"Model file not found at: {self.model_path}. "
                    f"Please run the training pipeline first to produce the model."
                )
                return

            if not self.scaler_path.exists():
                self.is_loaded = False
                self.load_error_message = (
                    f"Feature scaler not found at: {self.scaler_path}. "
                    f"Please run the training pipeline to generate the scaler."
                )
                return

            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.is_loaded = True
            self.load_error_message = ""

        except Exception as e:
            self.is_loaded = False
            self.load_error_message = f"Failed to load model artifacts: {str(e)}"

    def predict(self, feature_vector: np.ndarray) -> Dict[str, Any]:
        """
        Runs inference on a single frame feature vector.
        Vector must be shape (1, 9) ordered according to settings.FEATURE_NAMES.

        Returns:
            Dict containing:
            - is_drowsy: bool
            - state: "ALERT" or "DROWSY"
            - confidence: float (0.0 to 1.0)
            - proba_drowsy: float (probability of Drowsy class)
            - proba_alert: float (probability of Alert class)
            - model_operational: bool
        """
        if not self.is_loaded or self.model is None or self.scaler is None:
            return {
                "is_drowsy": False,
                "state": settings.STATE_ALERT,
                "confidence": 0.0,
                "proba_drowsy": 0.0,
                "proba_alert": 0.0,
                "model_operational": False,
                "error": self.load_error_message
            }

        if feature_vector is None or feature_vector.size == 0:
            return {
                "is_drowsy": False,
                "state": settings.STATE_ALERT,
                "confidence": 0.0,
                "proba_drowsy": 0.0,
                "proba_alert": 0.0,
                "model_operational": True,
                "error": "Empty feature vector provided."
            }

        try:
            # Scale features
            scaled_vector = self.scaler.transform(feature_vector)

            # Predict class
            prediction = int(self.model.predict(scaled_vector)[0])
            is_drowsy = (prediction == 1)

            # Predict probabilities
            if hasattr(self.model, "predict_proba"):
                probas = self.model.predict_proba(scaled_vector)[0]
                proba_alert = float(probas[0])
                proba_drowsy = float(probas[1])
                confidence = float(max(proba_alert, proba_drowsy))
            else:
                proba_drowsy = 1.0 if is_drowsy else 0.0
                proba_alert = 1.0 - proba_drowsy
                confidence = 1.0

            state_label = settings.STATE_DROWSY if is_drowsy else settings.STATE_ALERT

            return {
                "is_drowsy": is_drowsy,
                "state": state_label,
                "confidence": round(confidence, 4),
                "proba_drowsy": round(proba_drowsy, 4),
                "proba_alert": round(proba_alert, 4),
                "model_operational": True,
                "error": ""
            }

        except Exception as e:
            return {
                "is_drowsy": False,
                "state": settings.STATE_ALERT,
                "confidence": 0.0,
                "proba_drowsy": 0.0,
                "proba_alert": 0.0,
                "model_operational": False,
                "error": f"Inference execution failed: {str(e)}"
            }
