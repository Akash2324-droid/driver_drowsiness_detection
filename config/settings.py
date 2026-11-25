"""
Configuration settings for the AI-Powered Driver Drowsiness Detection System.
Strictly based on requirements and parameters specified in the project synopsis.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
BACKEND_DIR = BASE_DIR / "backend"
FRONTEND_DIR = BASE_DIR / "frontend"
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
CACHE_DIR = BASE_DIR / ".mpl_cache"

# Ensure runtime directories exist
DATASET_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Set Matplotlib cache dir to writable project folder
os.environ["MPLCONFIGDIR"] = str(CACHE_DIR)

# MediaPipe Face Landmarker Task Model Path
FACE_LANDMARKER_MODEL_PATH = MODELS_DIR / "face_landmarker.task"

# Camera and Video Settings
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30
CAMERA_BUFFER_SIZE = 1  # Minimize buffering delay for real-time capture

# Preprocessing Settings
STANDARDIZED_WIDTH = 640
STANDARDIZED_HEIGHT = 480
ENABLE_BRIGHTNESS_NORMALIZATION = False  # Optional as stated in synopsis

# MediaPipe Face Landmarker Settings
MAX_NUM_FACES = 1
REFINE_LANDMARKS = True
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Standard Facial Landmark Indices (MediaPipe 468/478 mesh)
# Left eye keypoints for EAR: [p1, p2, p3, p4, p5, p6]
# p1: outer corner, p4: inner corner, p2/p6: upper/lower outer, p3/p5: upper/lower inner
LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
# Right eye keypoints for EAR:
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]

# Mouth keypoints for MAR:
# Upper/lower lips vertical pairs and left/right mouth corners
MOUTH_CORNER_LEFT = 61
MOUTH_CORNER_RIGHT = 291
MOUTH_TOP_OUTER = 13
MOUTH_BOTTOM_OUTER = 14
MOUTH_TOP_INNER = 82
MOUTH_BOTTOM_INNER = 312
MOUTH_LANDMARKS = [61, 291, 13, 14, 82, 312]

# 3D Model Points for Head Pose Estimation (PnP with landmark coordinates)
# Nose tip, Chin, Left eye corner, Right eye corner, Left mouth corner, Right mouth corner
HEAD_POSE_LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]

# Drowsiness Feature Thresholds (Used for baseline geometric evaluation)
EAR_DROWSY_THRESHOLD = 0.22      # Eye closure threshold
MAR_YAWN_THRESHOLD = 0.55        # Mouth opening threshold for yawning
NORMAL_BLINK_MAX_FRAMES = 8      # Blinks typically last 100-300ms (3-8 frames at 30 fps)
PROLONGED_CLOSURE_MIN_FRAMES = 15 # Sustained eye closure threshold (~0.5s)
DOWNWARD_PITCH_THRESHOLD = -15.0 # Degrees head pitch downward

# Temporal Decision Engine Settings
SLIDING_WINDOW_SIZE = 30         # Rolling window (~1.0s at 30 fps)
WARNING_SCORE_THRESHOLD = 15     # Threshold for Warning state
DROWSY_SCORE_THRESHOLD = 30      # Threshold for Drowsy state
SCORE_INCREMENT_EYE_CLOSED = 2.0 # Weight for prolonged eye closure
SCORE_INCREMENT_YAWN = 1.0       # Weight for yawning
SCORE_INCREMENT_COMPOUND = 3.5   # Weight for yawning + eye closure
SCORE_DECREMENT_ALERT = 1.5      # Recovery rate when alert
MAX_WARNING_SCORE = 60.0

# Driver States
STATE_ALERT = "ALERT"
STATE_WARNING = "WARNING"
STATE_DROWSY = "DROWSY"
STATE_TRACKING_LOST = "TRACKING_LOST"

# Audio Alert Settings
AUDIO_ALERT_ENABLED = True
AUDIO_FREQUENCY_HZ = 2500
AUDIO_DURATION_MS = 400

# Machine Learning Settings
PRIMARY_MODEL_TYPE = "RandomForest"
ALTERNATIVE_MODEL_TYPE = "SVM"
DEFAULT_MODEL_PATH = MODELS_DIR / "drowsiness_rf_model.pkl"
SCALER_PATH = MODELS_DIR / "feature_scaler.pkl"

FEATURE_NAMES = [
    "ear",
    "mar",
    "blink_rate",
    "eye_closure_duration",
    "head_pitch",
    "head_roll",
    "ear_rolling_mean",
    "ear_rolling_std",
    "mar_rolling_mean"
]

# Random Forest Hyperparameters
RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 4,
    "min_samples_leaf": 2,
    "random_state": 42
}

# SVM Hyperparameters
SVM_PARAMS = {
    "C": 1.0,
    "kernel": "rbf",
    "gamma": "scale",
    "probability": True,
    "random_state": 42
}

# Logging Settings
DEFAULT_LOG_FILE = LOGS_DIR / "system_events.csv"
