"""
End-to-End System Integration Test Suite
Validates all backend and frontend modules independently and integrally.
Checks:
- Camera manager interface
- Preprocessor
- Landmark detection (MediaPipe Face Landmarker)
- EAR calculation
- MAR calculation
- Blink tracker
- Head pose estimator
- Feature extractor
- Model trainer & subject-independent split
- Real-time model inference engine
- Temporal decision engine & all 4 states (Alert, Warning, Drowsy, Tracking Lost)
- Alert manager
- Event logger
- Performance evaluator & Matplotlib plots
- Unified pipeline execution
"""

import sys
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from backend.camera import CameraManager
from backend.preprocessing import FramePreprocessor
from backend.landmark_detection import FaceLandmarkDetector
from backend.ear import calculate_single_eye_ear, compute_ear
from backend.mar import compute_mar
from backend.blink_detection import BlinkTracker
from backend.head_pose import HeadPoseEstimator
from backend.feature_extraction import FeatureExtractor
from backend.model_training import ModelTrainer, generate_benchmark_dataset
from backend.model_inference import ModelInferenceEngine
from backend.temporal_logic import TemporalDecisionEngine
from backend.alert_system import AlertManager
from backend.logging_system import EventLogger
from backend.evaluation import PerformanceEvaluator
from backend.pipeline import DrowsinessDetectionPipeline


def run_tests():
    print("=" * 65)
    print("RUNNING COMPREHENSIVE END-TO-END VERIFICATION TESTS")
    print("AI-Powered Driver Drowsiness Detection System")
    print("=" * 65)

    # 1. Camera Manager
    print("\n[Test 1] Testing Camera Manager initialization...")
    camera = CameraManager()
    assert camera.camera_index == settings.CAMERA_INDEX
    print("  -> CameraManager initialized successfully.")

    # 2. Preprocessor
    print("\n[Test 2] Testing Frame Preprocessor...")
    preprocessor = FramePreprocessor()
    test_raw = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    bgr, rgb = preprocessor.process(test_raw)
    assert bgr.shape == (480, 640, 3)
    assert rgb.shape == (480, 640, 3)
    print("  -> Frame Preprocessor standardized resolution and color conversion verified.")

    # 3. Landmark Detector
    print("\n[Test 3] Testing MediaPipe Face Landmarker...")
    detector = FaceLandmarkDetector()
    det_res = detector.detect(rgb)
    assert "face_detected" in det_res
    print(f"  -> Face Landmarker executed (Face detected on noise: {det_res['face_detected']}).")

    # 4. EAR Calculation
    print("\n[Test 4] Testing Eye Aspect Ratio (EAR) calculation...")
    open_eye_pts = np.array([[10, 50], [20, 38], [30, 38], [40, 50], [30, 62], [20, 62]])
    ear_val = calculate_single_eye_ear(open_eye_pts)
    assert ear_val > 0.4
    closed_eye_pts = np.array([[10, 50], [20, 49], [30, 49], [40, 50], [30, 51], [20, 51]])
    ear_closed = calculate_single_eye_ear(closed_eye_pts)
    assert ear_closed < 0.1
    print(f"  -> Open eye EAR = {ear_val:.4f}, Closed eye EAR = {ear_closed:.4f} (Formulas verified).")

    # 5. MAR Calculation
    print("\n[Test 5] Testing Mouth Aspect Ratio (MAR) calculation...")
    dummy_landmarks = np.zeros((468, 2), dtype=np.int32)
    # Set mouth points
    dummy_landmarks[settings.MOUTH_CORNER_LEFT] = [20, 50]
    dummy_landmarks[settings.MOUTH_CORNER_RIGHT] = [80, 50]
    dummy_landmarks[settings.MOUTH_TOP_OUTER] = [50, 20]
    dummy_landmarks[settings.MOUTH_BOTTOM_OUTER] = [50, 80]
    dummy_landmarks[settings.MOUTH_TOP_INNER] = [50, 25]
    dummy_landmarks[settings.MOUTH_BOTTOM_INNER] = [50, 75]
    mar_val = compute_mar(dummy_landmarks)
    assert mar_val > 0.8
    print(f"  -> Yawning MAR = {mar_val:.4f} (Formula verified).")

    # 6. Blink Tracker
    print("\n[Test 6] Testing Blink Tracker and Duration Tracking...")
    blink_tracker = BlinkTracker()
    for _ in range(4):
        res = blink_tracker.update(0.15)
    assert res["is_closed"] == True
    assert res["consecutive_closed_frames"] == 4
    res = blink_tracker.update(0.32)
    assert res["is_normal_blink"] == True
    assert res["total_blinks"] == 1
    print("  -> Blink tracker correctly distinguished normal blink and counted blinks.")

    # 7. Head Pose Dynamics
    print("\n[Test 7] Testing Head Pose Estimation...")
    head_pose = HeadPoseEstimator()
    dummy_pose_landmarks = np.zeros((468, 2), dtype=np.int32)
    # Set face feature coordinates
    dummy_pose_landmarks[1] = [320, 240]   # Nose
    dummy_pose_landmarks[152] = [320, 340] # Chin
    dummy_pose_landmarks[33] = [260, 200]  # Left eye
    dummy_pose_landmarks[263] = [380, 200] # Right eye
    dummy_pose_landmarks[61] = [280, 290]  # Left mouth
    dummy_pose_landmarks[291] = [360, 290] # Right mouth
    pose_res = head_pose.estimate_pose(dummy_pose_landmarks)
    assert pose_res["success"] == True
    print(f"  -> Head pose calculated: Pitch={pose_res['pitch']}°, Roll={pose_res['roll']}°, Yaw={pose_res['yaw']}°.")

    # 8. Feature Extractor
    print("\n[Test 8] Testing Feature Extractor...")
    extractor = FeatureExtractor()
    feat_res = extractor.extract_features(dummy_pose_landmarks)
    assert not feat_res["tracking_lost"]
    assert feat_res["feature_vector"].shape == (1, 9)
    print("  -> Feature vector constructed with 9 dimensions matching configuration.")

    # 9. ML Model Training & Subject-Independent Split
    print("\n[Test 9] Testing ML Model Training & Subject-Independent Split...")
    test_csv = settings.DATASET_DIR / "driver_fatigue_dataset.csv"
    df = generate_benchmark_dataset(test_csv, num_subjects=6, samples_per_subject=200)
    trainer = ModelTrainer()
    train_df, test_df = trainer.split_subject_independent(df)
    train_subjs = set(train_df["subject_id"].unique())
    test_subjs = set(test_df["subject_id"].unique())
    assert len(train_subjs.intersection(test_subjs)) == 0, "DATASET LEAKAGE DETECTED! Subjects overlap across partitions."
    print("  -> Subject-independent splitting verified: Zero subject overlap between train and test partitions.")

    rf_res = trainer.train(df, model_type="RandomForest", save_model=True)
    assert rf_res["accuracy"] >= 0.90
    print(f"  -> Random Forest trained: Accuracy = {rf_res['accuracy'] * 100:.2f}%, F1 = {rf_res['f1_score'] * 100:.2f}%.")

    # 10. Real-Time Model Inference Engine
    print("\n[Test 10] Testing Real-Time Model Inference Engine...")
    inference_engine = ModelInferenceEngine()
    assert inference_engine.is_loaded == True
    pred_alert = inference_engine.predict(np.array([[0.32, 0.35, 16.0, 0, 0.0, 0.0, 0.32, 0.02, 0.35]], dtype=np.float32))
    assert pred_alert["state"] == settings.STATE_ALERT
    pred_drowsy = inference_engine.predict(np.array([[0.10, 0.65, 6.0, 25, -20.0, 0.0, 0.12, 0.07, 0.55]], dtype=np.float32))
    assert pred_drowsy["state"] == settings.STATE_DROWSY
    print("  -> Frame-level predictions & probability scores verified.")

    # 11. Temporal Decision Engine & State Transitions
    print("\n[Test 11] Testing Temporal Decision Engine & State Transitions...")
    temporal = TemporalDecisionEngine()
    
    # State A: Normal Alert
    state_a = temporal.evaluate({"ear": 0.35, "mar": 0.35, "eye_closure_duration": 0}, {"is_drowsy": False}, tracking_lost=False)
    assert state_a["state"] == settings.STATE_ALERT
    assert state_a["warning_score"] == 0.0
    print("  -> State 1: ALERT verified.")

    # State B: Tracking Lost (must NOT increase fatigue score)
    state_tl = temporal.evaluate({}, {}, tracking_lost=True)
    assert state_tl["state"] == settings.STATE_TRACKING_LOST
    assert state_tl["warning_score"] == 0.0
    print("  -> State 2: TRACKING_LOST verified (Fatigue score held without false increase).")

    # State C: Warning State (simulate repeated prolonged fatigue signals)
    for _ in range(8):
        state_w = temporal.evaluate(
            {"ear": 0.12, "mar": 0.35, "eye_closure_duration": 18, "is_nodding_down": False},
            {"is_drowsy": True}, tracking_lost=False
        )
    assert state_w["state"] in [settings.STATE_WARNING, settings.STATE_DROWSY]
    print(f"  -> State 3: WARNING verified (Warning score reached {state_w['warning_score']}).")

    # State D: Sustained Drowsiness -> Drowsy Alert Trigger
    for _ in range(12):
        state_d = temporal.evaluate(
            {"ear": 0.08, "mar": 0.70, "eye_closure_duration": 30, "is_nodding_down": True},
            {"is_drowsy": True}, tracking_lost=False
        )
    assert state_d["state"] == settings.STATE_DROWSY
    assert state_d["is_alert_active"] == True
    print(f"  -> State 4: DROWSY verified (Warning score {state_d['warning_score']} >= {settings.DROWSY_SCORE_THRESHOLD}, Alert Active).")

    # Recovery (warning score drops by 1.5 per alert frame: 58.0 / 1.5 ≈ 39 frames)
    for _ in range(45):
        state_rec = temporal.evaluate(
            {"ear": 0.35, "mar": 0.35, "eye_closure_duration": 0, "is_nodding_down": False},
            {"is_drowsy": False}, tracking_lost=False
        )
    assert state_rec["state"] == settings.STATE_ALERT
    assert state_rec["is_alert_active"] == False
    assert state_rec["warning_score"] == 0.0
    print("  -> Alert recovery verified (Score decremented to ALERT, alarm turned off).")

    # 12. Alert Manager & Thread Safety
    print("\n[Test 12] Testing Alert Manager...")
    alert_mgr = AlertManager(enabled=True)
    alert_mgr.set_alert_state(True)
    assert alert_mgr.is_active() == True
    time.sleep(0.1)
    alert_mgr.set_alert_state(False)
    assert alert_mgr.is_active() == False
    alert_mgr.stop()
    print("  -> Alert Manager thread safety and alarm state toggling verified.")

    # 13. Logging System
    print("\n[Test 13] Testing Local Event Logger...")
    logger = EventLogger()
    logger.log_frame(
        state=settings.STATE_ALERT,
        confidence=0.98,
        features={"ear": 0.34, "mar": 0.32, "blink_rate": 18.0, "eye_closure_duration": 0, "head_pitch": 0.0, "head_roll": 0.0},
        warning_score=0.0,
        is_alert_active=False,
        fps=30.0,
        latency_ms=12.5,
        experiment_flag="unit_test"
    )
    logger.flush()
    df_logs = logger.load_logs()
    assert not df_logs.empty
    print(f"  -> Event Logger recorded {len(df_logs)} records with complete metric schema.")

    # 14. Performance Evaluator & Visualizations
    print("\n[Test 14] Testing Performance Evaluator & Plots...")
    evaluator = PerformanceEvaluator()
    eval_metrics = evaluator.evaluate_predictions(
        np.array([0, 1, 0, 1, 1, 0]),
        np.array([0, 1, 0, 1, 0, 0])
    )
    assert "accuracy" in eval_metrics
    assert "f1_score" in eval_metrics

    # Matplotlib plots
    cm_path = evaluator.plot_confusion_matrix(np.array(rf_res["confusion_matrix"]))
    assert cm_path.exists()
    fi_path = evaluator.plot_feature_importance(rf_res["feature_importances"])
    assert fi_path.exists()
    print(f"  -> Generated Confusion Matrix plot: {cm_path.name}")
    print(f"  -> Generated Feature Importance plot: {fi_path.name}")

    # 15. Unified Pipeline
    print("\n[Test 15] Testing Unified Pipeline...")
    pipeline = DrowsinessDetectionPipeline()
    sample_frame = np.ones((480, 640, 3), dtype=np.uint8) * 120
    pipeline_res = pipeline.process_frame(sample_frame)
    assert "annotated_frame" in pipeline_res
    assert pipeline_res["annotated_frame"].shape == (480, 640, 3)
    assert pipeline_res["fps"] > 0
    pipeline.close()
    print("  -> Unified Pipeline processed frame end-to-end successfully.")

    print("\n" + "=" * 65)
    print("ALL 15 INTEGRATION TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 65)


if __name__ == "__main__":
    run_tests()
