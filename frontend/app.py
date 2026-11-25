"""
Frontend Application: Streamlit Dashboard
Displays live webcam feed, real-time driver alertness metrics, warnings,
performance benchmarks, and evaluation reports.
Strictly separated from ML/CV logic (all computation handled in backend modules).
Strictly based on the project synopsis requirements.
"""

import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import pandas as pd
import numpy as np
import streamlit as st
from config import settings
from backend.camera import CameraManager
from backend.pipeline import DrowsinessDetectionPipeline
from backend.model_training import ModelTrainer, generate_benchmark_dataset
from backend.evaluation import PerformanceEvaluator
from frontend.ui_components import render_header, render_state_badge, render_metric_cards

# Configure Streamlit page
st.set_page_config(
    page_title="Driver Drowsiness Detection System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

render_header()

# Initialize Session State
if "pipeline" not in st.session_state:
    st.session_state.pipeline = DrowsinessDetectionPipeline()
if "is_monitoring" not in st.session_state:
    st.session_state.is_monitoring = False
if "evaluator" not in st.session_state:
    st.session_state.evaluator = PerformanceEvaluator()
if "latest_eval_results" not in st.session_state:
    st.session_state.latest_eval_results = None

# Sidebar Controls
st.sidebar.title("🎛️ System Controls")

st.sidebar.subheader("Camera Session")
col_start, col_stop = st.sidebar.columns(2)
start_clicked = col_start.button("▶️ Start", use_container_width=True, type="primary")
stop_clicked = col_stop.button("⏹️ Stop", use_container_width=True)
reset_clicked = st.sidebar.button("🔄 Reset Monitoring State", use_container_width=True)

if start_clicked:
    st.session_state.is_monitoring = True
if stop_clicked:
    st.session_state.is_monitoring = False
    st.session_state.pipeline.alert_manager.reset()
if reset_clicked:
    st.session_state.pipeline.reset_session()
    st.sidebar.success("Session state and warning score reset.")

st.sidebar.divider()
st.sidebar.subheader("System Status")
model_status = "✅ Operational" if st.session_state.pipeline.inference_engine.is_loaded else "⚠️ Model Missing"
st.sidebar.write(f"**ML Classifier:** {model_status}")
st.sidebar.write(f"**Primary Model:** Random Forest (100 trees)")
st.sidebar.write(f"**Alternative:** SVM (RBF kernel)")
st.sidebar.write(f"**Landmark Detector:** MediaPipe Face Landmarker")
st.sidebar.write(f"**Audio Alert:** {'🔊 Enabled' if settings.AUDIO_ALERT_ENABLED else '🔇 Disabled'}")

st.sidebar.divider()
st.sidebar.caption("Minor Project-I: Academic Prototype\nDev Bhoomi Uttarakhand University (DBUU)")

# Main Navigation Tabs
tab_monitor, tab_train, tab_logs, tab_architecture = st.tabs([
    "📹 Live Driver Monitoring",
    "🧠 Model Training & Evaluation",
    "📊 Operational Logs & Analytics",
    "📐 Architecture & Methodology"
])

# ==============================================================================
# TAB 1: LIVE DRIVER MONITORING
# ==============================================================================
with tab_monitor:
    feed_col, info_col = st.columns([7, 5])

    video_placeholder = feed_col.empty()
    status_placeholder = info_col.empty()
    metrics_placeholder = info_col.empty()
    alert_banner_placeholder = info_col.empty()

    if st.session_state.is_monitoring:
        camera = CameraManager()
        success, msg = camera.start()

        if not success:
            feed_col.error(f"Camera Initialization Error: {msg}")
            st.session_state.is_monitoring = False
        else:
            try:
                # Live acquisition loop
                while st.session_state.is_monitoring:
                    ret, raw_frame = camera.get_frame()
                    if not ret or raw_frame is None:
                        feed_col.warning("Waiting for webcam frame...")
                        time.sleep(0.03)
                        continue

                    # Execute full backend pipeline
                    result = st.session_state.pipeline.process_frame(raw_frame)

                    # Convert BGR to RGB for Streamlit video display
                    rgb_annotated = cv2.cvtColor(result["annotated_frame"], cv2.COLOR_BGR2RGB)
                    video_placeholder.image(rgb_annotated, channels="RGB", use_container_width=True)

                    # Update status badges
                    with status_placeholder.container():
                        render_state_badge(result["state"], result["is_alert_active"])

                    # Update metric cards
                    with metrics_placeholder.container():
                        render_metric_cards(
                            features=result["features"],
                            ml_pred=result["ml_prediction"],
                            warning_score=result["warning_score"],
                            fps=result["fps"],
                            latency_ms=result["latency_ms"]
                        )

                    # Contextual reasoning display
                    with alert_banner_placeholder.container():
                        st.info(f"**Decision Reason:** {result['trigger_reason']}")

                    time.sleep(0.01)  # Yield execution for Streamlit reactivity

            finally:
                camera.stop()
                st.session_state.pipeline.alert_manager.reset()
    else:
        video_placeholder.info("Click '▶️ Start' in the left sidebar to begin real-time driver alertness monitoring.")
        with status_placeholder.container():
            render_state_badge("STANDBY", False)
        with metrics_placeholder.container():
            render_metric_cards(
                features={"ear": 0.0, "mar": 0.0, "blink_rate": 0.0, "eye_closure_duration": 0, "head_pitch": 0.0, "head_roll": 0.0},
                ml_pred={"confidence": 0.0, "state": "STANDBY"},
                warning_score=0.0,
                fps=0.0,
                latency_ms=0.0
            )

# ==============================================================================
# TAB 2: MODEL TRAINING & EVALUATION
# ==============================================================================
with tab_train:
    st.subheader("Machine Learning Training Pipeline")
    st.write(
        "Train and validate Random Forest (baseline) and SVM (alternative) models "
        "using strict **subject-independent dataset splitting** to prevent driver leakage across partitions."
    )

    col_t1, col_t2 = st.columns([4, 8])

    with col_t1:
        st.markdown("#### Training Configuration")
        model_selection = st.selectbox("Select Classifier", ["RandomForest", "SVM"])
        num_subjects = st.slider("Number of Driver Subjects", min_value=4, max_value=12, value=6)
        samples_per_subject = st.slider("Samples per Subject", min_value=100, max_value=1000, value=300, step=50)

        train_button = st.button("🚀 Train Model", type="primary", use_container_width=True)

    with col_t2:
        if train_button:
            with st.spinner("Generating subject-independent benchmark dataset and training classifier..."):
                dataset_path = settings.DATASET_DIR / "driver_fatigue_dataset.csv"
                df_data = generate_benchmark_dataset(
                    output_path=dataset_path,
                    num_subjects=num_subjects,
                    samples_per_subject=samples_per_subject
                )

                trainer = ModelTrainer()
                results = trainer.train(df_data, model_type=model_selection, save_model=True)
                st.session_state.latest_eval_results = results
                # Reload model in pipeline
                st.session_state.pipeline.inference_engine._load_artifacts()
                st.success(f"{model_selection} model trained and saved successfully!")

        if st.session_state.latest_eval_results is not None:
            res = st.session_state.latest_eval_results
            st.markdown(f"#### Evaluation Results ({res['model_type']})")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{res['accuracy'] * 100:.2f}%")
            m2.metric("Precision", f"{res['precision'] * 100:.2f}%")
            m3.metric("Recall", f"{res['recall'] * 100:.2f}%")
            m4.metric("F1-Score", f"{res['f1_score'] * 100:.2f}%")

            st.write(f"**Subject-Independent Partitioning:**")
            st.write(f"- **Training Subjects:** {', '.join(res['train_subjects'])} ({res['train_samples']} samples)")
            st.write(f"- **Test Subjects (Unseen):** {', '.join(res['test_subjects'])} ({res['test_samples']} samples)")

            # Visualizations
            cm_arr = np.array(res["confusion_matrix"])
            cm_plot_path = st.session_state.evaluator.plot_confusion_matrix(cm_arr)

            v_col1, v_col2 = st.columns(2)
            with v_col1:
                st.image(str(cm_plot_path), caption="Confusion Matrix on Unseen Subjects", use_container_width=True)

            with v_col2:
                if res.get("feature_importances"):
                    fi_plot_path = st.session_state.evaluator.plot_feature_importance(res["feature_importances"])
                    st.image(str(fi_plot_path), caption="Random Forest Feature Importance", use_container_width=True)
                else:
                    st.info("Feature importance analysis is specific to Random Forest classifiers.")
        else:
            st.info("Click '🚀 Train Model' to execute subject-independent training and view validation metrics.")

# ==============================================================================
# TAB 3: OPERATIONAL LOGS & ANALYTICS
# ==============================================================================
with tab_logs:
    st.subheader("Operational Benchmarks & Event Logging")
    st.write("Real-time logging via Pandas to record frames, predictions, operational FPS, and latency.")

    df_logs = st.session_state.pipeline.logger.load_logs()

    op_metrics = st.session_state.evaluator.compute_operational_metrics(df_logs)

    if op_metrics.get("available", False):
        o1, o2, o3, o4 = st.columns(4)
        o1.metric("Total Frames Logged", op_metrics["total_frames_logged"])
        o2.metric("Average Frame Rate", f"{op_metrics['mean_fps']} FPS")
        o3.metric("Average Processing Latency", f"{op_metrics['mean_latency_ms']} ms")
        o4.metric("Drowsy Alarm Triggers", op_metrics["total_alert_frames"])

        st.markdown("#### Real-Time Session Dynamics")
        timeline_path = st.session_state.evaluator.plot_session_timeline(df_logs)
        if timeline_path and timeline_path.exists():
            st.image(str(timeline_path), caption="Temporal Dynamics of EAR, MAR, and Warning Score", use_container_width=True)

        st.markdown("#### Event Records Table")
        st.dataframe(df_logs.tail(50), use_container_width=True)

        # Download button
        csv_data = df_logs.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Event Log CSV",
            data=csv_data,
            file_name="driver_monitoring_events.csv",
            mime="text/csv"
        )
    else:
        st.info("No monitoring events recorded in current log file yet. Run a live monitoring session to generate logs.")

    if st.button("🗑️ Clear Local Logs"):
        st.session_state.pipeline.logger.clear_logs()
        st.success("Local log records cleared.")

# ==============================================================================
# TAB 4: ARCHITECTURE & METHODOLOGY
# ==============================================================================
with tab_architecture:
    st.subheader("System Architecture & Processing Methodology")
    st.markdown(
        """
        The software prototype strictly executes the end-to-end processing sequence specified in the project synopsis:
        
        ```text
        Camera Input ➔ Frame Pre-processing ➔ Face Detection ➔ Landmark Detection ➔
        Feature Calculation (EAR, MAR, Blinks, Head Pose) ➔ ML State Classification ➔
        Temporal Persistence Filtering ➔ On-Screen & Audio Alert Generation ➔ Event Logging
        ```
        
        #### Core Modules
        - **Module 1 (Video Acquisition):** Direct webcam capture with minimum buffer delay to avoid latency.
        - **Module 2 (Pre-processing):** Frame resizing (640x480), BGR to RGB conversion, optional brightness normalization.
        - **Module 3 & 4 (Face & Landmark Detection):** MediaPipe Face Landmarker extracting 468 2D/3D facial landmarks.
        - **Module 5 (Feature Engineering):**
          - **EAR (Eye Aspect Ratio):** Geometric ratio measuring eye openness.
          - **MAR (Mouth Aspect Ratio):** Geometric ratio measuring mouth opening and yawning.
          - **Blink Rate & Duration:** Tracking blink frequency and separating normal blinks from prolonged eye closure.
          - **Head Pose Dynamics:** PnP estimation for head pitch (nodding) and roll.
        - **Module 6 (Machine Learning):** Random Forest baseline and SVM alternative with subject-independent dataset splitting.
        - **Module 7 (Real-Time Inference):** Pre-loaded model performing instant frame prediction and confidence scoring.
        - **Module 8 (Temporal Decision Engine):** Multi-frame persistence tracking (Alert, Warning, Drowsy, Tracking Lost).
        - **Module 9 (Alert System):** Prominent on-screen warning and host audio alert.
        - **Module 10 (Local Logging):** Pandas-based local CSV logging for offline benchmarking.
        - **Module 11 (Evaluation):** Offline metrics (Accuracy, Precision, Recall, F1) and real-time operational benchmarks (FPS, Latency).
        """
    )
