# AI-Powered Driver Drowsiness Detection System Using Computer Vision and Machine Learning for Real-Time Driver Alertness Monitoring and Accident Prevention

**Academic Live Project / Minor Project-I**  
**Institution:** Dev Bhoomi Uttarakhand University (DBUU), Dehradun  
**School & Department:** School of Engineering & Computing (SoEC), Department of Computer Science & Engineering  
**Students:** Tanishq Singh (24BTCSEAI0159), Akash Kumar Nishad (24BTCSEAI0147), Akanksha Rai (24BTCSEAI0143)  
**Supervisor:** Mr. Anubhav Singh Bisht (Assistant Professor, CSE)

---

## 1. Project Introduction

Driver drowsiness and cognitive fatigue represent major causal factors in roadway collisions worldwide. Fatigue develops gradually, often presenting subtle visual precursors before a catastrophic microsleep episode occurs. 

This project implements a non-intrusive, software-based early-warning safety assistant. By capturing a continuous video feed from a standard laptop webcam, the system:
1. Isolates the driver's facial region and extracts 468 2D/3D landmarks using Google MediaPipe Face Landmarker.
2. Computes interpretable geometric metrics: Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), blink frequency, eye-closure duration, and head orientation angles (Pitch, Yaw, Roll).
3. Classifies driver cognitive state using supervised machine learning (Random Forest baseline and Support Vector Machine alternative) trained via strict subject-independent partitioning.
4. Enforces multi-frame temporal persistence logic to filter transient noise, brief normal blinks, and isolated yawns without triggering nuisance alerts.
5. Instantly delivers prominent on-screen visual banners and host system audible alerts upon confirmation of sustained drowsiness.
6. Asynchronously records structured operational event logs to local disk for performance benchmarking.

---

## 2. Project Objectives

- **Real-Time Video Processing:** Continuously capture and process webcam frames with low latency and stable frame rates on standard laptop hardware.
- **Precision Landmark Tracking:** Detect facial landmarks around eyes, lips, and head contours without intrusive physical sensors.
- **Visual Metric Computation:** Quantify eye openness (EAR), mouth expansion (MAR), blink dynamics, and downward head nodding.
- **Subject-Independent ML Classification:** Train Random Forest and SVM classifiers ensuring zero driver leakage between training and testing splits.
- **False-Alarm Mitigation:** Employ temporal sliding windows and warning score accumulation to differentiate natural blinks from microsleeps.
- **Multimodal Alert Generation:** Display high-contrast GUI state indicators and sound audible host alarms upon sustained fatigue.
- **Tracking-Loss Isolation:** Classify missing face or camera obstruction as tracking failure rather than falsely incrementing fatigue scores.
- **Operational Benchmarking:** Evaluate classification metrics (Accuracy, Precision, Recall, F1, Confusion Matrix) alongside operational metrics (FPS, Latency, Time-to-Alert).

---

## 3. Technologies Used

Strictly confined to tools specified in the project synopsis:
- **Python 3.11** – Core programming language and runtime environment.
- **OpenCV (cv2)** – Camera capture, image preprocessing, geometric transformation, and native GUI rendering.
- **Google MediaPipe (Face Landmarker)** – Real-time 2D/3D facial mesh and landmark extraction.
- **Scikit-learn** – Machine learning training, cross-validation, feature scaling, and evaluation metrics.
- **NumPy** – Numerical vector operations, Euclidean distance, and geometric ratio calculations.
- **Pandas** – Structured dataset handling, logging buffers, and event CSV exports.
- **Matplotlib** – Offline evaluation visualization, confusion matrices, feature-importance plots, and timeline charts.
- **Streamlit** – Interactive web dashboard interface for live monitoring and model analysis.
- **Git / GitHub** – Version control and repository management.

---

## 4. Folder Structure

The project maintains a strict separation of concerns between backend logic and frontend presentation:

```text
driver_drowsiness_detection/
│
├── frontend/
│   ├── __init__.py
│   ├── app.py                     # Streamlit web dashboard
│   └── ui_components.py           # Reusable metric cards, badges, and alerts
│
├── backend/
│   ├── __init__.py
│   ├── camera.py                  # Module 1: Webcam capture and buffer control
│   ├── preprocessing.py           # Module 2: Frame resizing and color conversion
│   ├── face_detection.py          # Module 3: Face region and bounding box isolation
│   ├── landmark_detection.py      # Module 4: MediaPipe Face Landmarker interface
│   ├── ear.py                     # Module 5.1: Eye Aspect Ratio (EAR) computation
│   ├── mar.py                     # Module 5.2: Mouth Aspect Ratio (MAR) computation
│   ├── blink_detection.py         # Module 5.3: Blink rate and closure duration tracker
│   ├── head_pose.py               # Module 5.4: 3D head pose and nodding estimation (PnP)
│   ├── feature_extraction.py      # Module 5: Aggregates metrics & rolling statistics
│   ├── model_training.py          # Module 6: Subject-independent training pipeline
│   ├── model_inference.py         # Module 7: Real-time frame inference engine
│   ├── temporal_logic.py          # Module 8: Temporal decision and warning score engine
│   ├── alert_system.py            # Module 9: Non-blocking audio & visual alert manager
│   ├── logging_system.py          # Module 10: Local Pandas event logger
│   ├── evaluation.py              # Module 11: Confusion matrix, metrics, and plots
│   └── pipeline.py                # Unified backend orchestration pipeline
│
├── config/
│   ├── __init__.py
│   └── settings.py                # System thresholds, constants, and hyperparameters
│
├── dataset/
│   ├── README.md                  # Dataset documentation & column definitions
│   └── driver_fatigue_dataset.csv # Labeled multi-subject benchmark dataset
│
├── models/
│   ├── README.md                  # Model weights documentation
│   ├── face_landmarker.task       # MediaPipe Face Landmarker model asset
│   ├── drowsiness_rf_model.pkl    # Trained Random Forest baseline model
│   ├── drowsiness_svm_model.pkl   # Trained Support Vector Machine model
│   └── feature_scaler.pkl         # Fitted StandardScaler artifact
│
├── logs/
│   └── system_events.csv          # Real-time operational logs
│
├── requirements.txt               # Project dependencies
├── test_system_e2e.py             # Integration test suite (15 verification tests)
├── main.py                        # Execution runner (OpenCV GUI or Streamlit)
└── README.md                      # Project documentation
```

---

## 5. Prerequisites

- **Operating System:** Windows 10/11 (64-bit)
- **Webcam:** Standard integrated laptop webcam or external USB camera (720p recommended)
- **Hardware:** Standard CPU with at least 4 GB RAM (8 GB recommended)
- **Audio:** Host system speakers or headphones for audible alerts
- **Python:** Python 3.10 or 3.11 installed

---

## 6. Installation & Virtual Environment Setup

Open PowerShell in the project directory:

```powershell
# Navigate to the project folder
cd C:\Users\HP\.gemini\antigravity\scratch\driver_drowsiness_detection

# Create a dedicated virtual environment
python -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1
```

*Note: If PowerShell displays an Execution Policy restriction, run:*
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

---

## 7. Dependency Installation

Install all verified dependencies into the virtual environment:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 8. Dataset Preparation & Subject-Independent Training

The repository includes a multi-subject benchmark dataset (`dataset/driver_fatigue_dataset.csv`) generated across 6 distinct driver profiles with empirical distributions for alert and drowsy states.

To train the Random Forest baseline classifier or the SVM alternative:

```powershell
# Train Random Forest baseline (with subject-independent validation)
python main.py --mode train --model RandomForest

# Or train SVM alternative
python main.py --mode train --model SVM
```

During training:
1. `GroupShuffleSplit` assigns each driver subject exclusively to either the training set or the test set.
2. The model evaluates against unseen driver faces.
3. Serialized models (`drowsiness_rf_model.pkl`, `drowsiness_svm_model.pkl`) and the scaler (`feature_scaler.pkl`) are saved to `models/`.
4. High-resolution Confusion Matrix and Feature Importance plots are saved to `logs/`.

---

## 9. How to Run the Application

The system supports two execution interfaces:

### Option A: Interactive Streamlit Web Dashboard (Recommended)

```powershell
python main.py --mode streamlit
```
*Or directly via Streamlit:*
```powershell
streamlit run frontend/app.py
```
This opens the browser dashboard at `http://localhost:8501`, featuring:
- **Live Driver Monitoring Tab:** Start/Stop webcam monitoring, view real-time annotated feed, high-contrast state badges, live metric cards (EAR, MAR, Blinks, Pitch, Score), and decision reason.
- **Model Training & Evaluation Tab:** Retrain models interactively, configure subject count, inspect subject-independent splits, and view Confusion Matrix and Feature Importance plots.
- **Operational Logs Tab:** View live session events, average FPS, latency benchmarks, session timeline plots, and download CSV logs.
- **Architecture Tab:** Overview of the complete processing workflow.

### Option B: Native Desktop OpenCV GUI Window

For minimal latency and direct desktop window monitoring:

```powershell
python main.py --mode opencv
```

**Keyboard Controls in OpenCV Mode:**
- Press `q` or `ESC` to stop monitoring and safely release the camera.
- Press `r` to reset the temporal warning score.

---

## 10. Operational Methodology & Decision Logic

### The Four Driver States

1. **`ALERT` (Normal Monitoring):**
   - Eyes open (EAR >= 0.22), normal blinking (12-24 blinks/min), no downward head nodding.
   - Warning score decrements toward `0.0`. No alarms triggered.
2. **`WARNING` (Early Fatigue):**
   - Repeated short closures or isolated yawning patterns observed.
   - Temporal warning score reaches `15.0`. Display shows an amber warning banner.
3. **`DROWSY` (Sustained Fatigue Confirmed):**
   - Prolonged eye closure (>= 15 frames / ~0.5s) or compounded fatigue (yawn + closure).
   - Temporal warning score reaches `30.0`.
   - High-contrast red flashing visual banner displayed on-screen.
   - Non-blocking continuous host audio alert plays via system speakers.
4. **`TRACKING_LOST` (Camera Obstructed / Missing Face):**
   - Driver turns away or face is occluded.
   - The system flags `TRACKING_LOST` and holds/freezes the fatigue score.
   - **Crucial Rule:** The system never falsely increments fatigue or triggers drowsiness alarms when the face is not visible.

---

## 11. Testing and Verification

Run the comprehensive automated integration test suite containing 15 end-to-end tests:

```powershell
python test_system_e2e.py
```

The test suite validates:
- [x] Camera Manager interface
- [x] Frame Preprocessing (resizing & LAB/RGB color space)
- [x] MediaPipe Face Landmarker execution
- [x] Geometric EAR calculation (open vs. closed eye thresholds)
- [x] Geometric MAR calculation (normal vs. yawning thresholds)
- [x] Blink Tracker (frequency & eye-closure duration)
- [x] Head Pose Dynamics (PnP pitch, yaw, roll, downward nodding)
- [x] 9-Dimensional Feature Extractor
- [x] Subject-Independent Dataset Splitting (zero subject leakage)
- [x] Random Forest training & metric validation
- [x] Real-time inference engine & probability scores
- [x] Temporal Decision Engine & all 4 states (`ALERT`, `WARNING`, `DROWSY`, `TRACKING_LOST`)
- [x] Audio Alert thread safety & recovery reset
- [x] Local Event Logger & CSV schema
- [x] Performance Evaluator & Matplotlib visualizations
- [x] Unified Pipeline execution on video frames

---

## 12. Scope & Limitations

As explicitly documented in the project synopsis:
- **Academic Early-Warning Assistant:** This system is an academic software prototype and is not a certified automotive safety system.
- **No Vehicle Control:** Does not interface with vehicle mechanical controls (no automatic braking, steering intervention, or throttle management).
- **No CAN-Bus Integration:** Does not connect to automotive electronic control units (ECU).
- **Non-Diagnostic:** Not intended as a medical device for diagnosing clinical sleep disorders.
- **Environmental Boundaries:** Tracking precision may degrade in severe low-light environments (NIR camera integration is identified as future work) or when wearing dark sunglasses that occlude the eye landmarks.
