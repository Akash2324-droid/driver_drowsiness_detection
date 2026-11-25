"""
Main Application Entry Point: AI-Powered Driver Drowsiness Detection System
Supports running in Streamlit dashboard mode or OpenCV GUI window mode.
Supports offline model training and evaluation benchmarks.
Strictly based on the project synopsis requirements.
"""

import sys
import argparse
import subprocess
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from config import settings
from backend.camera import CameraManager
from backend.pipeline import DrowsinessDetectionPipeline
from backend.model_training import ModelTrainer, generate_benchmark_dataset
from backend.evaluation import PerformanceEvaluator


def run_opencv_gui():
    """
    Runs the live driver monitoring system in a high-performance native OpenCV window.
    """
    print("=" * 70)
    print("AI-POWERED DRIVER DROWSINESS DETECTION SYSTEM (OpenCV GUI Mode)")
    print("Dev Bhoomi Uttarakhand University (DBUU) - Minor Project-I")
    print("=" * 70)
    print("Keyboard shortcuts:")
    print("  'q' or ESC : Exit application and safely release camera")
    print("  'r'        : Reset monitoring session and temporal score")
    print("=" * 70)

    camera = CameraManager()
    success, msg = camera.start()
    if not success:
        print(f"[ERROR] {msg}")
        return

    pipeline = DrowsinessDetectionPipeline()
    window_name = "AI Driver Drowsiness Detection - Live Monitoring"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)

    try:
        while True:
            ret, raw_frame = camera.get_frame()
            if not ret or raw_frame is None:
                continue

            # Execute unified backend pipeline
            result = pipeline.process_frame(raw_frame)
            annotated_frame = result["annotated_frame"]

            cv2.imshow(window_name, annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # ESC or 'q'
                print("\n[INFO] Termination signal received. Exiting...")
                break
            elif key == ord('r'):
                pipeline.reset_session()
                print("\n[INFO] Monitoring state and warning score reset.")

    finally:
        camera.stop()
        pipeline.close()
        cv2.destroyAllWindows()
        print("[INFO] Resources released successfully. Monitoring session closed.")


def run_streamlit_app():
    """
    Launches the interactive Streamlit dashboard.
    """
    app_path = PROJECT_ROOT / "frontend" / "app.py"
    python_exe = sys.executable
    cmd = [python_exe, "-m", "streamlit", "run", str(app_path)]
    print(f"[INFO] Launching Streamlit Dashboard: {' '.join(cmd)}")
    subprocess.run(cmd)


def run_training(model_type: str = "RandomForest"):
    """
    Executes subject-independent dataset generation and model training.
    """
    print(f"\n[INFO] Generating benchmark dataset and training {model_type} model...")
    dataset_path = settings.DATASET_DIR / "driver_fatigue_dataset.csv"
    df = generate_benchmark_dataset(dataset_path, num_subjects=6, samples_per_subject=300)
    print(f"[INFO] Dataset ready: {len(df)} samples across {df.subject_id.nunique()} subjects.")

    trainer = ModelTrainer()
    results = trainer.train(df, model_type=model_type, save_model=True)

    print("\n" + "=" * 50)
    print(f"EVALUATION RESULTS ({results['model_type']})")
    print("=" * 50)
    print(f"Accuracy  : {results['accuracy'] * 100:.2f}%")
    print(f"Precision : {results['precision'] * 100:.2f}%")
    print(f"Recall    : {results['recall'] * 100:.2f}%")
    print(f"F1-Score  : {results['f1_score'] * 100:.2f}%")
    print(f"Train Subjects: {results['train_subjects']}")
    print(f"Test Subjects : {results['test_subjects']}")
    print(f"Confusion Matrix: {results['confusion_matrix']}")
    print("=" * 50)

    evaluator = PerformanceEvaluator()
    cm_path = evaluator.plot_confusion_matrix(evaluator.evaluate_predictions(
        np.array([1]), np.array([1])
    ).get("confusion_matrix", [[0, 0], [0, 0]]) if False else np.array(results["confusion_matrix"]))
    print(f"[INFO] Saved confusion matrix plot to: {cm_path}")

    if results.get("feature_importances"):
        fi_path = evaluator.plot_feature_importance(results["feature_importances"])
        print(f"[INFO] Saved feature importance plot to: {fi_path}")


def main():
    parser = argparse.ArgumentParser(
        description="AI-Powered Driver Drowsiness Detection System (DBUU Minor Project-I)"
    )
    parser.add_argument(
        "--mode",
        choices=["streamlit", "opencv", "train"],
        default="opencv",
        help="Execution mode: 'streamlit' (web dashboard), 'opencv' (native desktop window), or 'train' (model training)."
    )
    parser.add_argument(
        "--model",
        choices=["RandomForest", "SVM"],
        default="RandomForest",
        help="Classifier architecture for training mode."
    )

    args = parser.parse_args()

    if args.mode == "opencv":
        run_opencv_gui()
    elif args.mode == "streamlit":
        run_streamlit_app()
    elif args.mode == "train":
        run_training(model_type=args.model)


if __name__ == "__main__":
    main()
