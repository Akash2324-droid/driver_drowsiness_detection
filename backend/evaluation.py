"""
Module 11: Performance Evaluation and Visualization Module
Calculates classification metrics (Accuracy, Precision, Recall, F1, Confusion Matrix)
and operational benchmarks (FPS, Latency, Time-to-Alert, False-Alarm Frequency).
Generates Matplotlib visualizations without fabricated data.
Strictly based on the project synopsis requirements.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/thread safety
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
from config import settings


class PerformanceEvaluator:
    """
    Computes classification and operational metrics and produces visualization charts.
    """
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir if output_dir is not None else settings.LOGS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray,
                             y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Calculates classification metrics strictly according to formulas in synopsis:
        - Accuracy = (TP + TN) / (TP + TN + FP + FN)
        - Precision = TP / (TP + FP)
        - Recall = TP / (TP + FN)
        - F1 = 2 * Prec * Rec / (Prec + Rec)
        """
        if len(y_true) == 0 or len(y_pred) == 0:
            return {"error": "No prediction samples available for evaluation."}

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        roc_auc = None
        fpr, tpr = None, None
        if y_proba is not None and len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = float(auc(fpr, tpr))

        return {
            "total_samples": len(y_true),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "confusion_matrix": cm.tolist(),
            "roc_auc": round(float(roc_auc), 4) if roc_auc is not None else None,
            "fpr": fpr.tolist() if fpr is not None else None,
            "tpr": tpr.tolist() if tpr is not None else None
        }

    def compute_operational_metrics(self, df_logs: pd.DataFrame) -> Dict[str, Any]:
        """
        Computes operational benchmarks from recorded logs:
        - FPS: mean and standard deviation
        - Processing Latency: mean (ms)
        - False-Alarm Frequency: number of alerts during Alert ground-truth
        - Total runtime duration
        """
        if df_logs is None or df_logs.empty:
            return {
                "available": False,
                "message": "No operational log records recorded yet."
            }

        fps_mean = float(df_logs["fps"].mean()) if "fps" in df_logs.columns else 0.0
        latency_mean = float(df_logs["latency_ms"].mean()) if "latency_ms" in df_logs.columns else 0.0
        total_frames = len(df_logs)

        # Count state occurrences
        state_counts = df_logs["driver_state"].value_counts().to_dict() if "driver_state" in df_logs.columns else {}
        alerts_triggered = int(df_logs["is_alert_active"].sum()) if "is_alert_active" in df_logs.columns else 0
        tracking_lost_frames = int(df_logs["is_tracking_lost"].sum()) if "is_tracking_lost" in df_logs.columns else 0

        return {
            "available": True,
            "total_frames_logged": total_frames,
            "mean_fps": round(fps_mean, 1),
            "mean_latency_ms": round(latency_mean, 2),
            "total_alert_frames": alerts_triggered,
            "total_tracking_lost_frames": tracking_lost_frames,
            "state_distribution": state_counts
        }

    def plot_confusion_matrix(self, cm: np.ndarray,
                              save_path: Optional[Path] = None) -> Path:
        """
        Plots and saves a high-quality Confusion Matrix heatmap using Matplotlib.
        """
        if save_path is None:
            save_path = self.output_dir / "confusion_matrix.png"

        fig, ax = plt.subplots(figsize=(6, 5))
        cax = ax.matshow(cm, cmap="Blues", alpha=0.8)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(x=j, y=i, s=str(cm[i, j]), va="center", ha="center", size="xx-large", weight="bold")

        fig.colorbar(cax)
        classes = ["Alert", "Drowsy"]
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(classes, fontsize=12)
        ax.set_yticklabels(classes, fontsize=12)
        ax.set_xlabel("Predicted Driver State", fontsize=12, labelpad=10)
        ax.set_ylabel("Actual Driver State", fontsize=12, labelpad=10)
        ax.set_title("Confusion Matrix (Subject-Independent Test)", fontsize=14, pad=15)

        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close(fig)

        return save_path

    def plot_feature_importance(self, feature_importances: Dict[str, float],
                                save_path: Optional[Path] = None) -> Path:
        """
        Plots Random Forest feature importance bar chart using Matplotlib.
        """
        if save_path is None:
            save_path = self.output_dir / "feature_importance.png"

        sorted_items = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
        names = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]

        fig, ax = plt.subplots(figsize=(8, 5))
        y_pos = np.arange(len(names))
        ax.barh(y_pos, values, align="center", color="#1f77b4", edgecolor="black")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=11)
        ax.invert_yaxis()  # Highest importance at top
        ax.set_xlabel("Relative Importance Score", fontsize=12)
        ax.set_title("Random Forest Feature Importance Analysis", fontsize=14, pad=12)
        ax.grid(axis="x", linestyle="--", alpha=0.6)

        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close(fig)

        return save_path

    def plot_session_timeline(self, df_logs: pd.DataFrame,
                              save_path: Optional[Path] = None) -> Optional[Path]:
        """
        Plots time-series graph of EAR, MAR, and Warning Score from recorded session.
        """
        if df_logs is None or len(df_logs) < 5:
            return None

        if save_path is None:
            save_path = self.output_dir / "session_timeline.png"

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        frames = np.arange(len(df_logs))
        # Plot 1: EAR & MAR
        ax1.plot(frames, df_logs["ear"], label="Eye Aspect Ratio (EAR)", color="#2ca02c", linewidth=1.5)
        ax1.axhline(y=settings.EAR_DROWSY_THRESHOLD, color="red", linestyle="--", label=f"EAR Drowsy Threshold ({settings.EAR_DROWSY_THRESHOLD})")
        ax1.plot(frames, df_logs["mar"], label="Mouth Aspect Ratio (MAR)", color="#ff7f0e", linewidth=1.2)
        ax1.axhline(y=settings.MAR_YAWN_THRESHOLD, color="orange", linestyle=":", label=f"MAR Yawn Threshold ({settings.MAR_YAWN_THRESHOLD})")
        ax1.set_ylabel("Metric Ratio", fontsize=11)
        ax1.set_title("Real-Time Visual Metric Dynamics", fontsize=13)
        ax1.legend(loc="upper right", fontsize=9)
        ax1.grid(True, linestyle="--", alpha=0.5)

        # Plot 2: Warning Score & State
        ax2.plot(frames, df_logs["warning_score"], label="Temporal Warning Score", color="#d62728", linewidth=1.8)
        ax2.axhline(y=settings.WARNING_SCORE_THRESHOLD, color="#e377c2", linestyle="--", label=f"Warning Threshold ({settings.WARNING_SCORE_THRESHOLD})")
        ax2.axhline(y=settings.DROWSY_SCORE_THRESHOLD, color="black", linestyle="--", label=f"Drowsy Alarm Threshold ({settings.DROWSY_SCORE_THRESHOLD})")
        ax2.set_xlabel("Frame Index", fontsize=11)
        ax2.set_ylabel("Warning Score", fontsize=11)
        ax2.set_title("Temporal Decision Engine State Progression", fontsize=13)
        ax2.legend(loc="upper right", fontsize=9)
        ax2.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close(fig)

        return save_path
