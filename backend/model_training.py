"""
Module 6: Machine Learning Training Pipeline
Implements Random Forest (baseline) and SVM (alternative candidate) classifiers.
Enforces strict subject-independent dataset partitioning to prevent driver leakage.
Produces trained model artifacts, feature importances, and validation metrics.
Strictly based on the project synopsis requirements.
"""

import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from config import settings


def generate_benchmark_dataset(output_path: Optional[Path] = None,
                               num_subjects: int = 6,
                               samples_per_subject: int = 400,
                               random_state: int = 42) -> pd.DataFrame:
    """
    Generates a realistic, labeled benchmark dataset across multiple subjects
    following empirical distributions of alert and drowsy driver facial metrics.
    Features:
    - ear, mar, blink_rate, eye_closure_duration, head_pitch, head_roll,
      ear_rolling_mean, ear_rolling_std, mar_rolling_mean
    - subject_id: string identifier
    - label: 0 (Alert), 1 (Drowsy)
    """
    np.random.seed(random_state)
    data = []

    for subj_idx in range(1, num_subjects + 1):
        subject_id = f"Driver_{subj_idx:02d}"
        
        # Individual subject baseline variations (eye shape, blink habit)
        subj_ear_base = np.random.normal(0.32, 0.02)
        subj_mar_base = np.random.normal(0.35, 0.03)

        n_samples = samples_per_subject
        n_alert = n_samples // 2
        n_drowsy = n_samples - n_alert

        # Alert State Samples (Normal alertness, occasional short blinks)
        for _ in range(n_alert):
            # 5% chance of a short normal blink in this sample
            is_blink = (np.random.rand() < 0.05)
            if is_blink:
                ear = np.random.uniform(0.12, 0.19)
                closure_dur = np.random.randint(1, 5)
            else:
                ear = np.clip(np.random.normal(subj_ear_base, 0.025), 0.25, 0.42)
                closure_dur = 0

            mar = np.clip(np.random.normal(subj_mar_base, 0.04), 0.20, 0.48)
            blink_rate = np.clip(np.random.normal(16.0, 4.0), 8.0, 26.0)
            pitch = np.random.normal(0.0, 5.0)
            roll = np.random.normal(0.0, 4.0)
            ear_mean = np.clip(np.random.normal(subj_ear_base, 0.02), 0.26, 0.38)
            ear_std = np.random.uniform(0.01, 0.04)
            mar_mean = np.clip(np.random.normal(subj_mar_base, 0.03), 0.25, 0.42)

            data.append({
                "subject_id": subject_id,
                "ear": ear,
                "mar": mar,
                "blink_rate": blink_rate,
                "eye_closure_duration": closure_dur,
                "head_pitch": pitch,
                "head_roll": roll,
                "ear_rolling_mean": ear_mean,
                "ear_rolling_std": ear_std,
                "mar_rolling_mean": mar_mean,
                "label": 0  # Alert
            })

        # Drowsy State Samples (Prolonged eye closure, yawning, drooping head)
        for _ in range(n_drowsy):
            # Fatigue pattern: prolonged eye closure or yawning or head nodding
            fatigue_mode = np.random.choice(["eye_closed", "yawn", "microsleep", "nodding"], p=[0.4, 0.25, 0.25, 0.10])
            
            if fatigue_mode in ["eye_closed", "microsleep"]:
                ear = np.random.uniform(0.08, 0.18)
                closure_dur = np.random.randint(12, 35) if fatigue_mode == "microsleep" else np.random.randint(8, 20)
                mar = np.clip(np.random.normal(subj_mar_base + 0.05, 0.04), 0.30, 0.50)
                pitch = np.random.normal(-12.0, 6.0)
            elif fatigue_mode == "yawn":
                ear = np.clip(np.random.normal(0.20, 0.03), 0.14, 0.25)
                closure_dur = np.random.randint(4, 12)
                mar = np.clip(np.random.normal(0.65, 0.08), 0.55, 0.85)  # Yawning MAR
                pitch = np.random.normal(-8.0, 5.0)
            else:  # nodding
                ear = np.clip(np.random.normal(0.19, 0.03), 0.12, 0.24)
                closure_dur = np.random.randint(6, 15)
                mar = np.clip(np.random.normal(subj_mar_base, 0.04), 0.28, 0.45)
                pitch = np.random.normal(-20.0, 5.0)  # Heavy head drop

            roll = np.random.normal(0.0, 6.0)
            blink_rate = np.clip(np.random.normal(8.0, 3.0), 3.0, 14.0)  # Slowed blink frequency
            ear_mean = np.clip(np.random.normal(0.18, 0.03), 0.12, 0.23)
            ear_std = np.random.uniform(0.04, 0.08)
            mar_mean = np.clip(np.random.normal(0.48 if fatigue_mode == "yawn" else 0.38, 0.05), 0.30, 0.65)

            data.append({
                "subject_id": subject_id,
                "ear": ear,
                "mar": mar,
                "blink_rate": blink_rate,
                "eye_closure_duration": closure_dur,
                "head_pitch": pitch,
                "head_roll": roll,
                "ear_rolling_mean": ear_mean,
                "ear_rolling_std": ear_std,
                "mar_rolling_mean": mar_mean,
                "label": 1  # Drowsy
            })

    df = pd.DataFrame(data)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    return df


class ModelTrainer:
    """
    Trains and evaluates Random Forest and SVM models using subject-independent splitting.
    """
    def __init__(self, feature_names: list = None):
        self.feature_names = feature_names if feature_names is not None else settings.FEATURE_NAMES
        self.scaler = StandardScaler()
        self.model = None
        self.model_type = "RandomForest"

    def split_subject_independent(self, df: pd.DataFrame,
                                  test_size: float = 0.25,
                                  random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits dataset into Train and Test partitions using subject-independent grouping.
        All records belonging to a particular driver subject exist exclusively in one split.
        """
        if "subject_id" not in df.columns:
            raise ValueError("Dataset must contain 'subject_id' column for subject-independent splitting.")

        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(df, groups=df["subject_id"]))

        train_df = df.iloc[train_idx].copy().reset_index(drop=True)
        test_df = df.iloc[test_idx].copy().reset_index(drop=True)

        return train_df, test_df

    def train(self, df: pd.DataFrame,
              model_type: str = "RandomForest",
              save_model: bool = True) -> Dict[str, Any]:
        """
        Trains model with subject-independent validation.
        """
        self.model_type = model_type

        # Subject-independent split
        train_df, test_df = self.split_subject_independent(df)

        train_subjects = sorted(train_df["subject_id"].unique().tolist())
        test_subjects = sorted(test_df["subject_id"].unique().tolist())

        X_train = train_df[self.feature_names].values
        y_train = train_df["label"].values
        X_test = test_df[self.feature_names].values
        y_test = test_df["label"].values

        # Fit Scaler on training data only
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Initialize classifier
        if model_type == "RandomForest":
            clf = RandomForestClassifier(**settings.RF_PARAMS)
        elif model_type == "SVM":
            clf = SVC(**settings.SVM_PARAMS)
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Use 'RandomForest' or 'SVM'.")

        clf.fit(X_train_scaled, y_train)
        self.model = clf

        # Evaluation on unseen test subjects
        y_pred = clf.predict(X_test_scaled)
        y_proba = clf.predict_proba(X_test_scaled)[:, 1] if hasattr(clf, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        # Feature importances for Random Forest
        feature_importances = None
        if hasattr(clf, "feature_importances_"):
            feature_importances = dict(zip(self.feature_names, clf.feature_importances_.tolist()))

        # ROC Curve
        fpr, tpr, roc_auc = None, None, None
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = float(auc(fpr, tpr))

        # Save artifacts
        if save_model:
            model_slug = "rf" if model_type.lower() in ["randomforest", "rf"] else "svm"
            model_file = settings.MODELS_DIR / f"drowsiness_{model_slug}_model.pkl"
            scaler_file = settings.SCALER_PATH
            joblib.dump(clf, model_file)
            joblib.dump(self.scaler, scaler_file)

        return {
            "model_type": model_type,
            "train_subjects": train_subjects,
            "test_subjects": test_subjects,
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "confusion_matrix": cm.tolist(),
            "feature_importances": feature_importances,
            "roc_auc": roc_auc,
            "fpr": fpr.tolist() if fpr is not None else None,
            "tpr": tpr.tolist() if tpr is not None else None
        }
