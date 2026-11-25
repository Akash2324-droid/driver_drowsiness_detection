# Model Artifacts Directory

This directory stores serialized machine learning models, scalers, and neural vision weights.

## Artifacts

1. **`face_landmarker.task`**:
   - Google MediaPipe Face Landmarker model asset.
   - Detects 468 2D/3D facial landmarks with sub-pixel precision in real time.

2. **`drowsiness_rf_model.pkl`**:
   - Primary baseline classifier: Random Forest (100 estimators).
   - Trained on subject-independent multi-driver dataset.
   - Exports feature importances for explainability.

3. **`drowsiness_svm_model.pkl`**:
   - Alternative candidate model: Support Vector Machine (RBF kernel, probability calibration).
   - Maintained for comparative academic evaluation.

4. **`feature_scaler.pkl`**:
   - Scikit-learn `StandardScaler` fitted on training subjects' feature distributions.
   - Applied during real-time inference before passing feature vectors to the classifier.

## Model Training Command

To re-train the models at any time, run:

```powershell
python main.py --mode train --model RandomForest
```
or
```powershell
python main.py --mode train --model SVM
```
