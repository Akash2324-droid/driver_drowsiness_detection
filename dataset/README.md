# Dataset Management & Subject-Independent Partitioning

This directory stores datasets for training, validating, and testing driver drowsiness detection models.

## Labeled Dataset Format

Datasets are formatted as CSV files containing both geometric visual features and subject identifiers:

| Column | Type | Description |
|---|---|---|
| `subject_id` | String | Unique driver identifier (e.g. `Driver_01`, `Driver_02`) |
| `ear` | Float | Real-time Eye Aspect Ratio |
| `mar` | Float | Real-time Mouth Aspect Ratio |
| `blink_rate` | Float | Rolling blink frequency (blinks per minute) |
| `eye_closure_duration` | Integer | Consecutive frames where eyes are closed |
| `head_pitch` | Float | Estimated head pitch angle (degrees, downward nod) |
| `head_roll` | Float | Estimated lateral head tilt angle (degrees) |
| `ear_rolling_mean` | Float | Moving average of EAR over sliding temporal window |
| `ear_rolling_std` | Float | Moving standard deviation of EAR |
| `mar_rolling_mean` | Float | Moving average of MAR over sliding temporal window |
| `label` | Integer | Target state: `0` = Alert, `1` = Drowsy |

## Subject-Independent Dataset Splitting

As strictly mandated by the project synopsis:
- **No Frame Leakage:** Video frames belonging to a single driver subject are never randomly shuffled into both train and test splits.
- **Partitioning Strategy:** Uses `GroupShuffleSplit` grouping strictly by `subject_id`.
- Models are evaluated exclusively on unseen driver facial shapes, avoiding optimistic accuracy inflation.

## Synthetic Benchmark Generator

If no external video dataset is present, the pipeline provides a built-in benchmark generator (`generate_benchmark_dataset` in `backend/model_training.py`) that generates multi-subject distributions adhering to empirical driver fatigue patterns.
