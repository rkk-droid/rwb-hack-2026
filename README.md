# RWB Hackathon 2026 — Warehouse Shipment Volume Forecasting

**Leaderboard position: 24 / 228** | [Competition page](https://wbspace.wb.ru/competitions/otgruzki-bez-prostoev)

Forecast hourly shipment volumes per route for the next 4 hours (8 × 30-minute steps), using historical warehouse processing statuses as input features.

## Task

- **Target:** `target_1h` — number of items shipped per route in the last hour
- **Horizon:** 8 future steps × 30 min = 4 hours ahead
- **Scale:** 1,000 routes × 8 steps = 8,000 predictions
- **Metric:** WAPE + Relative Bias

$$\text{WAPE} + \text{RelBias} = \frac{\sum|y_i - \hat{y}_i|}{\sum y_i} + \left|\frac{\sum \hat{y}_i}{\sum y_i} - 1\right|$$

## Solution Overview

The pipeline has four sequential stages:

```
train_solo_track.parquet
        │
        ▼
[00] generate_features  →  features.parquet   (218 features, Polars)
        │
        ▼
[01] feature_selection  →  feature_masks.pkl  (RFECV per forecast step)
        │
        ▼
[02] model_fitting      →  submission.csv     (Ridge × 8 + calibration)
```

### Key design decisions

| Decision | Rationale |
|---|---|
| Train on October only | `status_4/5` show anomalous behaviour in Aug/Sep (bump in Sep, dips in Aug); excluding them improves generalisation |
| 8 separate models | Each forecast step has a different optimal feature set |
| Ridge over Gradient Boosting | GBM tested and performed noticeably worse, likely due to generalisation difficulties across 1,000 heterogeneous routes |
| RFECV per step | Importance of lag features shifts with horizon (short lags dominate early, weekly lags dominate late) |
| Per-group calibration | Directly optimises competition metric (WAPE + RelBias) with per-(route, step) affine transform |

## Repository Structure

```
.
├── modules/
│   ├── generate_features.py   # Polars feature engineering pipeline
│   ├── calibration.py         # Three calibration methods (global / per-group bias / per-group scale+bias)
│   └── prepare_data.py        # Optional data filters (cut august/september, etc.)
├── data/
│   ├── train_solo_track.parquet.zip   # Raw training data (unzip before use)
│   ├── test_solo_track.parquet.zip    # Raw test data   (unzip before use)
│   ├── features.parquet.zip           # Pre-built feature matrix (unzip before use)
│   └── feature_masks.pkl              # RFECV masks for each forecast step
├── 00-generate_features.ipynb         # Build features.parquet from raw training data
├── 01-feature_selection.ipynb         # RFECV feature selection (reads features.parquet)
├── 02-model_fitting.ipynb             # Training, prediction, calibration, submission
├── presentation/
│   └── presentation.pdf               # Solution presentation
└── README.md
```

## Setup

### Requirements

Install all dependencies:

```bash
pip install -r requirements.txt
```

### Data preparation

Unzip the data files before running the notebooks:

```bash
cd data
unzip features.parquet.zip
unzip train_solo_track.parquet.zip
unzip test_solo_track.parquet.zip
```

## Running the Pipeline

### Step 0 — Feature engineering (`00-generate_features.ipynb`)

Reads `data/train_solo_track.parquet`, builds all 218 features, filters out August and September, and writes `data/features.parquet`.

> `features.parquet` is also provided pre-built (zipped) — you can skip this step and go straight to Step 1.

### Step 1 — Feature selection

Open and run **`01-feature_selection.ipynb`**.

- Reads `data/features.parquet`
- Runs RFECV with Ridge regression (3-fold CV, subsample of 300K rows)
- Produces `data/feature_masks.pkl` — a separate feature mask for each of the 8 forecast steps

> This step is slow (~20–40 min). `feature_masks.pkl` is already provided in the repo.

### Step 2 — Training, calibration, submission

Open and run **`02-model_fitting.ipynb`**.

- Reads `data/features.parquet`, `data/test_solo_track.parquet`, `data/feature_masks.pkl`
- Trains 8 Ridge models (one per forecast step) with `StandardScaler`
- Generates predictions for all 8,000 test rows
- Calibrates via per-group scale + bias optimised directly on WAPE + RelBias
- Saves `submission.csv`

## Feature Groups (218 total)

| Group | Description |
|---|---|
| Status aggregates | Sums and proportions of `status_1–3` (current warehouse) and `status_4–6` (upstream warehouse) |
| Lag features | 14 lags: 1–8 short-term steps + 24/48/96/144/192/240/288/336 long-term steps, for all 6 statuses and `target_1h` |
| Rolling statistics | mean + std over windows 5, 10, 48, 336, for all 6 statuses and `target_1h` |
| Weekly periodicity | Average of `target_1h` at multiples of lags 327–336 (both past and future), smoothing individual-lag noise |
| Time features | One-hot hour of day (24), day of week (7), `is_weekend` |
| Target encoding | `route_mean`, `route_hour_mean`, `route_weekday_mean` |

## Calibration

Three methods compared on the last 20% of training data by time:

| Method | Val metric | WAPE | RelBias |
|---|---|---|---|
| No calibration | 0.3354 | 0.3281 | 0.0073 |
| Global scalar | 0.3275 | 0.3275 | 0.0000 |
| Per-group additive bias | 0.3260 | 0.3260 | 0.0000 |
| **Per-group scale + bias** | **0.3255** | **0.3255** | **0.0000** |

The selected method fits $\hat{y}' = k_{r,s} \cdot \hat{y} + b_{r,s}$ for each (route, step) pair using L-BFGS-B with analytical gradients, optimising the competition metric directly.

## Results

- **Validation metric (WAPE + RelBias):** 0.325
- **Leaderboard:** 24 / 228
- **Generalisation:** all test timestamps are Saturdays 10:30–14:00; the model was trained on all days and times with no test-specific tuning
