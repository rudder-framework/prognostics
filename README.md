# Prognostics

Reproduce C-MAPSS turbofan engine RUL prediction results.

Pre-computed feature matrices with anonymized columns. No feature engineering code — load, train, score.

## Quick Start

```bash
git clone https://github.com/rudder-framework/prognostics.git
cd prognostics
pip install -r requirements.txt
python run.py --dataset fd001 --seed 42
```

## Results

| Dataset | RMSE | NASA | Features | Gap |
|---------|------|------|----------|-----|
| FD001 | 10.45 | 147 | 88 | 1.12x |
| FD002 | 12.70 | 550 | 275 | 0.79x |
| FD003 | 10.74 | 179 | 149 | 0.92x |
| FD004 | 11.98 | 746 | 157 | 0.92x |

All results use seed=42 with deterministic seeding.

## 30-Seed Sweep

```bash
python run.py --dataset fd001 --seeds 0-29
```

## Datasets

Each dataset directory contains two parquet files:

- `train.parquet` — per-cycle features (cohort, F001..FNNN, RUL)
- `test.parquet` — last cycle per engine (cohort, F001..FNNN, RUL)

| Dataset | Train Engines | Test Engines | Operating Conditions | Fault Modes |
|---------|--------------|-------------|---------------------|-------------|
| FD001 | 100 | 100 | 1 | 1 |
| FD002 | 260 | 259 | 6 | 1 |
| FD003 | 100 | 100 | 1 | 2 |
| FD004 | 249 | 248 | 6 | 2 |

Raw data source: [NASA C-MAPSS](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)

## Method

- **Model:** Stacking ensemble — LightGBM + XGBoost + HistGradientBoosting → RidgeCV
- **Validation:** 5-fold GroupKFold on engine ID (no data leakage across engines)
- **Target:** RUL capped at 125 cycles
- **Scoring:** NASA PHM08 asymmetric penalty + RMSE

## Requirements

- Python 3.10+
- numpy, polars, scikit-learn, xgboost, lightgbm, pyarrow

## License

[PolyForm Strict 1.0.0](LICENSE.md) — use and reproduce, no distribution or modification.
