# C-MAPSS

Reproduce C-MAPSS turbofan engine RUL prediction results.

Pre-computed feature matrices with anonymized columns. No feature engineering
code — load, train, score.

## Quick Start

```bash
git clone https://github.com/rudder-framework/cmapss.git
cd cmapss
pip install -r requirements.txt
python run.py --dataset fd001 --seed 0
```

## Headline Results (30-seed validated)

| Dataset | RMSE mean±std | NASA mean±std | Gap | Features | vs Published SOTA |
|---------|--------------|--------------|-----|----------|-------------------|
| **FD001** | **10.31 ± 0.06** | **143.7 ± 1.7** | 1.14x | 88  | **−16.4%** RMSE |
| **FD002** | **12.90 ± 0.04** | **543.0 ± 4.1** | 0.78x | 275 | **−33.1%** RMSE |
| **FD003** | **10.69 ± 0.05** | **184.4 ± 2.8** | 0.93x | 149 | **−9.1%**  RMSE |
| **FD004** | **11.83 ± 0.03** | **724.5 ± 4.9** | 0.93x | 157 | **−40.8%** RMSE |

All 30 of 30 seeds beat published 2025/26 SOTA RMSE for every dataset.
See `RESULTS.md` for full statistics, error tails, and methodology.

## Reproduce a Single Result

```bash
python run.py --dataset fd001 --seed 0
# → RMSE 10.38, NASA 144, gap 1.13x
```

## 30-Seed Sweep

```bash
python run.py --dataset fd001 --seeds 0-29
```

Reports mean ± std for RMSE, NASA, gap ratio, and aggregate error tails.

## Datasets

Each dataset directory contains two parquet files:

- `train.parquet` — feature matrix (cohort, F001..FNNN, RUL)
- `test.parquet` — last cycle per engine (cohort, F001..FNNN, RUL)

| Dataset | Train Engines | Test Engines | Operating Conditions | Fault Modes |
|---------|--------------|-------------|---------------------|-------------|
| FD001 | 100 | 100 | 1 | 1 (HPC) |
| FD002 | 260 | 259 | 6 | 1 (HPC) |
| FD003 | 100 | 100 | 1 | 2 (HPC + Fan) |
| FD004 | 249 | 248 | 6 | 2 (HPC + Fan) |

The feature matrices are post-imputation, post-scaling. The researcher
loads them directly — no feature engineering, no imputation, no scaling
required.

Raw data source: [NASA C-MAPSS](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)

## Method

- **Model:** Stacking ensemble — LightGBM + XGBoost + HistGradientBoosting → RidgeCV
- **Validation:** 5-fold GroupKFold on engine ID (no leakage across engines)
- **Target:** RUL capped at 125 cycles (C-MAPSS standard)
- **Scoring:** RMSE + NASA PHM08 asymmetric penalty
- **No hyperparameter tuning.** Same architecture across all four datasets.
- **Deterministic:** seed propagated to every base model. Same seed produces
  bit-identical results.

For multi-operating-condition datasets (FD002, FD004), a physics-based
operating condition correction is applied at data ingestion. The pre-computed
feature matrices in this repository are derived from the corrected data.

## Leakage Audit

`run.py` automatically prints the maximum feature-target correlation before
training. All datasets fall within healthy bounds (max |corr| ≤ 0.89,
gap ratios 0.78–1.14x).

## Requirements

- Python 3.10+
- numpy, polars, scikit-learn, xgboost, lightgbm, pyarrow

## License

[PolyForm Strict 1.0.0](LICENSE.md) — use and reproduce, no distribution
or modification.

The feature engineering pipeline that produced the F001..FNNN columns is
not included in this repository. Researchers can verify the reported results
exactly using the provided feature matrices and `run.py`.
