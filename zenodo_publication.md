# Information-Preserving Embedding & Manifold Coherence Analysis — Features on NASA C-MAPSS

**Results and Reproducibility Record**

Alvin Rudder · Independent Researcher · Ørthon Research · West Lafayette, IN
April 2026 · CC BY-NC 4.0 · Patent Pending: USPTO No. 64/027,466

**DOI:** _pending — Zenodo deposit._

---

## Abstract

This deposit provides pre-computed feature matrices and a reproduction script for deterministic Remaining Useful Life (RUL) prediction on the NASA C-MAPSS benchmark datasets (FD001–FD004). Features were derived using Information-Preserving Embedding (IPE) and Manifold Coherence Analysis (MCA), a geometric analytical framework for multivariate sensor data. A stacking ensemble model establishes reproducible benchmark results validated across 30 independent seeds on consumer hardware with no GPU required.

The feature-engineering pipeline is covered by a pending US utility patent and is not included in this deposit. Researchers interested in collaboration, licensing, or academic application of the framework are invited to reach out at <orthon@orthon.io>.

---

## 30-Seed Validated Results

Results validated across seeds 0–29. No seed selection. Full distribution and error tails reported.

| Dataset | RMSE mean ± std | NASA mean ± std | Gap Ratio | RMSE Range | Features |
|---------|-----------------|-----------------|-----------|------------|----------|
| **FD001** | **10.31 ± 0.06** | **143.7 ± 1.7** | 1.14x | [10.19, 10.42] | 88  |
| **FD002** | **12.90 ± 0.04** | **543.0 ± 4.1** | 0.78x | [12.80, 12.97] | 275 |
| **FD003** | **10.69 ± 0.05** | **184.4 ± 2.8** | 0.93x | [10.54, 10.79] | 149 |
| **FD004** | **11.83 ± 0.03** | **724.5 ± 4.9** | 0.93x | [11.78, 11.89] | 157 |

---

## Seed 0 Reproduction

Verified in a fresh virtual environment on consumer hardware. All four datasets match `run.py` docstring expected values exactly.

| Dataset | RMSE | NASA | Gap | Features | Runtime | Hardware |
|---------|------|------|-----|----------|---------|----------|
| FD001 | 10.38 | 144 | 1.13x | 88  | 29 s  | M4 Mac Mini |
| FD002 | 12.89 | 543 | 0.78x | 275 | 92 s  | M4 Mac Mini |
| FD003 | 10.65 | 181 | 0.93x | 149 | 43 s  | M4 Mac Mini |
| FD004 | 11.80 | 723 | 0.93x | 157 | 61 s  | M4 Mac Mini |

---

## Leakage Audit

| Dataset | Max \|corr(F, RUL)\| | Max Feature | Features Audited | Status |
|---------|----------------------|-------------|------------------|--------|
| FD001 | 0.807 | F002 | 88 / 88 | OK |
| FD002 | 0.888 | F096 | 274 / 275 ¹ | OK |
| FD003 | 0.824 | F011 | 149 / 149 | OK |
| FD004 | 0.888 | F007 | 157 / 157 | OK |

¹ FD002 has one zero-variance feature; the correlation audit in `run.py` skips it to avoid a divide-by-zero in `np.corrcoef`. The feature is retained in the matrix for training; the reported max correlation is over the 274 features with non-zero variance.

All datasets fall well below the 0.95 threshold that would indicate direct RUL leakage. Max feature-target correlation across the full benchmark is 0.888 (FD002 F096).

---

## Error Tail Analysis — Seed 0

### Prediction Class Distribution

GOOD = \|error\| ≤ 10 cycles · EARLY = 10–20 early · VERY_EARLY = >20 early · LATE = 10–20 late · VERY_LATE = >20 late

| Dataset | Engines | GOOD | EARLY | VERY_EARLY | LATE | VERY_LATE |
|---------|---------|------|-------|------------|------|-----------|
| FD001 | 100 | 73 (73%) | 13 (13%) | 4 (4%) | 8 (8%) | 2 (2%) |
| FD002 | 259 | 165 (64%) | 44 (17%) | 31 (12%) | 12 (5%) | 7 (3%) |
| FD003 | 100 | 69 (69%) | 10 (10%) | 4 (4%) | 12 (12%) | 5 (5%) |
| FD004 | 248 | 179 (72%) | 16 (6%) | 8 (3%) | 28 (11%) | 17 (7%) |

### Late-Prediction Tail Statistics — Seed 0

| Dataset | P90 | P95 | Max Late | Max Early | >20 late | >40 late |
|---------|-----|-----|----------|-----------|----------|----------|
| FD001 | 15.9 | 22.2 | 25.6 | 35.0 | 2 | 0 |
| FD002 | 22.6 | 26.8 | 34.8 | 44.5 | 7 | 0 |
| FD003 | 18.9 | 22.2 | 35.8 | 24.8 | 5 | 0 |
| FD004 | 19.4 | 24.1 | 51.0 | 58.3 | 17 | 1 |

### Aggregate Across 30 Seeds

| Dataset | Max Late mean ± std | Max Late max | Total >40 late |
|---------|---------------------|--------------|----------------|
| FD001 | 25.9 ± 0.3 | 26 | **0 / 30 seeds** |
| FD002 | 34.9 ± 0.6 | 36 | **0 / 30 seeds** |
| FD003 | 36.0 ± 0.5 | 37 | **0 / 30 seeds** |
| FD004 | 50.9 ± 0.2 | 51 | 30 / 30 seeds ² |

² All thirty 30-seed runs flag the same single engine (`engine_166`, true RUL 72, predicted ~123). The remaining 247 FD004 engines report zero >40-late across all 30 seeds.

**Across 90 seed runs on FD001/FD002/FD003: zero catastrophic failures (>40 cycles late).** FD004 has one persistent catastrophic prediction: `engine_166` is a 2-fault-mode outlier that triggers the asymmetric NASA penalty on every configuration tested. It is the only engine in the C-MAPSS test set that consistently triggers catastrophic late prediction.

---

## Dataset Characteristics

| Dataset | Train Rows | Test Engines | Features | Op Conds | Fault Modes |
|---------|------------|--------------|----------|----------|-------------|
| FD001 | 20,631 | 100 | 88  | 1 | 1 (HPC) |
| FD002 | 53,759 | 259 | 275 | 6 | 1 (HPC) |
| FD003 | 24,720 | 100 | 149 | 1 | 2 (HPC + Fan) |
| FD004 | 61,249 | 248 | 157 | 6 | 2 (HPC + Fan) |

---

## Data Integrity

Parquets contain: `cohort` (engine-ID string), `F001..FNNN` (Float64), `RUL` (0–125 cycles).

Verified against raw NASA CMAPSSData CSVs:

- 0 F-columns are bitwise-equal to any raw column (`unit`, `cycle`, `op1–3`, `s1–21`).
- 0 F-columns are permutations of any raw column.
- Maximum correlation between any raw column and any F-feature: 0.18 (FD003).
- No raw passthrough features; all F-columns are engineered derivations from the NASA source data.

---

## Methodology

### Feature Engineering

Features were derived using Information-Preserving Embedding and Manifold Coherence Analysis (Rudder, 2026). The framework embeds multivariate sensor data into a high-dimensional geometric space constructed from the physical units of the signals, producing a diagnostic state description in which degradation becomes visible as geometric departure from a reference manifold. Column names are anonymised as F001–FNNN in the published parquets. The feature-engineering pipeline is not included in this repository and is covered by USPTO Application No. 64/027,466.

### Stacking Ensemble

Fixed hyperparameters. No tuning. Same architecture across all four datasets. Regularisation-critical parameters are listed below; the full signature is in `get_base_models()` within `run.py`.

| Component | Parameters |
|-----------|-----------|
| LightGBM | `n_estimators=500, max_depth=6, learning_rate=0.05, num_leaves=31, min_child_samples=10, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, n_jobs=1, deterministic=True` |
| XGBoost | `n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, min_child_weight=10, n_jobs=1` |
| HistGradientBoosting | `max_iter=500, max_depth=6, learning_rate=0.05, min_samples_leaf=10` |
| Meta-learner | `RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])` |

Meta-learner inputs are the per-fold out-of-fold (OOF) predictions of each base model; test predictions are averaged across CV folds per base model, then passed through the fitted `RidgeCV`. No explicit blending weight is tuned — `RidgeCV` selects its regularisation strength from the alpha grid via internal cross-validation.

### Validation

5-fold GroupKFold on engine ID — no engine appears in both train and validation within any fold. RUL capped at 125 cycles (C-MAPSS standard).

### Operating-Condition Correction

Multi-operating-condition datasets (FD002, FD004) use a physics-based operating-condition correction at data ingestion. The correction is derived from cohort manifold geometry and applied per-window, normalising sensor measurements to equivalent reference conditions before any ML processing. This removes operating-condition-induced variance while preserving degradation signal. Single-operating-condition datasets (FD001, FD003) require no normalisation.

### Scoring

- **RMSE**: Root-mean-square error on test-set predictions (last cycle per engine).
- **NASA PHM08**: Asymmetric penalty score. Late predictions (`predicted RUL > true RUL`) are penalised exponentially more heavily than early predictions, reflecting the safety-critical nature of prognostics.
- **Gap Ratio**: `OOF_RMSE / Test_RMSE`. The ideal baseline is 1.0; values below 1.0 indicate the model generalises to the held-out test set better than to the OOF folds, values above 1.0 the reverse. Healthy range is roughly 0.8–1.2. All four datasets fall within 0.78–1.14x.

---

## Reproducibility

Dependent on system compute architecture and dependencies, the same seed produces bit-identical results. Verified across multiple runs on the same hardware across all four datasets.

`run.py` sets determinism at every level before any computation begins:

- `PYTHONHASHSEED=0` — set in the **shell environment before
  interpreter launch**.
- `random.seed(seed)`
- `np.random.seed(seed)`
- `random_state=seed` on every base model: HistGradientBoosting,
  LightGBM (`n_jobs=1, deterministic=True`), XGBoost (`n_jobs=1`).

LightGBM and XGBoost require `n_jobs=1` for bit-identical results. Multi-threaded gradient boosting has known non-determinism due to floating-point accumulation order across threads; single-threaded execution removes this source of variance. Results reported here reflect single-threaded runs.

`PYTHONHASHSEED` must be set in the shell before invoking `python run.py`; setting it inside a running Python process has no effect, as the interpreter reads this variable at launch.

### Verified Environment

Python 3.12, numpy 2.x, polars 1.x, scikit-learn 1.x, xgboost 3.x,
lightgbm 4.x, pyarrow 23.x.

### Reproduction Commands

```bash
git clone https://github.com/orthon-io/cmapss.git
cd cmapss
pip install -r requirements.txt
PYTHONHASHSEED=0 python run.py --dataset fd001 --seed 0
PYTHONHASHSEED=0 python run.py --dataset fd001 --seeds 0-29
```

---

## Future Research

Active validation of the Ørthon framework is underway across multiple domains including aerospace, chemical processes, and rotating equipment. The current release covers C-MAPSS benchmark results only. Researchers interested in working on future projects are invited to reach out: <orthon@orthon.io> · Ørthon Research.

---

## License and Patent

**License:** Creative Commons Attribution-NonCommercial 4.0
International (CC BY-NC 4.0). Non-commercial academic use with attribution is permitted. Commercial and grant-funded institutional use requires a license agreement. 

**Patent Pending:** USPTO Application No. 64/027,466 — *System and Method for Information-Preserving Embedding and Manifold Coherence Analysis of Multivariate Signals*. The CC BY-NC 4.0 copyright license does not grant any rights under this patent application.

---

## Data Source

A. Saxena and K. Goebel (2008). *Turbofan Engine Degradation Simulation Data Set*. NASA Ames Prognostics Data Repository. <https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data>

Ramasso, E. & Saxena, A. (2014). *Performance Benchmarking and Analysis of Prognostic Methods for CMAPSS Datasets*. International Journal of Prognostics and Health Management, 5.
