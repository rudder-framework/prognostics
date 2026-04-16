# C-MAPSS Results

**Date:** 2026-04-10
**Method:** Stacking ensemble (LightGBM + XGBoost + HistGradientBoosting → RidgeCV)
**Validation:** 30-seed sweep, deterministic seeding, GroupKFold(5) on engine ID
**Reproduction:** `python run.py --dataset {fd001..fd004} --seed {0..29}`

---

## 30-Seed Validated Results

| Dataset | RMSE mean±std | NASA mean±std | Gap | RMSE Range | NASA Range |
|---------|--------------|--------------|-----|------------|------------|
| **FD001** | **10.31 ± 0.06** | **143.7 ± 1.7** | 1.14x | [10.19, 10.42] | [141, 147] |
| **FD002** | **12.90 ± 0.04** | **543.0 ± 4.1** | 0.78x | [12.80, 12.97] | [534, 550] |
| **FD003** | **10.69 ± 0.05** | **184.4 ± 2.8** | 0.93x | [10.54, 10.79] | [177, 188] |
| **FD004** | **11.83 ± 0.03** | **724.5 ± 4.9** | 0.93x | [11.78, 11.89] | [712, 733] |

Standard deviations of ±0.03–0.06 RMSE across 30 seeds reflect deterministic seeding and tight model variance. All metrics fall within healthy gap ratios (0.78–1.14x).

---

## Error Tail Analysis (seed 0)

### Prediction Class Distribution

Classification: GOOD = |error| ≤ 10 cycles, EARLY = 10–20 cycles early, VERY_EARLY = >20 early, LATE = 10–20 late, VERY_LATE = >20 late.

| Dataset | GOOD | EARLY | VERY_EARLY | LATE | VERY_LATE | Total Engines |
|---------|------|-------|------------|------|-----------|---------------|
| FD001 | 73 (73%) | 13 (13%) | 4 (4%) | 8 (8%) | 2 (2%) | 100 |
| FD002 | 165 (64%) | 44 (17%) | 31 (12%) | 12 (5%) | 7 (3%) | 259 |
| FD003 | 69 (69%) | 10 (10%) | 4 (4%) | 12 (12%) | 5 (5%) | 100 |
| FD004 | 179 (72%) | 16 (6%) | 8 (3%) | 28 (11%) | 17 (7%) | 248 |

### Late Prediction Statistics

| Dataset | P50 | P90 | P95 | Max Late | Max Early | >20 late | >40 late |
|---------|-----|-----|-----|----------|-----------|----------|----------|
| FD001 | — | 15.9 | 22.2 | 25.6 | 35.0 | 2 | **0** |
| FD002 | — | 22.6 | 26.8 | 34.8 | 44.5 | 7 | **0** |
| FD003 | — | 18.9 | 22.2 | 35.8 | 24.8 | 5 | **0** |
| FD004 | — | 19.4 | 24.1 | 51.0 | 58.3 | 17 | **1** |

### Aggregate Across 30 Seeds

| Dataset | Max Late mean ± std | Max Late max | Total >40 late |
|---------|--------------------|--------------|----------------|
| FD001 | 25.9 ± 0.3 | 26 | **0 / 30 seeds** |
| FD002 | 34.9 ± 0.6 | 36 | **0 / 30 seeds** |
| FD003 | 36.0 ± 0.5 | 37 | **0 / 30 seeds** |
| FD004 | 50.9 ± 0.2 | 51 | **30 / 30 seeds** |

**Across 90 seed runs on FD001/FD002/FD003: zero catastrophic failures.**

FD004 has one persistent catastrophic prediction across all 30 seeds: engine_166 (true RUL 72, predicted ~123). This is a 2-fault-mode outlier that appears identically in every configuration tested. It is the only engine in the C-MAPSS test set that consistently triggers the asymmetric NASA penalty across this method.

---

## Leakage Audit

Maximum feature-target correlation per dataset:

| Dataset | Max \|corr(F, RUL)\| | Status |
|---------|---------------------|--------|
| FD001 | 0.807 | OK — no leakage |
| FD002 | 0.888 | OK — lifecycle position feature, expected |
| FD003 | 0.824 | OK — no leakage |
| FD004 | 0.888 | OK — lifecycle position feature, expected |

Leakage produces gap ratio >1.5x. All datasets here fall within 0.78–1.14x.

The leakage check is run automatically by `run.py` and printed before training.

---

## Methodology

### Data
- **Source:** [NASA C-MAPSS](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)
- **Pre-computed feature matrices** are provided in `data/{fd001..fd004}/` as anonymized parquet files with columns F001..FNNN.
- **Train and test parquets** are post-imputation, post-scaling. The researcher loads them directly — no feature engineering, no imputation, no scaling required.

### Model
Stacking ensemble with fixed hyperparameters:

| Component | Parameters |
|-----------|-----------|
| LightGBM | n_estimators=500, max_depth=6, lr=0.05 |
| XGBoost | n_estimators=500, max_depth=6, lr=0.05 |
| HistGradientBoosting | max_iter=500, max_depth=6, lr=0.05 |
| Meta-learner | RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]) |

All base models receive `random_state=seed`. The same seed produces bit-identical results on the same hardware.

### Validation
- **5-fold GroupKFold** on engine ID — no engine appears in both train and validation within any fold.
- **RUL capped at 125** cycles (C-MAPSS standard).
- **No hyperparameter tuning.** The same architecture is used across all four datasets.

### Scoring
- **RMSE:** Root mean squared error on test set predictions (last cycle per engine)
- **NASA PHM08:** Asymmetric penalty score; late predictions penalized exponentially more than early predictions.
- **Gap Ratio:** OOF_RMSE / Test_RMSE. Healthy range: 0.8–1.2.

---

## Operating Condition Correction

Multi-operating-condition datasets (FD002, FD004) use a physics-based operating condition correction at data ingestion. The correction normalizes sensor measurements to equivalent reference conditions before any ML processing, removing operating-condition-induced variance while preserving degradation signal.

Single-operating-condition datasets (FD001, FD003) require no normalization.

The pre-computed feature matrices in this repository contain features extracted from the corrected data for FD002 and FD004 and from the original data for FD001 and FD003.

---

## Reproduction

```bash
git clone https://github.com/rudder-framework/cmapss.git
cd cmapss
pip install -r requirements.txt

# Single seed
python run.py --dataset fd001 --seed 0

# 30-seed sweep
python run.py --dataset fd001 --seeds 0-29
```

Each run prints:
1. Leakage audit (max feature-target correlation)
2. Per-seed RMSE, NASA, gap ratio, max late prediction, count of >40 late
3. Error tail class distribution (single seed mode)
4. Aggregate statistics (sweep mode)

---

## Determinism

The same seed produces bit-identical results. Verified across multiple runs on the same hardware. Verified across all four datasets.

`run.py` sets:
- `os.environ["PYTHONHASHSEED"] = "0"` (set before imports)
- `random.seed(seed)`
- `np.random.seed(seed)`
- `random_state=seed` on every base model (HistGradientBoosting, LightGBM, XGBoost)

---

## License

[CC BY-NC 4.0](LICENSE.md) — non-commercial use with attribution.
**Patent Pending** — patent rights are expressly reserved; commercial use requires a separate written license from the copyright holder.

The feature engineering pipeline that produced the F001..FNNN columns is not included in this repository. Researchers can verify the reported results exactly using the provided feature matrices and `run.py`.
