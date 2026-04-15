# FD001 — Contractive Autoencoder as Feature Substitute

**Branch:** `experiment/fd001-cae`
**Date:** 2026-04-15
**Stack:** Python 3.12.12, torch 2.10.0, sklearn 1.8.0, xgboost 3.2.0,
          lightgbm 4.6.0, polars 1.39.3, numpy 2.4.3, M4 Mac Mini (CPU-only)
**Determinism:** `PYTHONHASHSEED=42`, `torch.use_deterministic_algorithms(True)`,
                 `torch.set_num_threads(1)`, `n_jobs=1`/`deterministic=True`
                 on LGB/XGB.

---

## TL;DR — FAIL on all pass criteria

| Metric | Canonical (public kit) | CAE substitute | Δ |
|---|---|---|---|
| **RMSE mean ± std** | 10.31 ± 0.06 | **14.08 ± 0.07** | **+37% worse** |
| **NASA mean ± std** | 143.7 ± 1.7 | **402.0 ± 7.2** | **+180% worse** |
| Max Late mean ± std | 25.9 ± 0.3 | 41.7 ± 0.6 | +61% worse |
| >40 late count (30 seeds) | 0 | **50** | catastrophic |
| Gap ratio | 1.14x | 1.08x | healthy |
| Seed-0 bit-identity | ✓ | ✓ | verified |

The 32-dim CAE latent, used as a drop-in replacement for the 88 ORTHON
features, produces a meaningfully worse stacking ensemble across every
error metric and a striking regression on catastrophic failures
(0 → 50 over 30 seeds).

---

## Pipeline

### Stage 1 — CAE (one-time training, seed 0)

```
Data:         FD001 Train observations.parquet
              15 health-state signals (T24, T30, T50, P15, P30, Ps30,
              phi, NRf, NRc, BPR, htBleed, Nf, Nc, W31, W32).
              Op-condition / metadata / setpoint signals excluded per
              C-MAPSS data contract.
Window:       size=30, stride=1, per-cohort, backward. Left-pad with
              first cycle for cycles < 29.
Healthy set:  first 20% of each cohort's cycles (label-free cycle-
              fraction rule; RUL never inspected at any point).
              → 4,086 training windows pooled across 100 engines.
Normalization: per-signal z-score fit on the 4,086 training-subset
               windows only.
Architecture: MLP, 450 → 128 → 64 → latent 32; decoder symmetric.
              Encoder and decoder both use ReLU between hidden layers.
Loss:         MSE_recon + 0.1 * ||∂h/∂x||_F^2
              Jacobian computed via autograd over all 32 latent dims
              per batch (32 backward passes per step).
Training:     Adam, lr=1e-3, batch 256, 300 epochs.
              CPU-only, 1 thread, torch.use_deterministic_algorithms(True).
              Seed 0.
Wall time:    39 seconds.
Final losses: recon 0.316, jacobian 0.019, total 0.318.
Weights:      cae/fd001_cae_weights.pt (86 KB).
Meta:         cae/fd001_cae_meta.json (config + mu/sigma + training log).
```

### Stage 2 — Encode all windows

```
Train windows: every (cohort, cycle) in FD001 Train observations
               → 20,631 rows. Row order: cohorts alphabetical,
               cycles ascending — exactly matching public kit's
               data/fd001/train.parquet row order.
Test windows:  one last-window per engine (window ending at each
               engine's max cycle) → 100 rows, matching public kit's
               data/fd001/test.parquet.
Output:        data/fd001_cae/{train,test}.parquet
               Schema: (cohort, L01..L32, RUL). RUL computed as
               clip(max_cycle - cycle, 0, 125) for train, inherited
               from public test.parquet (RUL_FD001.txt truth) for test.
```

### Stage 3 — Stacking ensemble, 30-seed sweep

```
Command: PYTHONHASHSEED=42 python run.py --dataset fd001_cae --seeds 0-29
Change:  feature filter generalized from `startswith("F")` to any
         non-grain numeric column — picks up L01..L32 automatically.
         No other modification to run.py.
CAE weights frozen; ensemble trains on latents as the sole feature
family. Same HistGB + LightGBM + XGBoost → RidgeCV stack, same
hyperparameters, same GroupKFold CV on engine ID.
```

---

## Full per-seed distribution (30 seeds)

```
seed   RMSE   NASA  gap  max_late  >40late
────────────────────────────────────────────
   0  14.06   397  1.08       41        2
   1  14.04   404  1.09       42        2
   2  14.09   398  1.08       42        1
   3  14.06   394  1.09       41        2
   4  14.05   394  1.08       41        2
   5  14.23   417  1.07       43        2
   6  14.11   400  1.08       41        2
   7  13.99   391  1.09       42        1
   8  14.12   409  1.08       42        2
   9  14.15   411  1.08       42        2
  10  14.14   408  1.08       41        2
  11  14.07   400  1.08       42        2
  12  14.03   396  1.09       42        1
  13  14.17   411  1.07       43        2
  14  14.10   404  1.08       42        2
  15  13.92   391  1.09       41        1
  16  14.19   413  1.07       43        2
  17  14.15   405  1.08       42        2
  18  13.99   397  1.09       41        1
  19  14.18   413  1.07       42        2
  20  14.11   412  1.08       42        1
  21  14.04   395  1.09       41        2
  22  14.05   398  1.09       41        2
  23  14.14   409  1.08       42        2
  24  14.06   399  1.08       42        2
  25  14.12   406  1.08       42        2
  26  14.05   398  1.08       41        2
  27  14.00   391  1.09       40        1
  28  14.06   402  1.09       43        1
  29  14.04   402  1.09       42        1
────────────────────────────────────────────
mean  14.08  402.0        41.7       50 total
std    0.07    7.2         0.6
```

---

## Error-tail comparison (seed 0, reproducible)

| Class | Canonical | CAE |
|---|---|---|
| GOOD (\|err\| ≤ 10) | 73% | 61% |
| EARLY (10–20 early) | 13% | 7% |
| VERY_EARLY (>20 early) | 4% | 6% |
| LATE (10–20 late) | 8% | 17% |
| VERY_LATE (>20 late) | 2% | 9% |
| P90 (\|err\|) | 15.9 | 23.6 |
| P95 (\|err\|) | 22.2 | 32.0 |
| Max late | 25.6 | 41.1 |
| Max early | 35.0 | 32.0 |
| >20 late | 2 | 9 |
| >40 late | 0 | 2 |

CAE nearly doubles P95 absolute error and pushes the LATE/VERY_LATE
bins from 10% combined to 26%. Half-errors shifted from the left tail
(early predictions, which NASA penalizes mildly) to the right tail
(late predictions, which NASA penalizes exponentially).

---

## Bit-identity verification

Same seed → same print-precision result across independent invocations:

```
run A, seed 7: RMSE=13.99 NASA=391 gap=1.09x feat=32 max_late=42 >40late=1
run B, seed 7: RMSE=13.99 NASA=391 gap=1.09x feat=32 max_late=42 >40late=1
```

Ensemble is deterministic under the fixed CAE encoder + fixed
stacking stack (`n_jobs=1, deterministic=True, PYTHONHASHSEED=42`).

---

## Leakage audit

`run.py` prints the maximum feature-target correlation pre-training.
Every seed reports:

```
Leakage check: max |corr(F, RUL)| = 0.841 (L16) -- OK
```

L16's absolute correlation with RUL is 0.841 — below the 0.95 warning
threshold. No leakage in the engineered sense. Note: the CAE was
trained without ever seeing RUL, and the "first 20% of cycles"
selection rule is purely based on cycle fraction.

---

## Interpretation (plain)

This is a clear negative result against the pass criteria. The CAE
latent replaces 88 ORTHON features with 32 latents and loses ~40% of
RMSE and ~180% of NASA. Three properties of the CAE + this
configuration that are plausibly responsible:

1. **Training on healthy data only** means the encoder is never
   exposed to degradation dynamics. It learns to map all inputs onto
   the manifold of healthy states, flattening degradation-relevant
   structure.
2. **Contractive penalty** (λ=0.1) specifically rewards insensitivity
   to local input perturbations, which is the opposite of what a
   degradation feature needs (sharp response to state drift).
3. **The public kit's 88 ORTHON features** include hand-engineered
   quantities like RT centroid distance, Mahalanobis-in-PC-subspace,
   and per-cohort departure metrics that are explicitly contrastive
   with the healthy baseline. The CAE latent is a generic compression
   of the local window; it has no corresponding contrastive signal
   unless you use `recon_error` as a feature (which we did not,
   per the spec).

None of this makes the result wrong — it makes it an expected
negative. The substitution test was fair: same stacking, same seeds,
same grain, label-free pretraining. The answer is that a contractive
MLP over raw windows is not a substitute for the public-repo feature
set at this capacity.

---

## Artifacts on this branch

```
cae/model.py               — SmallContractiveAutoencoder + Jacobian helper
cae/data.py                — windowed dataset builder (HEALTH_SIGNALS, zscore)
cae/train.py               — training driver
cae/encode.py              — encoding driver (emits data/fd001_cae/*.parquet)
cae/fd001_cae_weights.pt   — trained weights
cae/fd001_cae_meta.json    — config + mu/sigma + loss log
data/fd001_cae/train.parquet  (20,631 × 34: cohort + L01..L32 + RUL)
data/fd001_cae/test.parquet   (100     × 34)
experiments/fd001_cae/RESULTS.md    — this file
run.py                     — minimal edits: generic feature filter,
                             fd001_cae added as dataset choice
```

No change to `data/fd001/*.parquet`, no change to canonical notebooks
or canonical README. Main result (FD001 10.31 ± 0.06) is untouched.
