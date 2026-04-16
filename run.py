"""
Prognostics — Reproduce results from pre-computed feature matrices.

Features are post-imputation, post-scaling. No assembly. No imputation.
No scaling. Just load, train, predict.

Usage:
    python run.py --dataset fd001 --seed 0
    python run.py --dataset fd001 --seeds 0-29

Expected results (deterministic, seed-dependent):
    FD001: RMSE 10.38, NASA 144  (seed 0)
    FD002: RMSE 12.89, NASA 543  (seed 0)
    FD003: RMSE 10.65, NASA 181  (seed 0)
    FD004: RMSE 11.80, NASA 723  (seed 0)
"""

import argparse
import os
import random
import time
from pathlib import Path

# Capture the shell-level PYTHONHASHSEED state BEFORE we overwrite it
# below, so main() can warn the user if they forgot to set it. The
# sys.flags path doesn't work here — it only distinguishes
# PYTHONHASHSEED=0 from "anything else", not "unset" from "deterministic".
_SHELL_PYTHONHASHSEED_SET = os.environ.get("PYTHONHASHSEED") is not None

# PYTHONHASHSEED must be set in the shell before launch — os.environ
# assignment here is a no-op (Python reads PYTHONHASHSEED at interpreter
# startup, before this line runs). Kept for intent and to remind callers
# to invoke as `PYTHONHASHSEED=0 python run.py ...`. PYTHONHASHSEED=0
# disables Python's hash randomization entirely, giving consistent
# dict/set iteration order across runs. Numerical determinism of the
# stacking ensemble is enforced separately via n_jobs=1 and
# deterministic=True inside get_base_models().
os.environ["PYTHONHASHSEED"] = "0"

import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

RUL_CAP = 125.0
N_SPLITS = 5


def nasa_score(y_true, y_pred):
    errors = y_pred - y_true
    return float(sum(
        np.exp(-e / 13) - 1 if e < 0 else np.exp(e / 10) - 1
        for e in errors
    ))


def compute_error_tails(y_true, y_pred):
    """Compute error tail statistics for prognostics evaluation."""
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    n = len(errors)

    # Failure class distribution
    classes = {
        "GOOD": int(np.sum(abs_errors <= 10)),
        "EARLY": int(np.sum((errors > -20) & (errors < -10))),
        "VERY_EARLY": int(np.sum(errors <= -20)),
        "LATE": int(np.sum((errors > 10) & (errors < 20))),
        "VERY_LATE": int(np.sum(errors >= 20)),
    }

    return {
        "classes": classes,
        "max_late": float(np.max(errors)) if len(errors) > 0 else 0.0,
        "max_early": float(abs(np.min(errors))) if len(errors) > 0 else 0.0,
        "p50": float(np.percentile(abs_errors, 50)),
        "p75": float(np.percentile(abs_errors, 75)),
        "p90": float(np.percentile(abs_errors, 90)),
        "p95": float(np.percentile(abs_errors, 95)),
        "gt20_late": int(np.sum(errors > 20)),
        "gt40_late": int(np.sum(errors > 40)),
        "gt20_total": int(np.sum(abs_errors > 20)),
    }


def print_error_tails(y_true, y_pred, dataset):
    """Print formatted error tail report."""
    et = compute_error_tails(y_true, y_pred)
    n = len(y_true)
    print(f"  Error Tails ({n} engines):")
    cls = et["classes"]
    print(f"    GOOD       {cls['GOOD']:4d} ({cls['GOOD']/n*100:.0f}%)")
    print(f"    EARLY      {cls['EARLY']:4d} ({cls['EARLY']/n*100:.0f}%)")
    print(f"    VERY_EARLY {cls['VERY_EARLY']:4d} ({cls['VERY_EARLY']/n*100:.0f}%)")
    print(f"    LATE       {cls['LATE']:4d} ({cls['LATE']/n*100:.0f}%)")
    print(f"    VERY_LATE  {cls['VERY_LATE']:4d} ({cls['VERY_LATE']/n*100:.0f}%)")
    print(f"    Max late:  {et['max_late']:.1f}  Max early: {et['max_early']:.1f}")
    print(f"    P90:       {et['p90']:.1f}  P95: {et['p95']:.1f}")
    print(f"    >20 late:  {et['gt20_late']}  >40 late: {et['gt40_late']}")


def get_base_models(seed=0):
    models = {
        "hist": HistGradientBoostingRegressor(
            max_iter=500, max_depth=6, learning_rate=0.05,
            min_samples_leaf=10, random_state=seed,
        ),
    }
    if HAS_LGB:
        models["lgb"] = LGBMRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            num_leaves=31, min_child_samples=10, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            verbose=-1, random_state=seed,
            n_jobs=1, deterministic=True,
        )
    if HAS_XGB:
        models["xgb"] = XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
            reg_lambda=1.0, min_child_weight=10, verbosity=0,
            random_state=seed,
            n_jobs=1,
        )
    return models


def run(dataset: str, seed: int = 0, data_dir: Path = None, verbose: bool = True):
    if data_dir is None:
        data_dir = Path(__file__).parent / "data" / dataset

    if not data_dir.exists():
        print(f"ERROR: {data_dir} not found")
        return None

    # Full determinism — set all seeds
    random.seed(seed)
    np.random.seed(seed)

    t0 = time.time()

    train = pl.read_parquet(data_dir / "train.parquet")
    test = pl.read_parquet(data_dir / "test.parquet")

    # Feature columns are all non-grain numeric columns in the parquet.
    grain_cols = {"cohort", "RUL"}
    feature_cols = [c for c in train.columns if c not in grain_cols]
    # Preserve column order from parquet (already in correct order)

    X_train = train.select(feature_cols).to_numpy().astype(np.float64)
    y_train = np.clip(train["RUL"].to_numpy().astype(np.float64), 0, RUL_CAP)
    groups = train["cohort"].to_numpy()

    X_test = test.select(feature_cols).to_numpy().astype(np.float64)
    y_test = test["RUL"].to_numpy().astype(np.float64)

    # Leakage audit — skip zero-variance columns (they carry no information
    # and would otherwise produce divide-by-zero warnings inside np.corrcoef).
    variances = np.var(X_train, axis=0)
    valid_cols = variances > 0
    n_skipped = int((~valid_cols).sum())
    corrs = np.zeros(X_train.shape[1])
    valid_idx = np.where(valid_cols)[0]
    for i in valid_idx:
        corrs[i] = abs(np.corrcoef(X_train[:, i], y_train)[0, 1])
    corrs = np.nan_to_num(corrs, nan=0.0)
    max_corr = float(np.max(corrs))
    max_feat = feature_cols[int(np.argmax(corrs))]
    skip_note = f" (skipped {n_skipped} constant feature{'s' if n_skipped != 1 else ''})" if n_skipped else ""
    print(f"  Leakage check: max |corr(F, RUL)| = {max_corr:.3f} ({max_feat}){skip_note} "
          f"{'-- OK' if max_corr < 0.95 else '-- WARNING'}")

    # Train stacking ensemble
    base_models = get_base_models(seed)
    model_names = list(base_models.keys())
    gkf = GroupKFold(n_splits=N_SPLITS)
    n_train = len(X_train)
    n_test = len(X_test)

    oof = {name: np.zeros(n_train) for name in model_names}
    test_preds = {name: np.zeros(n_test) for name in model_names}

    for train_idx, val_idx in gkf.split(X_train, y_train, groups):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr = y_train[train_idx]
        for name, model in base_models.items():
            clone = type(model)(**model.get_params())
            clone.fit(X_tr, y_tr)
            oof[name][val_idx] = clone.predict(X_val)
            test_preds[name] += clone.predict(X_test) / N_SPLITS

    oof_stack = np.column_stack([oof[n] for n in model_names])
    test_stack = np.column_stack([test_preds[n] for n in model_names])

    meta = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    meta.fit(oof_stack, y_train)
    y_pred = np.clip(meta.predict(test_stack), 0, RUL_CAP)
    oof_pred = meta.predict(oof_stack)

    oof_rmse = float(np.sqrt(np.mean((oof_pred - y_train) ** 2)))
    rmse = float(np.sqrt(np.mean((y_pred - y_test) ** 2)))
    nasa = nasa_score(y_test, y_pred)
    gap = oof_rmse / rmse if rmse > 0 else float("inf")

    et = compute_error_tails(y_test, y_pred)

    if verbose:
        print_error_tails(y_test, y_pred, dataset)

    elapsed = time.time() - t0
    return {
        "dataset": dataset.upper(), "seed": seed,
        "rmse": round(rmse, 2), "nasa": round(nasa, 1),
        "oof_rmse": round(oof_rmse, 2), "gap": round(gap, 2),
        "n_features": len(feature_cols), "elapsed_s": round(elapsed, 1),
        "error_tails": et,
    }


def main():
    # Warn if PYTHONHASHSEED wasn't set in the shell before launch.
    # The module-level os.environ assignment up top is a no-op for
    # Python's hash randomization (already seeded at interpreter
    # startup), but it poisons any os.environ.get() check here — so
    # we consult _SHELL_PYTHONHASHSEED_SET, captured before the
    # assignment.
    if not _SHELL_PYTHONHASHSEED_SET:
        print("WARNING: PYTHONHASHSEED is not set. Results may not be "
              "100% reproducible.\n"
              "         For bit-identity, invoke as:\n"
              "           PYTHONHASHSEED=0 python run.py --dataset ...")

    parser = argparse.ArgumentParser(description="Reproduce C-MAPSS RUL predictions")
    parser.add_argument("--dataset", required=True,
                        choices=["fd001", "fd002", "fd003", "fd004"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.seeds:
        start, end = args.seeds.split("-")
        seeds = list(range(int(start), int(end) + 1))
    else:
        seeds = [args.seed]

    sweep_mode = len(seeds) > 1
    results = []
    for seed in seeds:
        r = run(args.dataset, seed=seed, data_dir=args.data_dir, verbose=not sweep_mode)
        if r is None:
            continue
        results.append(r)
        et = r["error_tails"]
        print(f"{r['dataset']} seed={r['seed']:2d}: "
              f"RMSE={r['rmse']:.2f} NASA={r['nasa']:.0f} "
              f"gap={r['gap']:.2f}x feat={r['n_features']} "
              f"max_late={et['max_late']:.0f} >40late={et['gt40_late']} "
              f"({r['elapsed_s']:.0f}s)")

    if len(results) > 1:
        import statistics
        rmses = [r["rmse"] for r in results]
        nasas = [r["nasa"] for r in results]
        gaps = [r["gap"] for r in results]
        max_lates = [r["error_tails"]["max_late"] for r in results]
        gt40s = [r["error_tails"]["gt40_late"] for r in results]
        print(f"\n{results[0]['dataset']} {len(results)}-seed summary:")
        print(f"  RMSE:     {statistics.mean(rmses):.2f} +/- {statistics.stdev(rmses):.2f}  [{min(rmses):.2f}, {max(rmses):.2f}]")
        print(f"  NASA:     {statistics.mean(nasas):.1f} +/- {statistics.stdev(nasas):.1f}  [{min(nasas):.0f}, {max(nasas):.0f}]")
        print(f"  Gap:      {statistics.mean(gaps):.2f}x")
        print(f"  Max late: {statistics.mean(max_lates):.1f} +/- {statistics.stdev(max_lates):.1f}")
        print(f"  >40 late: {sum(gt40s)} total across {len(results)} seeds")


if __name__ == "__main__":
    main()
