"""
Prognostics — Reproduce C-MAPSS RUL prediction results.

Pre-computed feature matrices with anonymized column names.
No feature engineering code needed — just load, train, score.

Usage:
    python run.py --dataset fd001 --seed 42
    python run.py --dataset fd001 --seeds 0-29

Expected results (seed=42, deterministic):
    FD001: RMSE 10.45, NASA 147,  88 features
    FD002: RMSE 12.70, NASA 550, 275 features
    FD003: RMSE 10.74, NASA 179, 149 features
    FD004: RMSE 11.98, NASA 746, 157 features
"""

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

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


def get_base_models(seed=42):
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
        )
    if HAS_XGB:
        models["xgb"] = XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
            reg_lambda=1.0, min_child_weight=10, verbosity=0,
            random_state=seed,
        )
    return models


def run(dataset: str, seed: int = 42, data_dir: Path = None):
    if data_dir is None:
        data_dir = Path(__file__).parent / "data" / dataset

    t0 = time.time()

    # Load pre-computed feature matrices
    train = pl.read_parquet(data_dir / "train.parquet")
    test = pl.read_parquet(data_dir / "test.parquet")

    # Separate features, target, groups
    feature_cols = sorted([c for c in train.columns if c.startswith("F")])
    X_train = train.select(feature_cols).to_numpy().astype(np.float64)
    y_train = np.clip(train["RUL"].to_numpy().astype(np.float64), 0, RUL_CAP)
    groups = train["cohort"].to_numpy()

    X_test = test.select(feature_cols).to_numpy().astype(np.float64)
    y_test = test["RUL"].to_numpy().astype(np.float64)

    # Clean
    X_train = np.where(np.isinf(X_train), np.nan, X_train)
    X_test = np.where(np.isinf(X_test), np.nan, X_test)

    # Impute + scale
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train stacking ensemble with GroupKFold
    base_models = get_base_models(seed)
    model_names = list(base_models.keys())
    gkf = GroupKFold(n_splits=N_SPLITS)
    n_train = len(X_train)
    n_test = len(X_test)

    oof = {name: np.zeros(n_train) for name in model_names}
    test_preds = {name: np.zeros(n_test) for name in model_names}

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr = y_train[train_idx]

        for name, model in base_models.items():
            clone = type(model)(**model.get_params())
            clone.fit(X_tr, y_tr)
            oof[name][val_idx] = clone.predict(X_val)
            test_preds[name] += clone.predict(X_test) / N_SPLITS

    # Fit all base models on full training data
    for name, model in base_models.items():
        model.fit(X_train, y_train)

    # Meta-learner
    oof_stack = np.column_stack([oof[n] for n in model_names])
    test_stack = np.column_stack([test_preds[n] for n in model_names])

    meta = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    meta.fit(oof_stack, y_train)

    y_pred = np.clip(meta.predict(test_stack), 0, RUL_CAP)
    oof_pred = meta.predict(oof_stack)

    # Metrics
    oof_rmse = float(np.sqrt(np.mean((oof_pred - y_train) ** 2)))
    rmse = float(np.sqrt(np.mean((y_pred - y_test) ** 2)))
    nasa = nasa_score(y_test, y_pred)
    gap = oof_rmse / rmse if rmse > 0 else float("inf")

    elapsed = time.time() - t0
    return {
        "dataset": dataset.upper(),
        "seed": seed,
        "rmse": round(rmse, 2),
        "nasa": round(nasa, 1),
        "oof_rmse": round(oof_rmse, 2),
        "gap": round(gap, 2),
        "n_features": len(feature_cols),
        "elapsed_s": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Reproduce C-MAPSS RUL predictions")
    parser.add_argument("--dataset", required=True,
                        choices=["fd001", "fd002", "fd003", "fd004"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default=None,
                        help="Range like 0-29 for sweep")
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.seeds:
        start, end = args.seeds.split("-")
        seeds = list(range(int(start), int(end) + 1))
    else:
        seeds = [args.seed]

    results = []
    for seed in seeds:
        r = run(args.dataset, seed=seed, data_dir=args.data_dir)
        results.append(r)
        print(f"{r['dataset']} seed={r['seed']:2d}: "
              f"RMSE={r['rmse']:.2f} NASA={r['nasa']:.0f} "
              f"gap={r['gap']:.2f}x features={r['n_features']} "
              f"({r['elapsed_s']:.0f}s)")

    if len(results) > 1:
        import statistics
        rmses = [r["rmse"] for r in results]
        nasas = [r["nasa"] for r in results]
        print(f"\n{results[0]['dataset']} {len(results)}-seed summary:")
        print(f"  RMSE: {statistics.mean(rmses):.2f} +/- {statistics.stdev(rmses):.2f}")
        print(f"  NASA: {statistics.mean(nasas):.1f} +/- {statistics.stdev(nasas):.1f}")


if __name__ == "__main__":
    main()
