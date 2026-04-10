"""
Prognostics — Reproduce results from pre-computed anonymized features.

Supports any dataset exported through the anonymization pipeline.
Automatically detects regression (RUL) vs classification (fault_label).

Usage:
    python run.py --dataset fd001 --seed 42
    python run.py --dataset fd001 --seeds 0-29
    python run.py --dataset tep --seed 42
    python run.py --dataset bearing --seed 42

The data/ directory contains one subdirectory per dataset:
    data/{name}/train.parquet   — columns: cohort, F001..FNNN, RUL or fault_label
    data/{name}/test.parquet    — columns: cohort, F001..FNNN, RUL or fault_label
"""

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

RUL_CAP = 125.0
N_SPLITS = 5


# ── Scoring ──────────────────────────────────────────────────────────────

def nasa_score(y_true, y_pred):
    errors = y_pred - y_true
    return float(sum(
        np.exp(-e / 13) - 1 if e < 0 else np.exp(e / 10) - 1
        for e in errors
    ))


# ── Models ───────────────────────────────────────────────────────────────

def get_regression_models(seed=42):
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


def get_classification_models(seed=42, n_classes=2):
    models = {
        "hist": HistGradientBoostingClassifier(
            max_iter=500, max_depth=6, learning_rate=0.05,
            min_samples_leaf=10, random_state=seed,
        ),
    }
    if HAS_LGB:
        models["lgb"] = LGBMClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            num_leaves=31, min_child_samples=10, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            verbose=-1, random_state=seed,
        )
    if HAS_XGB:
        models["xgb"] = XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
            reg_lambda=1.0, min_child_weight=10, verbosity=0,
            random_state=seed, use_label_encoder=False,
            eval_metric="mlogloss" if n_classes > 2 else "logloss",
        )
    return models


# ── Run ──────────────────────────────────────────────────────────────────

def run(dataset: str, seed: int = 42, data_dir: Path = None):
    if data_dir is None:
        data_dir = Path(__file__).parent / "data" / dataset

    if not data_dir.exists():
        print(f"ERROR: {data_dir} not found")
        return None

    t0 = time.time()

    train = pl.read_parquet(data_dir / "train.parquet")
    test = pl.read_parquet(data_dir / "test.parquet")

    feature_cols = sorted([c for c in train.columns if c.startswith("F")])

    # Detect task type
    if "RUL" in train.columns:
        task = "regression"
        target_col = "RUL"
    elif "fault_label" in train.columns:
        task = "classification"
        target_col = "fault_label"
    else:
        target_cols = [c for c in train.columns if c not in {"cohort"} and not c.startswith("F")]
        if target_cols:
            target_col = target_cols[0]
            task = "classification" if train[target_col].n_unique() < 50 else "regression"
        else:
            print("ERROR: no target column found")
            return None

    X_train = train.select(feature_cols).to_numpy().astype(np.float64)
    groups = train["cohort"].to_numpy()
    X_test = test.select(feature_cols).to_numpy().astype(np.float64)

    X_train = np.where(np.isinf(X_train), np.nan, X_train)
    X_test = np.where(np.isinf(X_test), np.nan, X_test)

    # Leakage audit — report max feature-target correlation
    if task == "regression":
        y_check = np.clip(train[target_col].to_numpy().astype(np.float64), 0, RUL_CAP)
        imp_check = SimpleImputer(strategy="median")
        X_check = imp_check.fit_transform(X_train)
        corrs = np.array([abs(np.corrcoef(X_check[:, i], y_check)[0, 1])
                          for i in range(X_check.shape[1])])
        corrs = np.nan_to_num(corrs, nan=0.0)
        max_corr = float(np.max(corrs))
        max_feat = feature_cols[int(np.argmax(corrs))]
        print(f"  Leakage check: max |corr(F, {target_col})| = {max_corr:.3f} ({max_feat})"
              f" {'— OK' if max_corr < 0.95 else '— WARNING: possible leakage'}")

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if task == "regression":
        y_train = np.clip(train[target_col].to_numpy().astype(np.float64), 0, RUL_CAP)
        y_test = test[target_col].to_numpy().astype(np.float64)
        return _run_regression(dataset, X_train, y_train, X_test, y_test,
                               groups, feature_cols, seed, t0)
    else:
        y_train = train[target_col].to_numpy()
        y_test = test[target_col].to_numpy()
        return _run_classification(dataset, X_train, y_train, X_test, y_test,
                                   groups, feature_cols, seed, t0)


def _run_regression(dataset, X_train, y_train, X_test, y_test,
                    groups, feature_cols, seed, t0):
    base_models = get_regression_models(seed)
    model_names = list(base_models.keys())
    gkf = GroupKFold(n_splits=N_SPLITS)

    oof = {name: np.zeros(len(X_train)) for name in model_names}
    test_preds = {name: np.zeros(len(X_test)) for name in model_names}

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr = y_train[train_idx]
        for name, model in base_models.items():
            clone = type(model)(**model.get_params())
            clone.fit(X_tr, y_tr)
            oof[name][val_idx] = clone.predict(X_val)
            test_preds[name] += clone.predict(X_test) / N_SPLITS

    for name, model in base_models.items():
        model.fit(X_train, y_train)

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

    return {
        "dataset": dataset.upper(), "task": "regression", "seed": seed,
        "rmse": round(rmse, 2), "nasa": round(nasa, 1),
        "oof_rmse": round(oof_rmse, 2), "gap": round(gap, 2),
        "n_features": len(feature_cols), "elapsed_s": round(time.time() - t0, 1),
    }


def _run_classification(dataset, X_train, y_train, X_test, y_test,
                        groups, feature_cols, seed, t0):
    n_classes = len(set(y_train))
    base_models = get_classification_models(seed, n_classes)
    model_names = list(base_models.keys())
    gkf = GroupKFold(n_splits=N_SPLITS)

    oof = {name: np.zeros(len(X_train), dtype=y_train.dtype) for name in model_names}
    test_preds = {name: [] for name in model_names}

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr = y_train[train_idx]
        for name, model in base_models.items():
            clone = type(model)(**model.get_params())
            clone.fit(X_tr, y_tr)
            oof[name][val_idx] = clone.predict(X_val)
            test_preds[name].append(clone.predict(X_test))

    # Majority vote for test predictions
    final_test_preds = {}
    for name in model_names:
        preds_array = np.array(test_preds[name])
        from scipy.stats import mode
        final_test_preds[name], _ = mode(preds_array, axis=0)
        final_test_preds[name] = final_test_preds[name].flatten()

    # Simple majority vote across models for final prediction
    all_preds = np.column_stack([final_test_preds[n] for n in model_names])
    y_pred, _ = mode(all_preds, axis=1)
    y_pred = y_pred.flatten()

    oof_preds_all = np.column_stack([oof[n] for n in model_names])
    oof_final, _ = mode(oof_preds_all, axis=1)
    oof_final = oof_final.flatten()

    f1 = f1_score(y_test, y_pred, average="macro")
    acc = accuracy_score(y_test, y_pred)
    oof_f1 = f1_score(y_train, oof_final, average="macro")

    return {
        "dataset": dataset.upper(), "task": "classification", "seed": seed,
        "f1_macro": round(f1, 4), "accuracy": round(acc, 4),
        "oof_f1": round(oof_f1, 4),
        "n_classes": n_classes, "n_features": len(feature_cols),
        "elapsed_s": round(time.time() - t0, 1),
    }


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Reproduce results from anonymized features")
    parser.add_argument("--dataset", required=True,
                        help="Dataset name (subdirectory of data/)")
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
        if r is None:
            continue
        results.append(r)

        if r["task"] == "regression":
            print(f"{r['dataset']} seed={r['seed']:2d}: "
                  f"RMSE={r['rmse']:.2f} NASA={r['nasa']:.0f} "
                  f"gap={r['gap']:.2f}x features={r['n_features']} "
                  f"({r['elapsed_s']:.0f}s)")
        else:
            print(f"{r['dataset']} seed={r['seed']:2d}: "
                  f"F1={r['f1_macro']:.4f} acc={r['accuracy']:.4f} "
                  f"classes={r['n_classes']} features={r['n_features']} "
                  f"({r['elapsed_s']:.0f}s)")

    if len(results) > 1:
        import statistics
        if results[0]["task"] == "regression":
            rmses = [r["rmse"] for r in results]
            nasas = [r["nasa"] for r in results]
            print(f"\n{results[0]['dataset']} {len(results)}-seed summary:")
            print(f"  RMSE: {statistics.mean(rmses):.2f} +/- {statistics.stdev(rmses):.2f}")
            print(f"  NASA: {statistics.mean(nasas):.1f} +/- {statistics.stdev(nasas):.1f}")
        else:
            f1s = [r["f1_macro"] for r in results]
            print(f"\n{results[0]['dataset']} {len(results)}-seed summary:")
            print(f"  F1:   {statistics.mean(f1s):.4f} +/- {statistics.stdev(f1s):.4f}")


if __name__ == "__main__":
    main()
