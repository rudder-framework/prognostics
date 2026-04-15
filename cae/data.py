"""FD001 windowed dataset builder — long-format observations to (window, signal) matrices."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


# The 15 health-state signals per the C-MAPSS data contract (~/prime/CLAUDE.md).
# Excludes op1/op2/op3 (operating conditions), Nf_dmd/PCNfR_dmd (demand setpoints),
# P2/T2/epr/farB (metadata/derived).
HEALTH_SIGNALS: tuple[str, ...] = (
    "T24", "T30", "T50",
    "P15", "P30", "Ps30",
    "phi",
    "NRf", "NRc",
    "BPR",
    "htBleed",
    "Nf", "Nc",
    "W31", "W32",
)

WINDOW_SIZE = 30
EARLY_PCT   = 0.20


@dataclass
class WindowedData:
    X: np.ndarray            # (n_rows, WINDOW_SIZE * N_SIGNALS) float32
    cohort: np.ndarray       # (n_rows,) object
    cycle: np.ndarray        # (n_rows,) int64  — the LAST cycle in the window
    is_training_subset: np.ndarray  # (n_rows,) bool — first EARLY_PCT per cohort
    mu: np.ndarray           # (N_SIGNALS,) per-signal mean fit on training subset
    sigma: np.ndarray        # (N_SIGNALS,) per-signal std fit on training subset
    n_signals: int
    window_size: int
    signal_order: tuple[str, ...]


def _wide_by_signal(obs: pl.DataFrame, signals: tuple[str, ...]) -> pl.DataFrame:
    """Pivot long-format (cohort, signal_0, signal_id, value) to wide
    (cohort, signal_0, <signal>_1, <signal>_2, ...) with one column per signal."""
    wide = obs.filter(pl.col("signal_id").is_in(list(signals)))
    wide = wide.pivot(on="signal_id", index=["cohort", "signal_0"], values="value")
    # Enforce column order: cohort, signal_0, then signals in the canonical order.
    wide = wide.select(["cohort", "signal_0"] + list(signals))
    return wide.sort(["cohort", "signal_0"])


def build(
    obs_path: str,
    signals: tuple[str, ...] = HEALTH_SIGNALS,
    window_size: int = WINDOW_SIZE,
    early_pct: float = EARLY_PCT,
) -> WindowedData:
    """Build windowed dataset from observations.parquet.

    For each (cohort, cycle) with cycle >= 0, construct the window
    [cycle - window_size + 1 .. cycle] over the `signals` columns. When
    the cycle is too early (cycle < window_size - 1), left-pad with the
    first row's value repeated. Resulting row vector has shape
    (window_size * len(signals),), with signals in the order given and
    within each signal, cycles in chronological order.

    `is_training_subset`: True for rows whose cycle falls in the first
    `early_pct` of that cohort's trajectory.

    `mu`, `sigma`: per-signal z-score parameters, fit on the training
    subset only (no test leakage).
    """
    obs = pl.read_parquet(obs_path)
    wide = _wide_by_signal(obs, signals)

    X_rows: list[np.ndarray] = []
    cohort_rows: list[str] = []
    cycle_rows: list[int] = []
    is_train: list[bool] = []

    train_stack: list[np.ndarray] = []  # for computing mu/sigma on training subset

    for c in wide["cohort"].unique().to_list():
        coh = wide.filter(pl.col("cohort") == c).sort("signal_0")
        mat = coh.select(list(signals)).to_numpy().astype(np.float64)  # (n_cycles, n_signals)
        n_cycles = mat.shape[0]
        n_early = max(1, int(early_pct * n_cycles))

        for t in range(n_cycles):
            start = t - window_size + 1
            if start < 0:
                # Left-pad with mat[0] repeated
                pad = np.repeat(mat[0:1], -start, axis=0)
                window = np.concatenate([pad, mat[: t + 1]], axis=0)
            else:
                window = mat[start : t + 1]
            assert window.shape == (window_size, len(signals))
            # Flatten: [cycle_0_signal_0, cycle_0_signal_1, ..., cycle_W-1_signal_N-1]
            X_rows.append(window.reshape(-1))
            cohort_rows.append(c)
            cycle_rows.append(t)
            is_train.append(t < n_early)
            if t < n_early:
                train_stack.append(window)  # full window goes to train subset for scaler fit

    X = np.ascontiguousarray(np.stack(X_rows, axis=0), dtype=np.float32)
    cohort_arr = np.asarray(cohort_rows, dtype=object)
    cycle_arr = np.asarray(cycle_rows, dtype=np.int64)
    train_mask = np.asarray(is_train, dtype=bool)

    # Per-signal z-score: fit on the training subset only.
    # train_stack is a list of (window_size, n_signals) arrays — concatenate to (T_total, n_signals).
    train_pts = np.concatenate(train_stack, axis=0)  # (T_total, n_signals)
    mu = train_pts.mean(axis=0).astype(np.float32)
    sigma = train_pts.std(axis=0).astype(np.float32)
    sigma[sigma < 1e-6] = 1.0

    return WindowedData(
        X=X,
        cohort=cohort_arr,
        cycle=cycle_arr,
        is_training_subset=train_mask,
        mu=mu,
        sigma=sigma,
        n_signals=len(signals),
        window_size=window_size,
        signal_order=signals,
    )


def zscore(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray, n_signals: int,
           window_size: int) -> np.ndarray:
    """Apply z-score to flattened windows.
    X has shape (n_rows, window_size * n_signals); scaling parameters
    repeat per-signal across the window."""
    mu_tile = np.tile(mu, window_size).astype(np.float32)
    sig_tile = np.tile(sigma, window_size).astype(np.float32)
    return (X - mu_tile) / sig_tile
