"""Encode FD001 windows to CAE latents, produce data/fd001_cae/{train,test}.parquet.

Schema matches the public kit: (cohort, L01..L32, RUL). Row order matches
the public kit's fd001/train.parquet (cohorts alphabetically, cycles ASC
within each cohort) and fd001/test.parquet (last window per engine, same
engine order as public test.parquet).

Usage:
    PYTHONHASHSEED=42 ~/machine/.venv/bin/python cae/encode.py

Writes:
    data/fd001_cae/train.parquet   (20,631 rows × 34 cols)
    data/fd001_cae/test.parquet    (100 rows × 34 cols)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.use_deterministic_algorithms(True)
torch.set_num_threads(1)
torch.manual_seed(0)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cae.data import HEALTH_SIGNALS, WINDOW_SIZE, _wide_by_signal, zscore
from cae.model import SmallContractiveAutoencoder

RUL_CAP = 125.0

REPO         = Path(__file__).resolve().parents[1]
OBS_TRAIN    = Path("/Users/jasonrudder/domains/cmapss/FD_001/Train/observations.parquet")
OBS_TEST     = Path("/Users/jasonrudder/domains/cmapss/FD_001/Test/observations.parquet")
PUBLIC_TEST  = REPO / "data/fd001/test.parquet"
WEIGHTS      = REPO / "cae/fd001_cae_weights.pt"
META         = REPO / "cae/fd001_cae_meta.json"
OUT_DIR      = REPO / "data/fd001_cae"
OUT_TRAIN    = OUT_DIR / "train.parquet"
OUT_TEST     = OUT_DIR / "test.parquet"


def _load_model(meta: dict) -> SmallContractiveAutoencoder:
    h = meta["hyper"]
    model = SmallContractiveAutoencoder(
        input_dim=h["input_dim"], latent_dim=h["latent_dim"],
        hidden=tuple(h["hidden"]),
    )
    state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def _window_for(mat: np.ndarray, t: int, window_size: int) -> np.ndarray:
    """Build the window ending at cycle t (inclusive). Left-pad with mat[0] if t < window_size-1."""
    start = t - window_size + 1
    if start < 0:
        pad = np.repeat(mat[0:1], -start, axis=0)
        return np.concatenate([pad, mat[: t + 1]], axis=0)
    return mat[start : t + 1]


def build_windows_for_split(obs_path: Path, window_size: int,
                            signals: tuple[str, ...], per_engine_last_only: bool,
                            ) -> tuple[np.ndarray, list[str], list[int], dict[str, int]]:
    """Return (X, cohort_list, cycle_list, max_cycle_per_cohort).

    If per_engine_last_only, emit one window per cohort (the last cycle).
    Otherwise emit one window per (cohort, cycle), cycles ascending, cohorts
    in sorted (alphabetical string) order.
    """
    obs = pl.read_parquet(obs_path)
    wide = _wide_by_signal(obs, signals)

    Xs: list[np.ndarray] = []
    coh: list[str] = []
    cyc: list[int] = []
    max_cyc: dict[str, int] = {}

    for c in sorted(wide["cohort"].unique().to_list()):
        sub = wide.filter(pl.col("cohort") == c).sort("signal_0")
        mat = sub.select(list(signals)).to_numpy().astype(np.float64)
        n = mat.shape[0]
        max_cyc[c] = n - 1
        cycles = range(n - 1, n) if per_engine_last_only else range(n)
        for t in cycles:
            w = _window_for(mat, t, window_size)
            Xs.append(w.reshape(-1))
            coh.append(c)
            cyc.append(t)

    X = np.ascontiguousarray(np.stack(Xs, axis=0), dtype=np.float32)
    return X, coh, cyc, max_cyc


def main() -> int:
    t0 = time.time()
    print("=" * 72)
    print("  Encode FD001 windows → CAE latents (train + test parquets)")
    print("=" * 72)

    if not WEIGHTS.exists() or not META.exists():
        print(f"  STOP: missing {WEIGHTS.name} or {META.name} — train CAE first")
        return 2

    meta = json.loads(META.read_text())
    hp = meta["hyper"]
    signals = tuple(hp["signals"])
    window_size = int(hp["window_size"])
    latent_dim = int(hp["latent_dim"])
    n_signals = int(hp["n_signals"])
    mu = np.asarray(meta["mu"], dtype=np.float32)
    sigma = np.asarray(meta["sigma"], dtype=np.float32)

    print(f"  loaded model: input_dim={hp['input_dim']}  latent={latent_dim}")
    print(f"  signals ({len(signals)}): {signals}")
    print(f"  window_size={window_size}  RUL_CAP={RUL_CAP}")

    model = _load_model(meta)

    # ── TRAIN: all (cohort, cycle) windows, cohorts alphabetically ─
    X_tr, coh_tr, cyc_tr, max_cyc_tr = build_windows_for_split(
        OBS_TRAIN, window_size, signals, per_engine_last_only=False,
    )
    print(f"  train windows: {len(X_tr):,}  cohorts: {len(max_cyc_tr)}")

    X_tr_z = zscore(X_tr, mu, sigma, n_signals, window_size)

    CHUNK = 4096
    lat_tr = np.empty((len(X_tr_z), latent_dim), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, len(X_tr_z), CHUNK):
            slab = torch.from_numpy(np.ascontiguousarray(X_tr_z[i:i+CHUNK]))
            z = model.encoder(slab)
            lat_tr[i:i+len(slab)] = z.numpy().astype(np.float64)

    # RUL per row: clip(max_cyc - cyc, 0, 125)
    rul_tr = np.array(
        [min(max(max_cyc_tr[c] - t, 0.0), RUL_CAP) for c, t in zip(coh_tr, cyc_tr)],
        dtype=np.float64,
    )

    # Emit train.parquet: cohort, L01..L32, RUL (match public kit schema)
    tr_df = pl.DataFrame({"cohort": coh_tr})
    for j in range(latent_dim):
        tr_df = tr_df.with_columns(pl.Series(f"L{j+1:02d}", lat_tr[:, j]))
    tr_df = tr_df.with_columns(pl.Series("RUL", rul_tr))

    # ── TEST: one last window per engine, RUL from public test.parquet ─
    X_te, coh_te, cyc_te, _ = build_windows_for_split(
        OBS_TEST, window_size, signals, per_engine_last_only=True,
    )
    print(f"  test  windows: {len(X_te)}  (last per engine)")

    X_te_z = zscore(X_te, mu, sigma, n_signals, window_size)
    with torch.no_grad():
        z_te = model.encoder(torch.from_numpy(np.ascontiguousarray(X_te_z)))
    lat_te = z_te.numpy().astype(np.float64)

    # Pull RUL from public test.parquet (ground truth from RUL_FD001.txt)
    public_te = pl.read_parquet(PUBLIC_TEST).select(["cohort", "RUL"])
    # Our test rows are in sorted(cohort) order; join by cohort to get RUL
    te_frame = pl.DataFrame({"cohort": coh_te})
    for j in range(latent_dim):
        te_frame = te_frame.with_columns(pl.Series(f"L{j+1:02d}", lat_te[:, j]))
    te_df = te_frame.join(public_te, on="cohort", how="left")

    # Reorder test rows to match public_te's exact cohort order — stacking
    # ensembles don't care, but matching the public kit's grain is cleaner.
    te_df = public_te.select("cohort").join(te_df, on="cohort", how="left")

    # Sanity: none null
    n_null_tr = int(tr_df["RUL"].is_null().sum())
    n_null_te = int(te_df["RUL"].is_null().sum())
    print(f"  train RUL null: {n_null_tr}   test RUL null: {n_null_te}")
    if n_null_tr or n_null_te:
        print("  STOP: null RUL values encountered")
        return 3

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tr_df.write_parquet(OUT_TRAIN)
    te_df.write_parquet(OUT_TEST)

    print()
    print(f"  wrote {OUT_TRAIN}  shape={tr_df.shape}")
    print(f"  wrote {OUT_TEST}   shape={te_df.shape}")
    print(f"  elapsed: {time.time() - t0:.1f}s")
    print(f"  schema check (first 3 cols): {tr_df.columns[:3]}  last col: {tr_df.columns[-1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
