"""Train the Contractive Autoencoder on FD001 healthy cycles.

Usage:
    PYTHONHASHSEED=42 ~/machine/.venv/bin/python cae/train.py

Writes:
    cae/fd001_cae_weights.pt    — state_dict
    cae/fd001_cae_meta.json     — mu, sigma, architecture, training log
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `import cae.*`
from cae.data import build, zscore, HEALTH_SIGNALS, WINDOW_SIZE
from cae.model import SmallContractiveAutoencoder, jacobian_frobenius_sq


# ── determinism guardrails (CPU only) ────────────────────────────
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.use_deterministic_algorithms(True)
torch.set_num_threads(1)

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ── hyperparams ───────────────────────────────────────────────────
OBS_PATH    = "/Users/jasonrudder/domains/cmapss/FD_001/Train/observations.parquet"
LATENT_DIM  = 32
HIDDEN      = (128, 64)
BATCH_SIZE  = 256
EPOCHS      = 300
LR          = 1e-3
LAMBDA_C    = 0.1    # contractive penalty coefficient (per-batch mean)

OUT_DIR     = Path(__file__).resolve().parent
OUT_WEIGHTS = OUT_DIR / "fd001_cae_weights.pt"
OUT_META    = OUT_DIR / "fd001_cae_meta.json"


def main() -> int:
    t0 = time.time()
    print("=" * 72)
    print("  Contractive Autoencoder — FD001 healthy-cycles pretrain")
    print("=" * 72)

    # ── build windowed dataset ────────────────────────────────────
    data = build(OBS_PATH, signals=HEALTH_SIGNALS, window_size=WINDOW_SIZE,
                 early_pct=0.20)
    n_total = data.X.shape[0]
    n_train = int(data.is_training_subset.sum())
    n_signals = data.n_signals
    input_dim = WINDOW_SIZE * n_signals
    print(f"  observations: {n_total:,} windows total   {n_train:,} in healthy subset")
    print(f"  input_dim: {input_dim}  (window={WINDOW_SIZE} × signals={n_signals})")
    print(f"  signals: {data.signal_order}")

    # Z-score the full set with mu/sigma fit on training subset only
    X_all = zscore(data.X, data.mu, data.sigma, n_signals, WINDOW_SIZE)
    X_train = X_all[data.is_training_subset]

    # ── model ─────────────────────────────────────────────────────
    model = SmallContractiveAutoencoder(
        input_dim=input_dim, latent_dim=LATENT_DIM, hidden=HIDDEN
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    mse = nn.MSELoss()

    X_t = torch.from_numpy(np.ascontiguousarray(X_train))
    rng = np.random.default_rng(SEED)
    n = X_t.shape[0]

    print(f"  model: input {input_dim} → hidden {list(HIDDEN)} → latent {LATENT_DIM}")
    print(f"  train config: batch={BATCH_SIZE} epochs={EPOCHS} lr={LR} λ_c={LAMBDA_C}")
    print(f"  determinism: torch.use_deterministic_algorithms(True), CPU, 1 thread")
    print("-" * 72)

    train_log: list[dict] = []
    for ep in range(EPOCHS):
        perm = rng.permutation(n)
        ep_mse = ep_jac = ep_total = 0.0
        n_batches = 0
        model.train()
        for start in range(0, n, BATCH_SIZE):
            idx = torch.from_numpy(np.ascontiguousarray(perm[start : start + BATCH_SIZE]))
            if len(idx) < 8:
                continue
            x = X_t[idx].clone().detach().requires_grad_(True)

            # Forward: encoder + decoder
            z = model.encoder(x)
            r = model.decoder(z)

            L_recon = mse(r, x)
            # Contractive penalty: ||∂h/∂x||_F^2, summed over batch, averaged by batch size
            L_jac_sum = jacobian_frobenius_sq(z, x)
            L_jac = L_jac_sum / len(idx)

            L = L_recon + LAMBDA_C * L_jac

            optimizer.zero_grad()
            L.backward()
            optimizer.step()

            ep_mse   += L_recon.item()
            ep_jac   += L_jac.item()
            ep_total += L.item()
            n_batches += 1

        if n_batches == 0:
            continue
        row = dict(
            epoch=ep + 1,
            mse_recon=ep_mse / n_batches,
            jac_frobenius=ep_jac / n_batches,
            total=ep_total / n_batches,
            wall_s=round(time.time() - t0, 1),
        )
        train_log.append(row)
        if ep == 0 or (ep + 1) % 20 == 0 or ep == EPOCHS - 1:
            print(f"  epoch {ep+1:>3d}/{EPOCHS}  "
                  f"recon={row['mse_recon']:.6f}  "
                  f"jac={row['jac_frobenius']:.4f}  "
                  f"total={row['total']:.4f}  "
                  f"t={row['wall_s']:.0f}s")

    # ── save ──────────────────────────────────────────────────────
    torch.save(model.state_dict(), OUT_WEIGHTS)
    print(f"\n  saved weights → {OUT_WEIGHTS}")

    meta = dict(
        hyper=dict(
            latent_dim=LATENT_DIM, hidden=list(HIDDEN), window_size=WINDOW_SIZE,
            n_signals=n_signals, input_dim=input_dim, batch_size=BATCH_SIZE,
            epochs=EPOCHS, lr=LR, lambda_c=LAMBDA_C, seed=SEED,
            early_pct=0.20, signals=list(data.signal_order),
        ),
        n_total_windows=int(n_total),
        n_train_windows=int(n_train),
        mu=data.mu.tolist(),
        sigma=data.sigma.tolist(),
        train_log=train_log,
        final=train_log[-1] if train_log else None,
        elapsed_s=round(time.time() - t0, 1),
    )
    OUT_META.write_text(json.dumps(meta, indent=2))
    print(f"  saved meta    → {OUT_META}")
    print(f"  elapsed: {meta['elapsed_s']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
