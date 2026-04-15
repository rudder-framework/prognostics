"""SmallContractiveAutoencoder — MLP with Jacobian penalty on latent w.r.t. input."""
from __future__ import annotations

import torch
import torch.nn as nn


class SmallContractiveAutoencoder(nn.Module):
    """Symmetric MLP autoencoder with ReLU. Encoder and decoder dims mirror.

    Default shape (public kit FD001 contract, window=30, 15 signals):
        input_dim  = 450  (30 cycles × 15 sensors, flattened)
        encoder    = [450 -> 128 -> 64 -> 32]
        decoder    = [32  -> 64  -> 128 -> 450]
        latent_dim = 32
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden: tuple[int, ...] = (128, 64),
    ) -> None:
        super().__init__()
        enc_layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden:
            enc_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        enc_layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers: list[nn.Module] = []
        prev = latent_dim
        for h in reversed(hidden):
            dec_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (reconstruction, latent)."""
        z = self.encoder(x)
        r = self.decoder(z)
        return r, z


def jacobian_frobenius_sq(encoder_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute sum over a batch of ||∂h/∂x||_F^2 using autograd.

    Assumes x has requires_grad=True and encoder_output was produced from it.
    Returns a scalar — the sum over the batch of squared Frobenius norms.

    Cost: latent_dim backward passes, each O(forward pass). Deterministic on CPU.
    """
    batch, latent_dim = encoder_output.shape
    total = torch.zeros((), dtype=encoder_output.dtype, device=encoder_output.device)
    for j in range(latent_dim):
        # sum of latent_j across the batch — its grad w.r.t. x has shape (batch, input_dim)
        scalar = encoder_output[:, j].sum()
        grad = torch.autograd.grad(
            outputs=scalar,
            inputs=x,
            create_graph=True,
            retain_graph=True,
        )[0]
        total = total + (grad ** 2).sum()
    # Return sum over batch; caller decides to divide by batch or not.
    return total
