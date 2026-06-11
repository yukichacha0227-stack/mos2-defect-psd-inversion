from __future__ import annotations

import numpy as np
import torch

from .config import DataConfig


def generate_synthetic_psd(
    samples: int,
    config: DataConfig | None = None,
    seed: int = 7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a public demo dataset with two Lorentzian PSD components.

    The generator is intentionally lightweight. It is not a replacement for
    experimental data; it is a reproducible stand-in for demos and tests.
    """

    cfg = config or DataConfig()
    rng = np.random.default_rng(seed)
    freq = np.arange(1, cfg.sequence_length + 1, dtype=np.float32)

    ea_alpha = rng.uniform(1.0, 15.0, size=samples).astype(np.float32)
    ea_beta = ea_alpha + rng.uniform(3.0, 20.0, size=samples).astype(np.float32)

    kbt_mev = 25.85
    f0_alpha = rng.uniform(120.0, 800.0, size=samples).astype(np.float32)
    f0_beta = rng.uniform(20.0, 220.0, size=samples).astype(np.float32)
    f_alpha = np.clip(f0_alpha * np.exp(-ea_alpha / kbt_mev), 1.0, 3000.0)
    f_beta = np.clip(f0_beta * np.exp(-ea_beta / kbt_mev), 1.0, 3000.0)

    amp_alpha = rng.lognormal(mean=-54.0, sigma=0.45, size=samples).astype(np.float32)
    amp_beta = rng.lognormal(mean=-54.2, sigma=0.45, size=samples).astype(np.float32)
    background = rng.lognormal(mean=-58.0, sigma=0.25, size=samples).astype(np.float32)

    psd = (
        amp_alpha[:, None] / (1.0 + (freq[None, :] / f_alpha[:, None]) ** 2)
        + amp_beta[:, None] / (1.0 + (freq[None, :] / f_beta[:, None]) ** 2)
        + background[:, None]
    )
    psd *= rng.lognormal(mean=0.0, sigma=0.03, size=psd.shape).astype(np.float32)

    inputs = torch.from_numpy(psd[:, None, :]).float()
    targets = torch.from_numpy(np.stack([ea_alpha, ea_beta], axis=1)).float()
    return inputs, targets
