"""Random positive weights sampling (Appendix E.11)."""

import numpy as np
import torch

from cauchy_generator.math_utils import (
    log_uniform as _log_uniform,
    log_uniform_torch as _log_uniform_torch,
)


def sample_random_weights_torch(
    dim: int,
    generator: torch.Generator,
    device: str,
    *,
    min_q_scale: float = 0.1,
    max_q: float = 6.0,
    sigma_min: float = 1e-4,
    sigma_max: float = 10.0,
    q: float | None = None,
    sigma: float | None = None,
) -> torch.Tensor:
    """Sample positive normalized weights using torch."""
    if dim <= 0:
        raise ValueError(f"dim must be > 0, got {dim}")

    m = torch.arange(1, dim + 1, dtype=torch.float32, device=device)
    if q is None:
        q_low = min_q_scale / np.log(dim + 1.0)
        q = _log_uniform_torch(generator, q_low, max_q, device)
    if sigma is None:
        sigma = _log_uniform_torch(generator, sigma_min, sigma_max, device)

    noise = torch.randn(dim, generator=generator, device=device) * sigma
    w = torch.pow(m, -q) * torch.exp(noise)
    w = torch.clamp(w, min=1e-12)
    w /= torch.sum(w)

    # Shuffle
    idx = torch.randperm(dim, generator=generator, device=device)
    return w[idx]


def sample_random_weights(
    dim: int,
    rng: np.random.Generator,
    *,
    min_q_scale: float = 0.1,
    max_q: float = 6.0,
    sigma_min: float = 1e-4,
    sigma_max: float = 10.0,
    q: float | None = None,
    sigma: float | None = None,
) -> np.ndarray:
    """
    Sample positive normalized weights with shuffled power-law decay.

    Mirrors Appendix E.11:
      w_m = m^{-q} * exp(Normal(0, sigma^2))

    When *q* and *sigma* are supplied they are used directly (shared-parameter
    mode for correlated weight matrices, per E.10).
    """

    if dim <= 0:
        raise ValueError(f"dim must be > 0, got {dim}")

    m = np.arange(1, dim + 1, dtype=np.float64)
    if q is None:
        q_low = min_q_scale / np.log(dim + 1.0)
        q = _log_uniform(rng, q_low, max_q)
    if sigma is None:
        sigma = _log_uniform(rng, sigma_min, sigma_max)

    w = np.power(m, -q) * np.exp(rng.normal(0.0, sigma, size=dim))
    w = np.maximum(w, 1e-12)
    w /= w.sum()
    rng.shuffle(w)
    return w.astype(np.float32)
