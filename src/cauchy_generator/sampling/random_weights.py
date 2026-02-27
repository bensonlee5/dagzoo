"""Random positive weights sampling."""

import math

import torch

from cauchy_generator.math_utils import log_uniform as _log_uniform


def sample_random_weights(
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
    sigma_multiplier: float = 1.0,
) -> torch.Tensor:
    """Sample positive normalized weights using torch."""
    if dim <= 0:
        raise ValueError(f"dim must be > 0, got {dim}")
    if not math.isfinite(float(sigma_multiplier)) or float(sigma_multiplier) <= 0.0:
        raise ValueError(f"sigma_multiplier must be a finite value > 0, got {sigma_multiplier!r}")

    m = torch.arange(1, dim + 1, dtype=torch.float32, device=device)
    if q is None:
        q_low = min_q_scale / math.log(dim + 1.0)
        q = _log_uniform(generator, q_low, max_q, device)
    if sigma is None:
        sigma = _log_uniform(generator, sigma_min, sigma_max, device)
    effective_sigma = float(sigma) * float(sigma_multiplier)

    noise = torch.randn(dim, generator=generator, device=device) * effective_sigma
    w = torch.pow(m, -q) * torch.exp(noise)
    w = torch.clamp(w, min=1e-12)
    w /= torch.sum(w)

    # Shuffle
    idx = torch.randperm(dim, generator=generator, device=device)
    return w[idx]
