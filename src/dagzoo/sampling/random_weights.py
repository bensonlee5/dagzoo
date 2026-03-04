"""Random positive weights sampling."""

import math

import torch

from dagzoo.math_utils import log_uniform as _log_uniform
from dagzoo.sampling.noise import NoiseSamplingSpec, sample_noise_from_spec

_LOG_WEIGHT_CLAMP = 60.0


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
    noise_spec: NoiseSamplingSpec | None = None,
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

    noise = sample_noise_from_spec(
        (dim,),
        generator=generator,
        device=device,
        noise_spec=noise_spec,
        scale_multiplier=effective_sigma,
    )
    log_w = (-float(q) * torch.log(m)) + noise
    log_w = torch.nan_to_num(
        log_w,
        nan=0.0,
        posinf=_LOG_WEIGHT_CLAMP,
        neginf=-_LOG_WEIGHT_CLAMP,
    )
    log_w = torch.clamp(log_w, min=-_LOG_WEIGHT_CLAMP, max=_LOG_WEIGHT_CLAMP)
    log_w = log_w - torch.max(log_w)
    w = torch.exp(log_w)
    w = torch.clamp(w, min=1e-12)
    w /= torch.sum(w)

    # Shuffle
    idx = torch.randperm(dim, generator=generator, device=device)
    return w[idx]
