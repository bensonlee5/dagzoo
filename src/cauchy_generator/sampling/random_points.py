"""Random points mechanism."""

import torch

from cauchy_generator.functions.random_functions import apply_random_function
from cauchy_generator.sampling.random_weights import sample_random_weights


def _sample_unit_ball(n: int, d: int, generator: torch.Generator, device: str) -> torch.Tensor:
    """Sample points uniformly from the d-dimensional unit ball in torch."""
    v = torch.randn(n, d, generator=generator, device=device)
    v /= torch.clamp(torch.norm(v, dim=1, keepdim=True), min=1e-6)
    u = torch.empty(n, 1, device=device).uniform_(0.0, 1.0, generator=generator)
    r = torch.pow(u, 1.0 / d)
    return v * r


def _sample_random_covariance_normal(
    n: int,
    d: int,
    generator: torch.Generator,
    device: str,
    *,
    sigma_multiplier: float = 1.0,
) -> torch.Tensor:
    """Sample normal points with random anisotropic covariance transform."""
    x = torch.randn(n, d, generator=generator, device=device)
    w = sample_random_weights(d, generator, device, sigma_multiplier=sigma_multiplier)
    a = torch.randn(d, d, generator=generator, device=device)
    return (x * w.unsqueeze(0)) @ a.t()


def sample_random_points(
    n_rows: int,
    dim: int,
    generator: torch.Generator,
    device: str,
    *,
    mechanism_logit_tilt: float = 0.0,
    noise_sigma_multiplier: float = 1.0,
) -> torch.Tensor:
    """Sample base points and transform through a random function in torch."""
    if n_rows <= 0 or dim <= 0:
        raise ValueError(f"n_rows and dim must be > 0. Got n_rows={n_rows}, dim={dim}")

    kinds = ["normal", "uniform", "unit_ball", "normal_cov"]
    idx = torch.randint(0, len(kinds), (1,), generator=generator).item()
    base_kind = kinds[int(idx)]

    if base_kind == "normal":
        base = torch.randn(n_rows, dim, generator=generator, device=device)
    elif base_kind == "uniform":
        base = torch.empty(n_rows, dim, device=device).uniform_(-1.0, 1.0, generator=generator)
    elif base_kind == "unit_ball":
        base = _sample_unit_ball(n_rows, dim, generator, device)
    else:
        base = _sample_random_covariance_normal(
            n_rows,
            dim,
            generator,
            device,
            sigma_multiplier=noise_sigma_multiplier,
        )

    return apply_random_function(
        base,
        generator,
        out_dim=dim,
        mechanism_logit_tilt=mechanism_logit_tilt,
        noise_sigma_multiplier=noise_sigma_multiplier,
    )
