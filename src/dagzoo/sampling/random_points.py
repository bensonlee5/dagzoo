"""Random points mechanism."""

import torch

from dagzoo.core.layout_types import MechanismFamily
from dagzoo.functions._rng_helpers import randint_scalar
from dagzoo.functions.random_functions import apply_random_function
from dagzoo.sampling.noise import NoiseSamplingSpec, sample_noise_from_spec
from dagzoo.sampling.random_weights import sample_random_weights


def _sample_unit_ball(
    n: int,
    d: int,
    generator: torch.Generator,
    device: str,
) -> torch.Tensor:
    """Sample points uniformly from the d-dimensional unit ball in torch."""
    # Uniform sphere directions require normalized Gaussian draws.
    # Using non-Gaussian draws here breaks rotational invariance.
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
    noise_spec: NoiseSamplingSpec | None = None,
) -> torch.Tensor:
    """Sample normal points with random anisotropic covariance transform."""
    x = sample_noise_from_spec(
        (n, d),
        generator=generator,
        device=device,
        noise_spec=noise_spec,
    )
    w = sample_random_weights(
        d,
        generator,
        device,
        sigma_multiplier=sigma_multiplier,
        noise_spec=noise_spec,
    )
    a = sample_noise_from_spec(
        (d, d),
        generator=generator,
        device=device,
        noise_spec=noise_spec,
    )
    return (x * w.unsqueeze(0)) @ a.t()


def sample_random_points(
    n_rows: int,
    dim: int,
    generator: torch.Generator,
    device: str,
    *,
    mechanism_logit_tilt: float = 0.0,
    function_family_mix: dict[MechanismFamily, float] | None = None,
    noise_sigma_multiplier: float = 1.0,
    noise_spec: NoiseSamplingSpec | None = None,
) -> torch.Tensor:
    """Sample base points and transform through a random function in torch."""
    if n_rows <= 0 or dim <= 0:
        raise ValueError(f"n_rows and dim must be > 0. Got n_rows={n_rows}, dim={dim}")

    kinds = ["normal", "uniform", "unit_ball", "normal_cov"]
    idx = randint_scalar(0, len(kinds), generator)
    base_kind = kinds[int(idx)]

    if base_kind == "normal":
        base = sample_noise_from_spec(
            (n_rows, dim),
            generator=generator,
            device=device,
            noise_spec=noise_spec,
        )
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
            noise_spec=noise_spec,
        )

    return apply_random_function(
        base,
        generator,
        out_dim=dim,
        mechanism_logit_tilt=mechanism_logit_tilt,
        function_family_mix=function_family_mix,
        noise_sigma_multiplier=noise_sigma_multiplier,
        noise_spec=noise_spec,
    )
