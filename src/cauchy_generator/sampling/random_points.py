"""Random points mechanism (Appendix E.12)."""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch
else:
    try:
        import torch
    except ImportError:
        torch = None

from cauchy_generator.functions.random_functions import (
    apply_random_function,
    apply_random_function_torch,
)
from cauchy_generator.sampling.random_weights import (
    sample_random_weights,
    sample_random_weights_torch,
)


def _sample_unit_ball(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Sample points uniformly from the d-dimensional unit ball."""

    v = rng.normal(size=(n, d)).astype(np.float32)
    v /= np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-6, None)
    r = np.power(rng.uniform(0.0, 1.0, size=(n, 1)), 1.0 / d).astype(np.float32)
    return (v * r).astype(np.float32)


def _sample_unit_ball_torch(
    n: int, d: int, generator: torch.Generator, device: str
) -> torch.Tensor:
    """Sample points uniformly from the d-dimensional unit ball in torch."""
    v = torch.randn(n, d, generator=generator, device=device)
    v /= torch.clamp(torch.norm(v, dim=1, keepdim=True), min=1e-6)
    u = torch.empty(n, 1, device=device).uniform_(0.0, 1.0, generator=generator)
    r = torch.pow(u, 1.0 / d)
    return v * r


def _sample_random_covariance_normal(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Sample normal points with random anisotropic covariance transform."""

    x = rng.normal(size=(n, d)).astype(np.float32)
    w = sample_random_weights(d, rng).astype(np.float32)
    a = rng.normal(size=(d, d)).astype(np.float32)
    return ((x * w[None, :]) @ a.T).astype(np.float32)


def _sample_random_covariance_normal_torch(
    n: int, d: int, generator: torch.Generator, device: str
) -> torch.Tensor:
    """Sample normal points with random covariance in torch."""
    x = torch.randn(n, d, generator=generator, device=device)
    w = sample_random_weights_torch(d, generator, device)
    a = torch.randn(d, d, generator=generator, device=device)
    return (x * w.unsqueeze(0)) @ a.t()


def sample_random_points_torch(
    n_rows: int,
    dim: int,
    generator: torch.Generator,
    device: str,
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
        base = _sample_unit_ball_torch(n_rows, dim, generator, device)
    else:
        base = _sample_random_covariance_normal_torch(n_rows, dim, generator, device)

    return apply_random_function_torch(base, generator, out_dim=dim)


def sample_random_points(
    n_rows: int,
    dim: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample base points and transform through a random function."""

    if n_rows <= 0 or dim <= 0:
        raise ValueError(f"n_rows and dim must be > 0. Got n_rows={n_rows}, dim={dim}")

    base_kind = str(rng.choice(["normal", "uniform", "unit_ball", "normal_cov"]))
    if base_kind == "normal":
        base = rng.normal(size=(n_rows, dim)).astype(np.float32)
    elif base_kind == "uniform":
        base = rng.uniform(-1.0, 1.0, size=(n_rows, dim)).astype(np.float32)
    elif base_kind == "unit_ball":
        base = _sample_unit_ball(n_rows, dim, rng)
    else:
        base = _sample_random_covariance_normal(n_rows, dim, rng)

    return apply_random_function(base, rng, out_dim=dim).astype(np.float32)
