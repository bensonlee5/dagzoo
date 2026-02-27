"""Random matrix generation."""

from __future__ import annotations

import math

import torch

from cauchy_generator.functions.activations import apply_random_activation
from cauchy_generator.math_utils import log_uniform as _log_uniform
from cauchy_generator.sampling.random_weights import sample_random_weights


def _row_normalize(m: torch.Tensor) -> torch.Tensor:
    """Normalize matrix rows to unit L2 norm in torch."""
    norms = torch.linalg.norm(m, dim=1, keepdim=True)
    return m / torch.clamp(norms, min=1e-6)


def _sample_gaussian(m: int, k: int, generator: torch.Generator, device: str) -> torch.Tensor:
    """Sample i.i.d. Gaussian matrix in torch."""
    return torch.randn(m, k, generator=generator, device=device)


def _sample_weights_matrix(
    m: int,
    k: int,
    generator: torch.Generator,
    device: str,
    *,
    noise_sigma_multiplier: float = 1.0,
) -> torch.Tensor:
    """Sample weight-modulated Gaussian matrix in torch."""
    g = torch.randn(m, k, generator=generator, device=device)
    q_low = 0.1 / math.log(k + 1.0)
    shared_q = _log_uniform(generator, q_low, 6.0, device)
    shared_sigma = _log_uniform(generator, 1e-4, 10.0, device)
    rows = torch.stack(
        [
            sample_random_weights(
                k,
                generator,
                device,
                q=shared_q,
                sigma=shared_sigma,
                sigma_multiplier=noise_sigma_multiplier,
            )
            for _ in range(m)
        ],
        dim=0,
    )
    return _row_normalize(g * rows)


def _sample_singular_values_matrix(
    m: int,
    k: int,
    generator: torch.Generator,
    device: str,
    *,
    noise_sigma_multiplier: float = 1.0,
) -> torch.Tensor:
    """Sample matrix with random singular values in torch."""
    d = min(m, k)
    u = torch.randn(m, d, generator=generator, device=device)
    v = torch.randn(d, k, generator=generator, device=device)
    w = sample_random_weights(d, generator, device, sigma_multiplier=noise_sigma_multiplier)
    s = torch.diag(w)
    return u @ s @ v


def _sample_kernel_matrix(m: int, k: int, generator: torch.Generator, device: str) -> torch.Tensor:
    """Sample kernel-derived matrix in torch."""
    pts = torch.randn(k + m, 3, generator=generator, device=device)
    gamma = _log_uniform(generator, 0.1, 10.0, device)
    left = pts[:m].unsqueeze(1)
    right = pts[m:].unsqueeze(0)
    dist = torch.norm(left - right, dim=2)
    kernel = torch.exp(-gamma * dist)

    s = torch.empty(m, k, device=device).uniform_(0, 1, generator=generator)
    sign = torch.where(s < 0.5, -1.0, 1.0)
    return kernel * sign


def sample_random_matrix(
    out_dim: int,
    in_dim: int,
    generator: torch.Generator,
    device: str,
    *,
    kind: str | None = None,
    noise_sigma_multiplier: float = 1.0,
) -> torch.Tensor:
    """Sample one of the supported matrix families in torch."""
    if out_dim <= 0 or in_dim <= 0:
        raise ValueError(f"Matrix dimensions must be > 0. Got out_dim={out_dim}, in_dim={in_dim}")

    kinds = ["gaussian", "weights", "singular_values", "kernel", "activation"]
    if kind is None:
        idx = torch.randint(0, len(kinds), (1,), generator=generator).item()
        kind = kinds[int(idx)]

    if kind == "gaussian":
        m = _sample_gaussian(out_dim, in_dim, generator, device)
    elif kind == "weights":
        m = _sample_weights_matrix(
            out_dim,
            in_dim,
            generator,
            device,
            noise_sigma_multiplier=noise_sigma_multiplier,
        )
    elif kind == "singular_values":
        m = _sample_singular_values_matrix(
            out_dim,
            in_dim,
            generator,
            device,
            noise_sigma_multiplier=noise_sigma_multiplier,
        )
    elif kind == "kernel":
        m = _sample_kernel_matrix(out_dim, in_dim, generator, device)
    elif kind == "activation":
        other_kinds = ["gaussian", "weights", "singular_values", "kernel"]
        idx = torch.randint(0, len(other_kinds), (1,), generator=generator).item()
        m = sample_random_matrix(
            out_dim,
            in_dim,
            generator,
            device,
            kind=other_kinds[int(idx)],
            noise_sigma_multiplier=noise_sigma_multiplier,
        )
        m = apply_random_activation(m, generator, with_standardize=False)
        m = m + 1e-3 * torch.randn(m.shape, generator=generator, device=device)
    else:
        raise ValueError(f"Unknown matrix kind: {kind}")

    m = m + 1e-6 * torch.randn(m.shape, generator=generator, device=device)
    return _row_normalize(m)
