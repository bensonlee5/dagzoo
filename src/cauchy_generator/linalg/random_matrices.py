"""Random matrix generation (Appendix E.10)."""

from __future__ import annotations

import typing
from typing import Literal

import numpy as np
import torch

from cauchy_generator.functions.activations import (
    apply_random_activation,
    apply_random_activation_torch,
)
from cauchy_generator.math_utils import (
    log_uniform as _log_uniform,
    log_uniform_torch as _log_uniform_torch,
)
from cauchy_generator.sampling.random_weights import (
    sample_random_weights,
    sample_random_weights_torch,
)

MatrixKind = Literal["gaussian", "weights", "singular_values", "kernel", "activation"]


def _row_normalize(m: np.ndarray) -> np.ndarray:
    """Normalize matrix rows to unit L2 norm."""

    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.clip(norms, 1e-6, None)


def _row_normalize_torch(m: torch.Tensor) -> torch.Tensor:
    """Normalize matrix rows to unit L2 norm in torch."""
    norms = torch.linalg.norm(m, dim=1, keepdim=True)
    return m / torch.clamp(norms, min=1e-6)


def _sample_gaussian(m: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Sample i.i.d. Gaussian matrix."""

    return rng.normal(size=(m, k)).astype(np.float32)


def _sample_gaussian_torch(m: int, k: int, generator: torch.Generator, device: str) -> torch.Tensor:
    """Sample i.i.d. Gaussian matrix in torch."""
    return torch.randn(m, k, generator=generator, device=device)


def _sample_weights_matrix(m: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Sample weight-modulated Gaussian matrix and row-normalize.

    Per E.10, q and sigma are shared across all rows so the weight vectors
    are correlated through the correlated sampling mechanism.
    """

    g = rng.normal(size=(m, k)).astype(np.float32)
    q_low = 0.1 / np.log(k + 1.0)
    shared_q = _log_uniform(rng, q_low, 6.0)
    shared_sigma = _log_uniform(rng, 1e-4, 10.0)
    rows = np.stack(
        [sample_random_weights(k, rng, q=shared_q, sigma=shared_sigma) for _ in range(m)],
        axis=0,
    )
    return _row_normalize(g * rows)


def _sample_weights_matrix_torch(
    m: int, k: int, generator: torch.Generator, device: str
) -> torch.Tensor:
    """Sample weight-modulated Gaussian matrix in torch."""
    g = torch.randn(m, k, generator=generator, device=device)
    q_low = 0.1 / np.log(k + 1.0)
    shared_q = _log_uniform_torch(generator, q_low, 6.0, device)
    shared_sigma = _log_uniform_torch(generator, 1e-4, 10.0, device)
    rows = torch.stack(
        [
            sample_random_weights_torch(k, generator, device, q=shared_q, sigma=shared_sigma)
            for _ in range(m)
        ],
        dim=0,
    )
    return _row_normalize_torch(g * rows)


def _sample_singular_values_matrix(m: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Sample matrix with random singular-value-like decay factors."""

    d = min(m, k)
    u = rng.normal(size=(m, d)).astype(np.float32)
    v = rng.normal(size=(d, k)).astype(np.float32)
    w = sample_random_weights(d, rng)
    s = np.diag(w.astype(np.float32))
    return u @ s @ v


def _sample_singular_values_matrix_torch(
    m: int, k: int, generator: torch.Generator, device: str
) -> torch.Tensor:
    """Sample matrix with random singular values in torch."""
    d = min(m, k)
    u = torch.randn(m, d, generator=generator, device=device)
    v = torch.randn(d, k, generator=generator, device=device)
    w = sample_random_weights_torch(d, generator, device)
    s = torch.diag(w)
    return u @ s @ v


def _sample_kernel_matrix(m: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Sample Laplace-kernel-derived matrix with random sign flips."""

    pts = rng.normal(size=(k + m, 3)).astype(np.float32)
    gamma = _log_uniform(rng, 0.1, 10.0)
    left = pts[:m][:, None, :]
    right = pts[m:][None, :, :]
    dist = np.linalg.norm(left - right, axis=2)
    kernel = np.exp(-gamma * dist).astype(np.float32)
    sign = rng.choice([-1.0, 1.0], size=(m, k)).astype(np.float32)
    return kernel * sign


def _sample_kernel_matrix_torch(
    m: int, k: int, generator: torch.Generator, device: str
) -> torch.Tensor:
    """Sample kernel-derived matrix in torch."""
    pts = torch.randn(k + m, 3, generator=generator, device=device)
    gamma = _log_uniform_torch(generator, 0.1, 10.0, device)
    left = pts[:m].unsqueeze(1)
    right = pts[m:].unsqueeze(0)
    dist = torch.norm(left - right, dim=2)
    kernel = torch.exp(-gamma * dist)

    s = torch.empty(m, k, device=device).uniform_(0, 1, generator=generator)
    sign = torch.where(s < 0.5, -1.0, 1.0)
    return kernel * sign


def sample_random_matrix_torch(
    out_dim: int,
    in_dim: int,
    generator: torch.Generator,
    device: str,
    *,
    kind: str | None = None,
) -> torch.Tensor:
    """Sample one of Appendix E.10 matrix families in torch."""
    kinds = ["gaussian", "weights", "singular_values", "kernel", "activation"]
    if kind is None:
        idx = torch.randint(0, len(kinds), (1,), generator=generator).item()
        kind = kinds[int(idx)]

    if kind == "gaussian":
        m = _sample_gaussian_torch(out_dim, in_dim, generator, device)
    elif kind == "weights":
        m = _sample_weights_matrix_torch(out_dim, in_dim, generator, device)
    elif kind == "singular_values":
        m = _sample_singular_values_matrix_torch(out_dim, in_dim, generator, device)
    elif kind == "kernel":
        m = _sample_kernel_matrix_torch(out_dim, in_dim, generator, device)
    elif kind == "activation":
        other_kinds = ["gaussian", "weights", "singular_values", "kernel"]
        idx = torch.randint(0, len(other_kinds), (1,), generator=generator).item()
        m = sample_random_matrix_torch(
            out_dim, in_dim, generator, device, kind=other_kinds[int(idx)]
        )
        m = apply_random_activation_torch(m, generator, with_standardize=False)
        m = m + 1e-3 * torch.randn(m.shape, generator=generator, device=device)
    else:
        raise ValueError(f"Unknown matrix kind: {kind}")

    m = m + 1e-6 * torch.randn(m.shape, generator=generator, device=device)
    return _row_normalize_torch(m)


def sample_random_matrix(
    out_dim: int,
    in_dim: int,
    rng: np.random.Generator,
    *,
    kind: MatrixKind | None = None,
) -> np.ndarray:
    """Sample one of Appendix E.10 matrix families and apply postprocessing."""

    if out_dim <= 0 or in_dim <= 0:
        raise ValueError(f"Matrix dimensions must be > 0. Got out_dim={out_dim}, in_dim={in_dim}")

    kinds = [
        "gaussian",
        "weights",
        "singular_values",
        "kernel",
        "activation",
    ]
    selected = kind or str(rng.choice(kinds))

    if selected == "gaussian":
        m = _sample_gaussian(out_dim, in_dim, rng)
    elif selected == "weights":
        m = _sample_weights_matrix(out_dim, in_dim, rng)
    elif selected == "singular_values":
        m = _sample_singular_values_matrix(out_dim, in_dim, rng)
    elif selected == "kernel":
        m = _sample_kernel_matrix(out_dim, in_dim, rng)
    elif selected == "activation":
        base_kind = typing.cast(
            MatrixKind, rng.choice(["gaussian", "weights", "singular_values", "kernel"])
        )
        m = sample_random_matrix(out_dim, in_dim, rng, kind=base_kind)
        flat = m.reshape(out_dim, in_dim)
        m = apply_random_activation(flat, rng, with_standardize=False)
        m = m + 1e-3 * rng.normal(size=m.shape).astype(np.float32)
    else:
        raise ValueError(f"Unknown matrix kind: {selected}")

    m = m + 1e-6 * rng.normal(size=m.shape).astype(np.float32)
    return _row_normalize(m.astype(np.float32))
