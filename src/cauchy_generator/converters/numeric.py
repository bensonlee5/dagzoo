"""Numeric converter implementations (Appendix E.6)."""

from __future__ import annotations

import numpy as np
import torch

from cauchy_generator.math_utils import (
    log_uniform as _log_uniform,
    log_uniform_torch as _log_uniform_torch,
)


def _minmax(x: np.ndarray) -> np.ndarray:
    """Min-max scale columns to `[0, 1]` with numerical safeguards."""

    lo = np.min(x, axis=0, keepdims=True)
    hi = np.max(x, axis=0, keepdims=True)
    return (x - lo) / np.clip(hi - lo, 1e-6, None)


def _minmax_torch(x: torch.Tensor) -> torch.Tensor:
    """Min-max scale columns in torch."""
    lo = torch.min(x, dim=0, keepdim=True).values
    hi = torch.max(x, dim=0, keepdim=True).values
    return (x - lo) / torch.clamp(hi - lo, min=1e-6)


def apply_numeric_converter_torch(
    x: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Numeric converter in torch."""
    y = x.to(torch.float32)
    if y.dim() == 1:
        y = y.unsqueeze(1)

    v = y[:, 0].clone()
    if torch.rand(1, generator=generator).item() < 0.5:
        return y, v

    device = str(y.device)
    a = _log_uniform_torch(generator, 0.2, 5.0, device)
    b = _log_uniform_torch(generator, 0.2, 5.0, device)

    scaled = _minmax_torch(y)
    warped = 1.0 - torch.pow(1.0 - torch.pow(torch.clamp(scaled, 0.0, 1.0), a), b)
    return warped, v


def apply_numeric_converter(
    x: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Numeric converter.

    Returns:
    - transformed node slice x'
    - extracted numeric column v
    """

    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]

    v = arr[:, 0].astype(np.float32)
    if rng.random() < 0.5:
        return arr, v

    # Appendix E.6 note: Kumaraswamy warping is applied to x' rather than v.
    a = _log_uniform(rng, 0.2, 5.0)
    b = _log_uniform(rng, 0.2, 5.0)
    scaled = _minmax(arr)
    warped = 1.0 - np.power(1.0 - np.power(np.clip(scaled, 0.0, 1.0), a), b)
    return warped.astype(np.float32), v
