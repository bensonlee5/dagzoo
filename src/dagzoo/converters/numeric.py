"""Numeric converter implementations."""

from __future__ import annotations

import torch

from dagzoo.math_utils import log_uniform as _log_uniform


def _minmax(x: torch.Tensor) -> torch.Tensor:
    """Min-max scale columns in torch."""
    lo = torch.min(x, dim=0, keepdim=True).values
    hi = torch.max(x, dim=0, keepdim=True).values
    return (x - lo) / torch.clamp(hi - lo, min=1e-6)


def apply_numeric_converter(
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
    a = _log_uniform(generator, 0.2, 5.0, device)
    b = _log_uniform(generator, 0.2, 5.0, device)

    scaled = _minmax(y)
    warped = 1.0 - torch.pow(1.0 - torch.pow(torch.clamp(scaled, 0.0, 1.0), a), b)
    return warped, v
