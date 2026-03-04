"""Device-aware scalar RNG helpers."""

from __future__ import annotations

import torch


def rand_scalar(generator: torch.Generator) -> float:
    """Draw a single uniform [0, 1) float on the generator's device."""
    return torch.rand(1, generator=generator, device=generator.device).item()


def randint_scalar(low: int, high: int, generator: torch.Generator) -> int:
    """Draw a single integer in [low, high) on the generator's device."""
    return int(torch.randint(low, high, (1,), generator=generator, device=generator.device).item())
