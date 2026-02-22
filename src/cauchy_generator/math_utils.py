"""Shared math utilities used across the generation pipeline."""

from __future__ import annotations

import numpy as np
import torch


def log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    """Sample from a log-uniform distribution on `[low, high]`."""
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def log_uniform_torch(generator: torch.Generator, low: float, high: float, device: str) -> float:
    """Sample from a log-uniform distribution using torch."""
    low_log = np.log(low)
    high_log = np.log(high)
    u = torch.empty(1, device=device).uniform_(low_log, high_log, generator=generator)
    return float(torch.exp(u).item())


def standardize(x: np.ndarray) -> np.ndarray:
    """Standardize columns with epsilon-protected standard deviations."""
    mu = np.mean(x, axis=0, keepdims=True)
    sigma = np.std(x, axis=0, keepdims=True)
    return (x - mu) / np.clip(sigma, 1e-6, None)


def standardize_torch(x: torch.Tensor) -> torch.Tensor:
    """Standardize columns with epsilon-protected standard deviations in torch."""
    mu = torch.mean(x, dim=0, keepdim=True)
    sigma = torch.std(x, dim=0, keepdim=True)
    return (x - mu) / torch.clamp(sigma, min=1e-6)
