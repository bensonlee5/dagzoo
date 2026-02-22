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


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute row-wise softmax with max-shift stabilization."""
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-9, None)


def to_numpy(value: object) -> np.ndarray:
    """Convert tensors or array-like values to NumPy arrays."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def sanitize_json(value: object) -> object:
    """Recursively sanitize metadata for strict JSON serialization."""
    import math

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: sanitize_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(v) for v in value]
    return value
