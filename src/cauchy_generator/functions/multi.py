"""Multi-input function composition (Appendix E.7)."""

from __future__ import annotations

import numpy as np
import torch

from cauchy_generator.functions.random_functions import (
    apply_random_function,
    apply_random_function_torch,
)


def _logsumexp(x: np.ndarray, axis: int) -> np.ndarray:
    """Numerically stable log-sum-exp reduction."""

    m = np.max(x, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))).squeeze(axis=axis)


def apply_multi_function_torch(
    inputs: list[torch.Tensor],
    generator: torch.Generator,
    *,
    out_dim: int,
) -> torch.Tensor:
    """Apply Appendix E.7 style composition across parent node tensors in torch."""
    if not inputs:
        raise ValueError("inputs must be non-empty")
    if len(inputs) == 1:
        return apply_random_function_torch(inputs[0], generator, out_dim=out_dim)

    if torch.rand(1, generator=generator).item() < 0.5:
        concat = torch.cat(inputs, dim=1)
        return apply_random_function_torch(concat, generator, out_dim=out_dim)

    transformed = [apply_random_function_torch(inp, generator, out_dim=out_dim) for inp in inputs]
    stacked = torch.stack(transformed, dim=1)  # (N, parents, out_dim)

    aggs = ["sum", "product", "max", "logsumexp"]
    idx = torch.randint(0, len(aggs), (1,), generator=generator).item()
    agg = aggs[int(idx)]

    if agg == "sum":
        return torch.sum(stacked, dim=1)
    if agg == "product":
        return torch.prod(stacked, dim=1)
    if agg == "max":
        return torch.max(stacked, dim=1).values
    return torch.logsumexp(stacked, dim=1)


def apply_multi_function(
    inputs: list[np.ndarray],
    rng: np.random.Generator,
    *,
    out_dim: int,
) -> np.ndarray:
    """Apply Appendix E.7 style composition across parent node tensors."""

    if not inputs:
        raise ValueError("inputs must be non-empty")
    if len(inputs) == 1:
        return apply_random_function(inputs[0], rng, out_dim=out_dim)

    if rng.random() < 0.5:
        concat = np.concatenate(inputs, axis=1).astype(np.float32)
        return apply_random_function(concat, rng, out_dim=out_dim)

    transformed = [apply_random_function(inp, rng, out_dim=out_dim) for inp in inputs]
    stacked = np.stack(transformed, axis=1)
    agg = str(rng.choice(["sum", "product", "max", "logsumexp"]))
    if agg == "sum":
        return np.sum(stacked, axis=1).astype(np.float32)
    if agg == "product":
        return np.prod(stacked, axis=1).astype(np.float32)
    if agg == "max":
        return np.max(stacked, axis=1).astype(np.float32)
    return _logsumexp(stacked, axis=1).astype(np.float32)
