"""Multi-input function composition."""

from __future__ import annotations

import torch

from cauchy_generator.functions.random_functions import apply_random_function


def apply_multi_function(
    inputs: list[torch.Tensor],
    generator: torch.Generator,
    *,
    out_dim: int,
    mechanism_logit_tilt: float = 0.0,
    noise_sigma_multiplier: float = 1.0,
) -> torch.Tensor:
    """Apply composition across parent node tensors in torch."""
    if not inputs:
        raise ValueError("inputs must be non-empty")
    if len(inputs) == 1:
        return apply_random_function(
            inputs[0],
            generator,
            out_dim=out_dim,
            mechanism_logit_tilt=mechanism_logit_tilt,
            noise_sigma_multiplier=noise_sigma_multiplier,
        )

    if torch.rand(1, generator=generator).item() < 0.5:
        concat = torch.cat(inputs, dim=1)
        return apply_random_function(
            concat,
            generator,
            out_dim=out_dim,
            mechanism_logit_tilt=mechanism_logit_tilt,
            noise_sigma_multiplier=noise_sigma_multiplier,
        )

    transformed = [
        apply_random_function(
            inp,
            generator,
            out_dim=out_dim,
            mechanism_logit_tilt=mechanism_logit_tilt,
            noise_sigma_multiplier=noise_sigma_multiplier,
        )
        for inp in inputs
    ]
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
