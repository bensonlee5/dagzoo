"""Multi-input function composition."""

from __future__ import annotations

import torch

from dagzoo.core.layout_types import AggregationKind, MechanismFamily
from dagzoo.functions.random_functions import apply_random_function
from dagzoo.sampling.noise import NoiseSamplingSpec

_AGGREGATION_KIND_ORDER: tuple[AggregationKind, ...] = ("sum", "product", "max", "logsumexp")


def _aggregate_parent_outputs(
    stacked: torch.Tensor,
    *,
    aggregation_kind: AggregationKind,
) -> torch.Tensor:
    """Aggregate transformed parent outputs using one explicit aggregation kind."""

    if aggregation_kind == "sum":
        return torch.sum(stacked, dim=1)
    if aggregation_kind == "product":
        return torch.prod(stacked, dim=1)
    if aggregation_kind == "max":
        return torch.max(stacked, dim=1).values
    if aggregation_kind == "logsumexp":
        return torch.logsumexp(stacked, dim=1)
    raise ValueError(f"Unknown aggregation kind: {aggregation_kind!r}")


def apply_multi_function(
    inputs: list[torch.Tensor],
    generator: torch.Generator,
    *,
    out_dim: int,
    aggregation_kind: AggregationKind | None = None,
    mechanism_logit_tilt: float = 0.0,
    function_family_mix: dict[MechanismFamily, float] | None = None,
    noise_sigma_multiplier: float = 1.0,
    noise_spec: NoiseSamplingSpec | None = None,
) -> torch.Tensor:
    """Apply composition across parent node tensors in torch.

    ``aggregation_kind`` only takes effect when the stacking composition path
    is selected (50 % chance).  When the concatenation path is taken, inputs
    are concatenated and passed through a single ``apply_random_function``,
    so the aggregation parameter is not used.
    """
    if not inputs:
        raise ValueError("inputs must be non-empty")
    if len(inputs) == 1:
        return apply_random_function(
            inputs[0],
            generator,
            out_dim=out_dim,
            mechanism_logit_tilt=mechanism_logit_tilt,
            function_family_mix=function_family_mix,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )

    if torch.rand(1, generator=generator).item() < 0.5:
        concat = torch.cat(inputs, dim=1)
        return apply_random_function(
            concat,
            generator,
            out_dim=out_dim,
            mechanism_logit_tilt=mechanism_logit_tilt,
            function_family_mix=function_family_mix,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )

    transformed = [
        apply_random_function(
            inp,
            generator,
            out_dim=out_dim,
            mechanism_logit_tilt=mechanism_logit_tilt,
            function_family_mix=function_family_mix,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
        for inp in inputs
    ]
    stacked = torch.stack(transformed, dim=1)  # (N, parents, out_dim)

    resolved_aggregation_kind = aggregation_kind
    if resolved_aggregation_kind is None:
        idx = torch.randint(0, len(_AGGREGATION_KIND_ORDER), (1,), generator=generator).item()
        resolved_aggregation_kind = _AGGREGATION_KIND_ORDER[int(idx)]
    return _aggregate_parent_outputs(stacked, aggregation_kind=resolved_aggregation_kind)
