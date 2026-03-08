"""Multi-input function composition."""

from __future__ import annotations

import torch

from dagzoo.core.execution_semantics import (
    _AGGREGATION_KIND_ORDER,
    sample_multi_source_plan,
)
from dagzoo.core.fixed_layout_batched import FixedLayoutBatchRng, apply_function_plan_batch
from dagzoo.core.fixed_layout_plan_types import ConcatNodeSource, StackedNodeSource
from dagzoo.core.layout_types import AggregationKind, MechanismFamily
from dagzoo.functions._rng_helpers import randint_scalar
from dagzoo.functions.random_functions import apply_random_function
from dagzoo.sampling.noise import NoiseSamplingSpec


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


def _resolved_aggregation_kind(
    aggregation_kind: AggregationKind | None,
    *,
    generator: torch.Generator,
) -> AggregationKind:
    """Resolve the aggregation kind, sampling one when not explicitly provided."""

    if aggregation_kind is not None:
        return aggregation_kind
    idx = randint_scalar(0, len(_AGGREGATION_KIND_ORDER), generator)
    return _AGGREGATION_KIND_ORDER[int(idx)]


def _aggregate_incrementally(
    aggregate: torch.Tensor,
    transformed_output: torch.Tensor,
    *,
    aggregation_kind: AggregationKind,
) -> torch.Tensor:
    """Update an aggregate tensor without materializing a full stack."""

    if aggregation_kind == "sum":
        return aggregate + transformed_output
    if aggregation_kind == "product":
        return aggregate * transformed_output
    if aggregation_kind == "max":
        return torch.maximum(aggregate, transformed_output)
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

    source = sample_multi_source_plan(
        generator,
        parent_count=len(inputs),
        out_dim=out_dim,
        mechanism_logit_tilt=mechanism_logit_tilt,
        function_family_mix=function_family_mix,
        aggregation_kind=aggregation_kind,
    )
    rng = FixedLayoutBatchRng.from_generator(
        generator,
        batch_size=1,
        device=str(inputs[0].device),
    )

    if isinstance(source, ConcatNodeSource):
        concat = torch.cat(inputs, dim=1)
        out = apply_function_plan_batch(
            concat.unsqueeze(0),
            rng,
            source.function,
            out_dim=out_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
        return out.squeeze(0)

    if not isinstance(source, StackedNodeSource):
        raise RuntimeError("Expected a stacked multi-source plan.")

    transformed_outputs = [
        apply_function_plan_batch(
            inp.unsqueeze(0),
            rng,
            source.parent_functions[plan_index],
            out_dim=out_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        ).squeeze(0)
        for plan_index, inp in enumerate(inputs)
    ]
    stacked = torch.stack(transformed_outputs, dim=1)  # (N, parents, out_dim)
    return _aggregate_parent_outputs(stacked, aggregation_kind=source.aggregation_kind)
