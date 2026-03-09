"""Node-level generation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from dagzoo.core.execution_semantics import sample_node_plan
from dagzoo.core.fixed_layout_plan_types import (
    CategoricalConverterGroup,
    CategoricalConverterPlan,
    FixedLayoutNodePlan,
)
from dagzoo.core.layout_types import ConverterKind, MechanismFamily
from dagzoo.sampling.noise import NoiseSamplingSpec


@dataclass(slots=True)
class ConverterSpec:
    key: str
    kind: ConverterKind
    dim: int
    cardinality: int | None = None


def _standalone_safe_node_plan(node_plan: FixedLayoutNodePlan) -> FixedLayoutNodePlan:
    """Split grouped center-random-fn categorical converters for standalone execution."""

    rewritten_groups = []
    changed = False
    for group in node_plan.converter_groups:
        if not isinstance(group, CategoricalConverterGroup) or len(group.spec_indices) <= 1:
            rewritten_groups.append(group)
            continue
        converter_plan = node_plan.converter_plans[int(group.spec_indices[0])]
        if (
            not isinstance(converter_plan, CategoricalConverterPlan)
            or converter_plan.variant != "center_random_fn"
        ):
            rewritten_groups.append(group)
            continue
        changed = True
        rewritten_groups.extend(
            CategoricalConverterGroup(spec_indices=(int(spec_index),))
            for spec_index in group.spec_indices
        )
    if not changed:
        return node_plan
    return replace(node_plan, converter_groups=tuple(rewritten_groups))


def apply_node_pipeline(
    parent_data: list[torch.Tensor],
    n_rows: int,
    converter_specs: list[ConverterSpec],
    generator: torch.Generator,
    device: str,
    *,
    mechanism_logit_tilt: float = 0.0,
    function_family_mix: dict[MechanismFamily, float] | None = None,
    noise_sigma_multiplier: float = 1.0,
    noise_spec: NoiseSamplingSpec | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply one sampled typed node plan in torch."""
    from dagzoo.core.fixed_layout_batched import FixedLayoutBatchRng, _apply_node_plan_batch

    node_plan = sample_node_plan(
        node_index=0,
        parent_indices=tuple(range(len(parent_data))),
        converter_specs=converter_specs,
        generator=generator,
        device=device,
        mechanism_logit_tilt=mechanism_logit_tilt,
        function_family_mix=function_family_mix,
    )
    node_plan = _standalone_safe_node_plan(node_plan)
    rng = FixedLayoutBatchRng.from_generator(
        generator,
        batch_size=1,
        device=device,
    )
    latent, extracted = _apply_node_plan_batch(
        None,
        node_plan,
        [parent.unsqueeze(0) for parent in parent_data],
        n_rows=n_rows,
        rng=rng,
        device=device,
        noise_sigma_multiplier=noise_sigma_multiplier,
        noise_spec=noise_spec,
    )
    return latent.squeeze(0), {key: value.squeeze(0) for key, value in extracted.items()}
