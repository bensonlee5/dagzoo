"""Numeric converter implementations."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from dagzoo.core.execution_semantics import sample_converter_plan
from dagzoo.core.fixed_layout_batched import FixedLayoutBatchRng, apply_numeric_converter_plan_batch
from dagzoo.core.fixed_layout_plan_types import NumericConverterPlan


def apply_numeric_converter(
    x: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Numeric converter in torch via the shared typed-plan semantics."""
    y = x.to(torch.float32)
    if y.dim() == 1:
        y = y.unsqueeze(1)

    plan = sample_converter_plan(
        generator,
        SimpleNamespace(key="value", kind="num", dim=int(y.shape[1]), cardinality=None),
        mechanism_logit_tilt=0.0,
        function_family_mix=None,
    )
    if not isinstance(plan, NumericConverterPlan):
        raise RuntimeError("Expected numeric converter plan for numeric converter.")

    rng = FixedLayoutBatchRng.from_generator(
        generator,
        batch_size=1,
        device=str(y.device),
    )
    x_prime, values = apply_numeric_converter_plan_batch(
        y.unsqueeze(0),
        rng,
        plan,
    )
    return x_prime.squeeze(0), values.squeeze(0)
