"""Random function families."""

from __future__ import annotations

import torch

from dagzoo.core.execution_semantics import (
    sample_function_family,
    sample_function_plan,
    sample_function_plan_for_family,
)
from dagzoo.core.fixed_layout_batched import FixedLayoutBatchRng, apply_function_plan_batch
from dagzoo.core.layout_types import MechanismFamily
from dagzoo.math_utils import sanitize_and_standardize
from dagzoo.sampling.noise import NoiseSamplingSpec

# Retained as a stable monkeypatch point for diagnostics tests and audit tools.
_sample_function_family = sample_function_family


def _standardize(x: torch.Tensor) -> torch.Tensor:
    """Standardize columns in torch after clipping non-finite/extreme values."""
    return sanitize_and_standardize(x)


def apply_random_function(
    x: torch.Tensor,
    generator: torch.Generator,
    *,
    out_dim: int | None = None,
    function_type: MechanismFamily | None = None,
    mechanism_logit_tilt: float = 0.0,
    function_family_mix: dict[MechanismFamily, float] | None = None,
    noise_sigma_multiplier: float = 1.0,
    noise_spec: NoiseSamplingSpec | None = None,
) -> torch.Tensor:
    """Apply one sampled typed function plan to `x` in torch."""
    y = x.to(torch.float32)
    if y.dim() == 1:
        y = y.unsqueeze(1)
    y = _standardize(y)

    dout = out_dim if out_dim is not None else y.shape[1]

    if function_type is None:
        plan = sample_function_plan(
            generator,
            out_dim=int(dout),
            mechanism_logit_tilt=mechanism_logit_tilt,
            function_family_mix=function_family_mix,
        )
    else:
        if function_family_mix is not None and function_type not in function_family_mix:
            raise ValueError(
                f"Mechanism family '{function_type}' is not enabled by mechanism.function_family_mix."
            )
        try:
            plan = sample_function_plan_for_family(
                generator,
                family=function_type,
                out_dim=int(dout),
                mechanism_logit_tilt=mechanism_logit_tilt,
                function_family_mix=function_family_mix,
            )
        except ValueError as exc:
            if "Unsupported mechanism family" in str(exc):
                raise ValueError(f"Unknown random function family: {function_type}") from exc
            raise

    rng = FixedLayoutBatchRng.from_generator(
        generator,
        batch_size=1,
        device=str(y.device),
    )
    out = apply_function_plan_batch(
        y.unsqueeze(0),
        rng,
        plan,
        out_dim=int(dout),
        noise_sigma_multiplier=noise_sigma_multiplier,
        noise_spec=noise_spec,
        standardize_input=False,
    )
    return out.squeeze(0)
