"""Node-level generation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import re

import torch

from dagzoo.core.execution_semantics import sample_node_plan
from dagzoo.core.layout_types import ConverterKind, MechanismFamily
from dagzoo.math_utils import log_uniform as _log_uniform
from dagzoo.sampling.noise import NoiseSamplingSpec, sample_noise_from_spec

_FEATURE_KEY_PATTERN = re.compile(r"^feature_(\d+)$")


@dataclass(slots=True)
class ConverterSpec:
    key: str
    kind: ConverterKind
    dim: int
    cardinality: int | None = None


def parse_feature_key(key: str) -> int | None:
    """Parse `feature_{index}` key format and return its index when valid."""

    match = _FEATURE_KEY_PATTERN.fullmatch(key)
    if match is None:
        return None
    return int(match.group(1))


def _resolve_latent_dimensions(
    converter_specs: list[ConverterSpec],
    generator: torch.Generator,
    device: str,
) -> tuple[int, int, int]:
    """Resolve required and total latent dimensions for one node execution."""

    required_dim = int(sum(max(1, spec.dim) for spec in converter_specs))
    sampled_latent_extra = max(1, int(_log_uniform(generator, 1.0, 32.0, device)))
    total_dim = required_dim + sampled_latent_extra
    return required_dim, sampled_latent_extra, total_dim


def _pad_latent_columns(
    latent: torch.Tensor,
    *,
    min_required_dim: int,
    generator: torch.Generator,
    device: str,
    noise_spec: NoiseSamplingSpec | None,
) -> torch.Tensor:
    """Pad latent tensor with sampled noise when converter slices exceed latent width."""

    current_dim = int(latent.shape[1])
    if min_required_dim <= current_dim:
        return latent
    missing_dim = min_required_dim - current_dim
    pad = sample_noise_from_spec(
        (latent.shape[0], missing_dim),
        generator=generator,
        device=device,
        noise_spec=noise_spec,
    )
    return torch.cat([latent, pad], dim=1)


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
