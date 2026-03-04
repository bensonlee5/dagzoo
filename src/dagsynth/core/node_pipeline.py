"""Node-level generation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import re

import torch

from dagsynth.converters.categorical import apply_categorical_converter
from dagsynth.converters.numeric import apply_numeric_converter
from dagsynth.core.layout_types import ConverterKind, MechanismFamily
from dagsynth.functions.multi import apply_multi_function
from dagsynth.math_utils import (
    log_uniform as _log_uniform,
    standardize as _standardize,
)
from dagsynth.sampling.noise import NoiseSamplingSpec, sample_noise_from_spec
from dagsynth.sampling.random_points import sample_random_points
from dagsynth.sampling.random_weights import sample_random_weights

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
    """Apply node transform in torch."""
    _, _, total_dim = _resolve_latent_dimensions(
        converter_specs,
        generator,
        device,
    )

    if parent_data:
        latent = apply_multi_function(
            parent_data,
            generator,
            out_dim=total_dim,
            mechanism_logit_tilt=mechanism_logit_tilt,
            function_family_mix=function_family_mix,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
    else:
        latent = sample_random_points(
            n_rows,
            total_dim,
            generator,
            device,
            mechanism_logit_tilt=mechanism_logit_tilt,
            function_family_mix=function_family_mix,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )

    latent = torch.nan_to_num(latent.to(torch.float32), nan=0.0, posinf=1e6, neginf=-1e6)
    latent = torch.clamp(latent, -1e6, 1e6)
    latent = _standardize(latent)

    w = sample_random_weights(
        latent.shape[1],
        generator,
        device,
        sigma_multiplier=noise_sigma_multiplier,
        noise_spec=noise_spec,
    )
    latent = latent * w.unsqueeze(0)

    mean_l2 = torch.mean(torch.norm(latent, dim=1))
    latent = latent / torch.clamp(mean_l2, min=1e-6)

    extracted: dict[str, torch.Tensor] = {}
    column_cursor = 0
    for spec in converter_specs:
        spec_dim = max(1, int(spec.dim))
        latent = _pad_latent_columns(
            latent,
            min_required_dim=column_cursor + spec_dim,
            generator=generator,
            device=device,
            noise_spec=noise_spec,
        )
        view = latent[:, column_cursor : column_cursor + spec_dim]
        if spec.kind == "cat":
            if spec.cardinality is None:
                raise ValueError(f"Missing cardinality for categorical spec: {spec.key}")
            x_prime, v = apply_categorical_converter(
                view,
                generator,
                n_categories=int(spec.cardinality),
                function_family_mix=function_family_mix,
            )
            extracted[spec.key] = v
        elif spec.kind == "num":
            x_prime, v = apply_numeric_converter(view[:, :1], generator)
            extracted[spec.key] = v
        elif spec.kind == "target_cls":
            cls = max(2, int(spec.cardinality or 2))
            x_prime, v = apply_categorical_converter(
                view,
                generator,
                n_categories=cls,
                function_family_mix=function_family_mix,
            )
            extracted[spec.key] = v
        elif spec.kind == "target_reg":
            x_prime, v = apply_numeric_converter(view[:, :1], generator)
            extracted[spec.key] = v
        else:
            raise ValueError(f"Unknown converter kind: {spec.kind}")

        if x_prime.shape[1] != spec_dim:
            if x_prime.shape[1] > spec_dim:
                x_prime = x_prime[:, :spec_dim]
            else:
                x_prime = torch.nn.functional.pad(x_prime, (0, spec_dim - x_prime.shape[1]))
        latent[:, column_cursor : column_cursor + spec_dim] = x_prime
        column_cursor += spec_dim

    scale = _log_uniform(generator, 0.1, 10.0, device)
    latent *= scale
    return latent, extracted
