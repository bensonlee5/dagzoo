"""Noise-runtime resolution helpers for dataset generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dagzoo.config import (
    NOISE_FAMILY_MIXTURE,
    GeneratorConfig,
    NoiseFamily,
)
from dagzoo.rng import SeedManager
from dagzoo.sampling.noise import (
    NoiseSamplingSpec,
    normalize_mixture_weights,
    sample_mixture_component_family,
)


@dataclass(slots=True, frozen=True)
class NoiseRuntimeSelection:
    """Resolved per-dataset noise-family runtime selection."""

    family_requested: NoiseFamily
    family_sampled: NoiseFamily
    sampling_strategy: str
    base_scale: float
    student_t_df: float
    mixture_weights: dict[str, float] | None = None


def _resolve_noise_runtime_selection(
    config: GeneratorConfig,
    *,
    run_seed: int,
) -> NoiseRuntimeSelection:
    """Resolve deterministic per-dataset noise-family selection."""

    family_requested = config.noise.family
    base_scale = float(config.noise.base_scale)
    student_t_df = float(config.noise.student_t_df)
    if family_requested != NOISE_FAMILY_MIXTURE:
        return NoiseRuntimeSelection(
            family_requested=family_requested,
            family_sampled=family_requested,
            sampling_strategy="dataset_level",
            base_scale=base_scale,
            student_t_df=student_t_df,
            mixture_weights=None,
        )

    mixture_weights_raw = (
        {str(key): float(value) for key, value in config.noise.mixture_weights.items()}
        if config.noise.mixture_weights is not None
        else None
    )
    normalized_weights = normalize_mixture_weights(mixture_weights_raw)
    selector = SeedManager(run_seed).torch_rng("noise_family", device="cpu")
    sampled_family = sample_mixture_component_family(
        generator=selector,
        device="cpu",
        mixture_weights=normalized_weights,
    )
    return NoiseRuntimeSelection(
        family_requested=family_requested,
        family_sampled=sampled_family,
        sampling_strategy="dataset_level",
        base_scale=base_scale,
        student_t_df=student_t_df,
        mixture_weights={key: float(value) for key, value in normalized_weights.items()},
    )


def _noise_sampling_spec(selection: NoiseRuntimeSelection) -> NoiseSamplingSpec:
    """Build a concrete noise sampling spec from runtime selection."""

    return NoiseSamplingSpec(
        family=selection.family_sampled,
        scale=float(selection.base_scale),
        student_t_df=float(selection.student_t_df),
    )


def _build_noise_distribution_metadata(selection: NoiseRuntimeSelection) -> dict[str, Any]:
    """Build per-dataset noise-distribution metadata payload."""

    return {
        "family_requested": str(selection.family_requested),
        "family_sampled": str(selection.family_sampled),
        "sampling_strategy": str(selection.sampling_strategy),
        "base_scale": float(selection.base_scale),
        "student_t_df": float(selection.student_t_df),
        "mixture_weights": (
            {key: float(value) for key, value in selection.mixture_weights.items()}
            if selection.mixture_weights is not None
            else None
        ),
    }
