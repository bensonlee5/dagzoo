"""Shift runtime parameter resolution for sampling-path integration."""

from __future__ import annotations

from dataclasses import dataclass
import math

from cauchy_generator.config import (
    GeneratorConfig,
    SHIFT_PROFILE_CUSTOM,
    SHIFT_PROFILE_GRAPH_DRIFT,
    SHIFT_PROFILE_MECHANISM_DRIFT,
    SHIFT_PROFILE_MIXED,
    SHIFT_PROFILE_NOISE_DRIFT,
    SHIFT_PROFILE_OFF,
)

_LOG_TWO = math.log(2.0)
_NOISE_VARIANCE_DB_SPAN = _LOG_TWO / 2.0

MECHANISM_FAMILY_ORDER: tuple[str, ...] = (
    "nn",
    "tree",
    "discretization",
    "gp",
    "linear",
    "quadratic",
    "em",
    "product",
)

MECHANISM_FAMILY_BASE_LOGITS: dict[str, float] = {
    "nn": 0.7,
    "tree": 0.7,
    "discretization": 0.5,
    "gp": 0.5,
    "linear": -0.8,
    "quadratic": -0.6,
    "em": -0.3,
    "product": 0.9,
}
NONLINEAR_MECHANISM_FAMILIES: tuple[str, ...] = (
    "nn",
    "tree",
    "discretization",
    "gp",
    "product",
)

_PROFILE_DEFAULT_SCALES: dict[str, tuple[float, float, float]] = {
    SHIFT_PROFILE_OFF: (0.0, 0.0, 0.0),
    SHIFT_PROFILE_GRAPH_DRIFT: (0.5, 0.0, 0.0),
    SHIFT_PROFILE_MECHANISM_DRIFT: (0.0, 0.5, 0.0),
    SHIFT_PROFILE_NOISE_DRIFT: (0.0, 0.0, 0.5),
    SHIFT_PROFILE_MIXED: (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    SHIFT_PROFILE_CUSTOM: (0.0, 0.0, 0.0),
}


@dataclass(slots=True, frozen=True)
class ShiftRuntimeParams:
    """Resolved shift runtime parameters for one generation run."""

    enabled: bool
    profile: str
    graph_scale: float
    mechanism_scale: float
    noise_scale: float
    edge_logit_bias_shift: float
    mechanism_logit_tilt: float
    noise_sigma_multiplier: float


def centered_mechanism_family_logits(families: tuple[str, ...]) -> tuple[float, ...]:
    """Return centered family logits used for mechanism drift sampling."""

    if not families:
        return ()
    raw = tuple(float(MECHANISM_FAMILY_BASE_LOGITS.get(name, 0.0)) for name in families)
    mean = sum(raw) / float(len(raw))
    return tuple(value - mean for value in raw)


def mechanism_family_probabilities(
    *,
    mechanism_logit_tilt: float,
    families: tuple[str, ...] = MECHANISM_FAMILY_ORDER,
) -> dict[str, float]:
    """Resolve mechanism family probabilities for a given tilt value."""

    if not families:
        return {}
    if mechanism_logit_tilt <= 0.0:
        uniform = 1.0 / float(len(families))
        return {family: uniform for family in families}

    centered_logits = centered_mechanism_family_logits(families)
    scaled = [float(mechanism_logit_tilt * logit) for logit in centered_logits]
    max_logit = max(scaled)
    exp_vals = [math.exp(logit - max_logit) for logit in scaled]
    denom = sum(exp_vals)
    return {
        family: (exp_val / denom if denom > 0.0 else 1.0 / float(len(families)))
        for family, exp_val in zip(families, exp_vals, strict=True)
    }


def mechanism_nonlinear_mass(
    *,
    mechanism_logit_tilt: float,
    families: tuple[str, ...] = MECHANISM_FAMILY_ORDER,
    nonlinear_families: tuple[str, ...] = NONLINEAR_MECHANISM_FAMILIES,
) -> float:
    """Return probability mass over nonlinear mechanism families."""

    if not families:
        return 0.0
    probs = mechanism_family_probabilities(
        mechanism_logit_tilt=mechanism_logit_tilt,
        families=families,
    )
    nonlinear_set = set(nonlinear_families)
    return float(sum(prob for family, prob in probs.items() if family in nonlinear_set))


def resolve_shift_runtime_params(config: GeneratorConfig) -> ShiftRuntimeParams:
    """Resolve shift profile/defaults/overrides into runtime coefficients."""

    shift = config.shift
    if not shift.enabled:
        return ShiftRuntimeParams(
            enabled=False,
            profile=SHIFT_PROFILE_OFF,
            graph_scale=0.0,
            mechanism_scale=0.0,
            noise_scale=0.0,
            edge_logit_bias_shift=0.0,
            mechanism_logit_tilt=0.0,
            noise_sigma_multiplier=1.0,
        )

    profile = str(shift.profile)
    default_graph_scale, default_mechanism_scale, default_noise_scale = _PROFILE_DEFAULT_SCALES[
        profile
    ]

    graph_scale = (
        float(shift.graph_scale) if shift.graph_scale is not None else float(default_graph_scale)
    )
    mechanism_scale = (
        float(shift.mechanism_scale)
        if shift.mechanism_scale is not None
        else float(default_mechanism_scale)
    )
    noise_scale = (
        float(shift.noise_scale) if shift.noise_scale is not None else float(default_noise_scale)
    )

    return ShiftRuntimeParams(
        enabled=True,
        profile=profile,
        graph_scale=graph_scale,
        mechanism_scale=mechanism_scale,
        noise_scale=noise_scale,
        edge_logit_bias_shift=float(_LOG_TWO * graph_scale),
        mechanism_logit_tilt=mechanism_scale,
        noise_sigma_multiplier=float(math.exp(_NOISE_VARIANCE_DB_SPAN * noise_scale)),
    )


__all__ = [
    "MECHANISM_FAMILY_BASE_LOGITS",
    "MECHANISM_FAMILY_ORDER",
    "NONLINEAR_MECHANISM_FAMILIES",
    "ShiftRuntimeParams",
    "centered_mechanism_family_logits",
    "mechanism_nonlinear_mass",
    "mechanism_family_probabilities",
    "resolve_shift_runtime_params",
]
