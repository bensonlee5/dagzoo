"""Configurable noise-family sampling primitives."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import operator

import torch

from dagzoo.math_utils import normalize_positive_weights
from dagzoo.config import (
    NOISE_FAMILY_GAUSSIAN,
    NOISE_FAMILY_LAPLACE,
    NOISE_FAMILY_MIXTURE,
    NOISE_FAMILY_STUDENT_T,
    NoiseFamily,
    normalize_noise_family,
)

_MIXTURE_COMPONENTS = (
    NOISE_FAMILY_GAUSSIAN,
    NOISE_FAMILY_LAPLACE,
    NOISE_FAMILY_STUDENT_T,
)
_MAX_TORCH_SEED = (1 << 31) - 1


@dataclass(slots=True, frozen=True)
class NoiseSamplingSpec:
    """Runtime noise selection parameters used by generation paths."""

    family: NoiseFamily = NOISE_FAMILY_GAUSSIAN
    scale: float = 1.0
    student_t_df: float = 5.0
    mixture_weights: Mapping[str, float] | None = None


def _coerce_shape(shape: Sequence[int] | torch.Size) -> tuple[int, ...]:
    """Validate and normalize output shape tuple."""

    parsed_dims: list[int] = []
    for dim in shape:
        if isinstance(dim, bool):
            raise ValueError(f"shape dimensions must be integers, got {shape!r}.")
        try:
            parsed_dim = operator.index(dim)
        except TypeError as exc:
            raise ValueError(f"shape dimensions must be integers, got {shape!r}.") from exc
        parsed_dims.append(int(parsed_dim))
    parsed = tuple(parsed_dims)
    if not parsed:
        raise ValueError("shape must include at least one dimension.")
    if any(dim <= 0 for dim in parsed):
        raise ValueError(f"shape dimensions must be > 0, got {parsed!r}.")
    return parsed


def _validate_positive_finite(name: str, value: float, *, lower_bound: float = 0.0) -> float:
    """Validate a finite float above a strict lower bound."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite value > {lower_bound}, got {value!r}.")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= lower_bound:
        raise ValueError(f"{name} must be a finite value > {lower_bound}, got {value!r}.")
    return parsed


def _validate_nonnegative_finite(name: str, value: float) -> float:
    """Validate a finite float in [0, +inf)."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite value >= 0, got {value!r}.")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be a finite value >= 0, got {value!r}.")
    return parsed


def _normalize_mixture_weights(
    mixture_weights: Mapping[str, float] | None,
) -> dict[str, float]:
    """Validate and normalize optional mixture component weights."""

    if mixture_weights is None:
        uniform = 1.0 / float(len(_MIXTURE_COMPONENTS))
        return {component: uniform for component in _MIXTURE_COMPONENTS}
    if not isinstance(mixture_weights, Mapping):
        raise ValueError("mixture_weights must be a mapping.")
    if not mixture_weights:
        raise ValueError(
            "mixture_weights must include at least one of gaussian, laplace, or student_t."
        )

    parsed: dict[str, float] = {}
    for raw_key, raw_weight in mixture_weights.items():
        if not isinstance(raw_key, str):
            raise ValueError(
                "mixture_weights keys must be strings: gaussian, laplace, or student_t."
            )
        key = raw_key.strip().lower()
        if key not in _MIXTURE_COMPONENTS:
            raise ValueError(
                f"Unsupported mixture_weights key '{raw_key}'. Expected gaussian, laplace, or student_t."
            )
        if key in parsed:
            raise ValueError(f"Duplicate mixture_weights key '{raw_key}' after normalization.")
        weight = _validate_nonnegative_finite(f"mixture_weights.{key}", raw_weight)
        parsed[key] = float(weight)

    return normalize_positive_weights(parsed, field_name="mixture_weights")


def normalize_mixture_weights(
    mixture_weights: Mapping[str, float] | None,
) -> dict[str, float]:
    """Normalize optional mixture component weights for runtime use."""

    return _normalize_mixture_weights(mixture_weights)


def _laplace(shape: tuple[int, ...], *, generator: torch.Generator, device: str) -> torch.Tensor:
    """Sample i.i.d. unit Laplace noise via inverse CDF."""

    u = torch.rand(shape, generator=generator, device=device)
    eps = torch.finfo(u.dtype).eps
    u = torch.clamp(u, min=eps, max=1.0 - eps)
    left = torch.log(2.0 * u)
    right = -torch.log(2.0 * (1.0 - u))
    return torch.where(u < 0.5, left, right)


def _student_t(
    shape: tuple[int, ...],
    *,
    df: float,
    generator: torch.Generator,
    device: str,
) -> torch.Tensor:
    """Sample i.i.d. Student-t noise with backend-safe deterministic fallback."""

    dof = _validate_positive_finite("student_t_df", df, lower_bound=2.0)
    resolved_device = torch.device(device)

    def _sample_with_standard_gamma(
        *,
        local_generator: torch.Generator,
        local_device: str,
    ) -> torch.Tensor:
        z = torch.randn(shape, generator=local_generator, device=local_device)
        alpha = torch.full(
            shape,
            float(dof) / 2.0,
            device=local_device,
            dtype=z.dtype,
        )
        chi2 = 2.0 * torch._standard_gamma(alpha, generator=local_generator)
        denom = torch.sqrt(torch.clamp(chi2 / float(dof), min=torch.finfo(z.dtype).tiny))
        return z / denom

    def _cpu_fallback() -> torch.Tensor:
        local_seed = int(
            torch.randint(
                0,
                _MAX_TORCH_SEED,
                (1,),
                generator=generator,
                device=device,
            ).item()
        )
        cpu_generator = torch.Generator(device="cpu")
        cpu_generator.manual_seed(local_seed)
        return _sample_with_standard_gamma(
            local_generator=cpu_generator,
            local_device="cpu",
        ).to(device=resolved_device)

    if resolved_device.type == "mps":
        return _cpu_fallback()

    try:
        return _sample_with_standard_gamma(
            local_generator=generator,
            local_device=device,
        )
    except NotImplementedError:
        return _cpu_fallback()


def _sample_family(
    *,
    family: NoiseFamily,
    shape: tuple[int, ...],
    generator: torch.Generator,
    device: str,
    student_t_df: float,
) -> torch.Tensor:
    """Sample one non-mixture family."""

    if family == NOISE_FAMILY_GAUSSIAN:
        return torch.randn(shape, generator=generator, device=device)
    if family == NOISE_FAMILY_LAPLACE:
        return _laplace(shape, generator=generator, device=device)
    if family == NOISE_FAMILY_STUDENT_T:
        return _student_t(shape, df=student_t_df, generator=generator, device=device)
    raise ValueError(f"Unsupported noise family for direct sampling: {family!r}.")


def _sample_mixture(
    *,
    shape: tuple[int, ...],
    generator: torch.Generator,
    device: str,
    student_t_df: float,
    mixture_weights: Mapping[str, float] | None,
) -> torch.Tensor:
    """Sample element-wise noise from a configured family mixture."""

    normalized = _normalize_mixture_weights(mixture_weights)
    components = [name for name in _MIXTURE_COMPONENTS if name in normalized]
    probs = torch.tensor(
        [normalized[name] for name in components], device=device, dtype=torch.float32
    )
    numel = math.prod(shape)
    assignments = torch.multinomial(
        probs, num_samples=int(numel), replacement=True, generator=generator
    )

    flat = torch.empty((int(numel),), device=device, dtype=torch.get_default_dtype())
    for idx, component in enumerate(components):
        mask = assignments == idx
        count = int(mask.sum().item())
        if count <= 0:
            continue
        sampled = _sample_family(
            family=component,  # type: ignore[arg-type]
            shape=(count,),
            generator=generator,
            device=device,
            student_t_df=student_t_df,
        )
        flat[mask] = sampled
    return flat.reshape(shape)


def sample_mixture_component_family(
    *,
    generator: torch.Generator,
    device: str,
    mixture_weights: Mapping[str, float] | None = None,
) -> NoiseFamily:
    """Sample one mixture component family according to normalized weights."""

    normalized = _normalize_mixture_weights(mixture_weights)
    components = [name for name in _MIXTURE_COMPONENTS if name in normalized]
    probs = torch.tensor(
        [normalized[name] for name in components], device=device, dtype=torch.float32
    )
    idx = int(torch.multinomial(probs, num_samples=1, replacement=True, generator=generator).item())
    return components[idx]  # type: ignore[return-value]


def sample_noise(
    shape: Sequence[int] | torch.Size,
    *,
    generator: torch.Generator,
    device: str,
    family: NoiseFamily | str = NOISE_FAMILY_GAUSSIAN,
    scale: float = 1.0,
    student_t_df: float = 5.0,
    mixture_weights: Mapping[str, float] | None = None,
) -> torch.Tensor:
    """Sample configurable noise-family draws with deterministic generator semantics."""

    parsed_shape = _coerce_shape(shape)
    normalized_family = normalize_noise_family(family)
    parsed_scale = _validate_positive_finite("scale", scale, lower_bound=0.0)

    if normalized_family == NOISE_FAMILY_MIXTURE:
        base = _sample_mixture(
            shape=parsed_shape,
            generator=generator,
            device=device,
            student_t_df=student_t_df,
            mixture_weights=mixture_weights,
        )
    else:
        if mixture_weights is not None:
            raise ValueError("mixture_weights is only allowed when family is 'mixture'.")
        base = _sample_family(
            family=normalized_family,
            shape=parsed_shape,
            generator=generator,
            device=device,
            student_t_df=student_t_df,
        )

    return base * float(parsed_scale)


def sample_noise_from_spec(
    shape: Sequence[int] | torch.Size,
    *,
    generator: torch.Generator,
    device: str,
    noise_spec: NoiseSamplingSpec | None = None,
    scale_multiplier: float = 1.0,
) -> torch.Tensor:
    """Sample noise using an optional runtime selection spec."""

    parsed_multiplier = _validate_positive_finite(
        "scale_multiplier", scale_multiplier, lower_bound=0.0
    )
    if noise_spec is None:
        return sample_noise(
            shape,
            generator=generator,
            device=device,
            family=NOISE_FAMILY_GAUSSIAN,
            scale=parsed_multiplier,
        )

    return sample_noise(
        shape,
        generator=generator,
        device=device,
        family=noise_spec.family,
        scale=float(noise_spec.scale) * float(parsed_multiplier),
        student_t_df=float(noise_spec.student_t_df),
        mixture_weights=noise_spec.mixture_weights,
    )


__all__ = [
    "NoiseSamplingSpec",
    "normalize_mixture_weights",
    "sample_mixture_component_family",
    "sample_noise",
    "sample_noise_from_spec",
]
