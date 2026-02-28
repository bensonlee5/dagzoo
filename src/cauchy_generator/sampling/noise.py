"""Configurable noise-family sampling primitives."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import torch

from cauchy_generator.config import (
    NOISE_FAMILY_GAUSSIAN,
    NOISE_FAMILY_LAPLACE,
    NOISE_FAMILY_LEGACY,
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


def _coerce_shape(shape: Sequence[int] | torch.Size) -> tuple[int, ...]:
    """Validate and normalize output shape tuple."""

    parsed = tuple(int(dim) for dim in shape)
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
        weight = _validate_nonnegative_finite(f"mixture_weights.{key}", float(raw_weight))
        parsed[key] = float(weight)

    total = float(sum(parsed.values()))
    if total <= 0.0:
        raise ValueError("mixture_weights must have a positive total weight.")
    return {key: value / total for key, value in parsed.items() if value > 0.0}


def _laplace(shape: tuple[int, ...], *, generator: torch.Generator, device: str) -> torch.Tensor:
    """Sample i.i.d. unit Laplace noise via inverse CDF."""

    u = torch.empty(shape, device=device).uniform_(-0.5, 0.5, generator=generator)
    return -torch.sign(u) * torch.log1p(-2.0 * torch.abs(u))


def _student_t(
    shape: tuple[int, ...],
    *,
    df: float,
    generator: torch.Generator,
    device: str,
) -> torch.Tensor:
    """Sample i.i.d. Student-t noise with generator-seeded forked RNG."""

    dof = _validate_positive_finite("student_t_df", df, lower_bound=2.0)
    local_seed = int(
        torch.randint(0, _MAX_TORCH_SEED, (1,), generator=generator, device=device).item()
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(local_seed)
        distribution = torch.distributions.StudentT(
            df=torch.tensor(dof, dtype=torch.float32),
            loc=torch.tensor(0.0, dtype=torch.float32),
            scale=torch.tensor(1.0, dtype=torch.float32),
        )
        samples = distribution.sample(torch.Size(shape))
    return samples.to(device=device, dtype=torch.float32)


def _sample_family(
    *,
    family: NoiseFamily,
    shape: tuple[int, ...],
    generator: torch.Generator,
    device: str,
    student_t_df: float,
) -> torch.Tensor:
    """Sample one non-mixture family."""

    if family in (NOISE_FAMILY_LEGACY, NOISE_FAMILY_GAUSSIAN):
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
    components = list(normalized.keys())
    probs = torch.tensor(
        [normalized[name] for name in components], device=device, dtype=torch.float32
    )
    numel = math.prod(shape)
    assignments = torch.multinomial(
        probs, num_samples=int(numel), replacement=True, generator=generator
    )

    flat = torch.empty((int(numel),), device=device, dtype=torch.float32)
    for idx, component in enumerate(components):
        mask = assignments == idx
        count = int(mask.sum().item())
        if count <= 0:
            continue
        flat[mask] = _sample_family(
            family=component,  # type: ignore[arg-type]
            shape=(count,),
            generator=generator,
            device=device,
            student_t_df=student_t_df,
        )
    return flat.reshape(shape)


def sample_noise(
    shape: Sequence[int] | torch.Size,
    *,
    generator: torch.Generator,
    device: str,
    family: NoiseFamily | str = NOISE_FAMILY_LEGACY,
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
