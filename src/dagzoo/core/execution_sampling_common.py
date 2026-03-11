"""Shared sampling utility helpers for fixed-layout execution semantics."""

from __future__ import annotations

import torch

from dagzoo.rng import KeyedRng, keyed_rng_from_generator


def _rand_scalar(generator: torch.Generator) -> float:
    return torch.rand(1, generator=generator, device=generator.device).item()


def _randint_scalar(low: int, high: int, generator: torch.Generator) -> int:
    return int(torch.randint(low, high, (1,), generator=generator, device=generator.device).item())


def _generator_device(generator: torch.Generator) -> str:
    return str(generator.device)


def _resolve_sampling_device(
    *,
    generator: torch.Generator | None,
    device: str | None,
) -> str:
    if device is not None:
        return str(device)
    if generator is not None:
        return _generator_device(generator)
    return "cpu"


def _resolve_sampling_generator(
    *,
    generator: torch.Generator | None,
    keyed_rng: KeyedRng | None,
    device: str | None,
) -> tuple[torch.Generator, str]:
    resolved_device = _resolve_sampling_device(generator=generator, device=device)
    if generator is not None:
        return generator, resolved_device
    if keyed_rng is None:
        raise TypeError("Either generator or keyed_rng must be provided.")
    return keyed_rng.torch_rng(device=resolved_device), resolved_device


def _resolve_sampling_root(
    *,
    generator: torch.Generator | None,
    keyed_rng: KeyedRng | None,
    device: str | None,
    namespace: str,
) -> tuple[KeyedRng, str]:
    resolved_device = _resolve_sampling_device(generator=generator, device=device)
    if keyed_rng is not None:
        return keyed_rng, resolved_device
    if generator is None:
        raise TypeError("Either generator or keyed_rng must be provided.")
    return keyed_rng_from_generator(generator, namespace), resolved_device


def _sample_bool(generator: torch.Generator, *, p: float = 0.5) -> bool:
    return bool(_rand_scalar(generator) < p)


__all__ = [
    "_generator_device",
    "_rand_scalar",
    "_randint_scalar",
    "_resolve_sampling_device",
    "_resolve_sampling_generator",
    "_resolve_sampling_root",
    "_sample_bool",
]
