"""Correlated scalar and categorical sampling."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(slots=True)
class _NumericParams:
    alpha: float
    beta: float


class CorrelatedSampler:
    """Name-keyed sampler with shared latent parameters per variable name."""

    def __init__(self, generator: torch.Generator, device: str):
        """Initialize correlated sampler state for one dataset generation run."""

        self._generator = generator
        self._device = device
        self._numeric_params: dict[str, _NumericParams] = {}
        self._categorical_weights: dict[tuple[str, int], torch.Tensor] = {}

    def _get_numeric_params(self, name: str) -> _NumericParams:
        """Create/retrieve latent beta parameters shared by a scalar variable name."""

        if name not in self._numeric_params:
            t = (
                torch.empty(1, device=self._device)
                .uniform_(0.0, 1.0, generator=self._generator)
                .item()
            )
            log_s = (
                torch.empty(1, device=self._device)
                .uniform_(math.log(0.1), math.log(10_000.0), generator=self._generator)
                .item()
            )
            s = math.exp(log_s)
            self._numeric_params[name] = _NumericParams(
                alpha=float(s * t), beta=float(s * (1.0 - t))
            )
        return self._numeric_params[name]

    def _sample_beta(self, alpha: float, beta: float) -> float:
        """Sample a scalar from Beta(alpha, beta) using generator-derived seed."""

        # Keep beta draws deterministic without touching process-wide RNG state.
        # We derive one seed from the run generator and sample with a local
        # CPU generator, so this method consumes one parent RNG draw per call.
        seed = int(
            torch.empty(1, dtype=torch.int64, device=self._device)
            .random_(generator=self._generator)
            .item()
        )
        local_generator = torch.Generator(device="cpu")
        local_generator.manual_seed(seed & 0x7FFFFFFF)
        concentration = torch.tensor([alpha, beta], dtype=torch.float64, device="cpu")
        probs = torch._sample_dirichlet(concentration, generator=local_generator)
        return float(probs[0].item())

    def sample_num(
        self,
        name: str,
        low: float,
        high: float,
        *,
        log_scale: bool = False,
        as_int: bool = False,
    ) -> float | int:
        """Sample a correlated scalar for a named variable."""

        params = self._get_numeric_params(name)
        u = self._sample_beta(params.alpha, params.beta)
        if log_scale:
            value = math.exp(math.log(low) + u * (math.log(high) - math.log(low)))
        else:
            value = low + u * (high - low)
        if as_int:
            return int(math.floor(value))
        return float(value)

    def sample_category(self, name: str, n_categories: int) -> int:
        """Sample a correlated categorical value for a named variable."""

        key = (name, n_categories)
        if key not in self._categorical_weights:
            raw = (
                torch.empty(n_categories, device=self._device).uniform_(
                    0, 1, generator=self._generator
                )
                + 1e-6
            )
            self._categorical_weights[key] = raw / raw.sum()
        probs = self._categorical_weights[key]
        return int(torch.multinomial(probs, 1, generator=self._generator).item())
