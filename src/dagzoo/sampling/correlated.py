"""Correlated scalar and categorical sampling."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from dagzoo.rng import KeyedRng


@dataclass(slots=True)
class _NumericParams:
    alpha: float
    beta: float


class CorrelatedSampler:
    """Name-keyed sampler with shared latent parameters per variable name."""

    def __init__(self, keyed_rng: KeyedRng, device: str):
        """Initialize correlated sampler state for one dataset generation run."""

        self._keyed_rng = keyed_rng
        self._device = device
        self._numeric_params: dict[str, _NumericParams] = {}
        self._categorical_weights: dict[tuple[str, int], torch.Tensor] = {}
        self._numeric_draw_counts: dict[str, int] = {}
        self._categorical_draw_counts: dict[tuple[str, int], int] = {}

    def _get_numeric_params(self, name: str) -> _NumericParams:
        """Create/retrieve latent beta parameters shared by a scalar variable name."""

        if name not in self._numeric_params:
            generator = self._keyed_rng.keyed("numeric_params", name).torch_rng(device=self._device)
            t = torch.empty(1, device=self._device).uniform_(0.0, 1.0, generator=generator).item()
            log_s = (
                torch.empty(1, device=self._device)
                .uniform_(math.log(0.1), math.log(10_000.0), generator=generator)
                .item()
            )
            s = math.exp(log_s)
            self._numeric_params[name] = _NumericParams(
                alpha=float(s * t), beta=float(s * (1.0 - t))
            )
        return self._numeric_params[name]

    def _sample_beta(self, alpha: float, beta: float, *, name: str) -> float:
        """Sample a scalar from Beta(alpha, beta) using keyed per-name draws."""

        draw_index = int(self._numeric_draw_counts.get(name, 0))
        self._numeric_draw_counts[name] = draw_index + 1
        local_generator = self._keyed_rng.keyed(
            "numeric_draw",
            name,
            draw_index,
        ).torch_rng(device="cpu")
        concentration = torch.tensor([alpha, beta], dtype=torch.float64, device="cpu")
        probs = torch._sample_dirichlet(concentration, generator=local_generator)
        return float(probs[0].item())

    def _categorical_draw_generator(self, name: str, n_categories: int) -> torch.Generator:
        key = (name, int(n_categories))
        draw_index = int(self._categorical_draw_counts.get(key, 0))
        self._categorical_draw_counts[key] = draw_index + 1
        return self._keyed_rng.keyed(
            "categorical_draw",
            name,
            int(n_categories),
            draw_index,
        ).torch_rng(device=self._device)

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
        u = self._sample_beta(params.alpha, params.beta, name=name)
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
            generator = self._keyed_rng.keyed(
                "categorical_weights",
                name,
                int(n_categories),
            ).torch_rng(device=self._device)
            raw = (
                torch.empty(n_categories, device=self._device).uniform_(0, 1, generator=generator)
                + 1e-6
            )
            self._categorical_weights[key] = raw / raw.sum()
        probs = self._categorical_weights[key]
        return int(
            torch.multinomial(
                probs,
                1,
                generator=self._categorical_draw_generator(name, n_categories),
            ).item()
        )
