"""Appendix E.2 correlated scalar and categorical sampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class _NumericParams:
    alpha: float
    beta: float


class CorrelatedSampler:
    """Name-keyed sampler with shared latent parameters per variable name."""

    def __init__(self, rng: np.random.Generator):
        """Initialize correlated sampler state for one dataset generation run."""

        self._rng = rng
        self._numeric_params: dict[str, _NumericParams] = {}
        self._categorical_weights: dict[tuple[str, int], np.ndarray] = {}

    def _get_numeric_params(self, name: str) -> _NumericParams:
        """Create/retrieve latent beta parameters shared by a scalar variable name."""

        if name not in self._numeric_params:
            t = self._rng.uniform(0.0, 1.0)
            s = np.exp(self._rng.uniform(np.log(0.1), np.log(10_000.0)))
            self._numeric_params[name] = _NumericParams(alpha=float(s * t), beta=float(s * (1.0 - t)))
        return self._numeric_params[name]

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
        u = self._rng.beta(params.alpha, params.beta)
        if log_scale:
            value = np.exp(np.log(low) + u * (np.log(high) - np.log(low)))
        else:
            value = low + u * (high - low)
        if as_int:
            return int(np.floor(value))
        return float(value)

    def sample_category(self, name: str, n_categories: int) -> int:
        """Sample a correlated categorical value for a named variable."""

        key = (name, n_categories)
        if key not in self._categorical_weights:
            raw = self._rng.random(size=n_categories) + 1e-6
            self._categorical_weights[key] = raw / raw.sum()
        probs = self._categorical_weights[key]
        return int(self._rng.choice(np.arange(n_categories), p=probs))
