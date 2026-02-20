"""Sampling primitives for the prior."""

from .correlated import CorrelatedSampler
from .random_weights import sample_random_weights

__all__ = ["CorrelatedSampler", "sample_random_weights"]
