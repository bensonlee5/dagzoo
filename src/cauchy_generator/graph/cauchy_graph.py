"""Appendix E.4 random Cauchy graph sampler."""

from __future__ import annotations

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic transform for edge logits."""

    return 1.0 / (1.0 + np.exp(-x))


def sample_cauchy_dag(n_nodes: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample a DAG adjacency matrix using Appendix E.4 style probabilities.

    For i < j:
      p_ij = sigmoid(A + B_i + C_j)
    where A, B_i, C_j ~ standard Cauchy.
    """

    if n_nodes < 2:
        return np.zeros((n_nodes, n_nodes), dtype=bool)

    a = rng.standard_cauchy()
    b = rng.standard_cauchy(size=n_nodes)
    c = rng.standard_cauchy(size=n_nodes)

    logits = a + b[:, None] + c[None, :]
    probs = _sigmoid(logits)

    upper_mask = np.triu(np.ones((n_nodes, n_nodes), dtype=bool), k=1)
    edges = rng.random((n_nodes, n_nodes)) < probs
    adjacency = edges & upper_mask
    return adjacency
