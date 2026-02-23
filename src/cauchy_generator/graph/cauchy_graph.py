"""Appendix E.4 random Cauchy graph sampler."""

from __future__ import annotations

import math

import torch


def sample_cauchy_dag(n_nodes: int, generator: torch.Generator, device: str) -> torch.Tensor:
    """
    Sample a DAG adjacency matrix using Appendix E.4 style probabilities.

    For i < j:
      p_ij = sigmoid(A + B_i + C_j)
    where A, B_i, C_j ~ standard Cauchy.
    """

    if n_nodes < 2:
        return torch.zeros((n_nodes, n_nodes), dtype=torch.bool, device=device)

    # Standard Cauchy: tan(pi * (U - 0.5)) where U ~ Uniform(0, 1)
    u_a = torch.empty(1, device=device).uniform_(0, 1, generator=generator)
    a = torch.tan(math.pi * (u_a - 0.5)).item()

    u_b = torch.empty(n_nodes, device=device).uniform_(0, 1, generator=generator)
    b = torch.tan(math.pi * (u_b - 0.5))

    u_c = torch.empty(n_nodes, device=device).uniform_(0, 1, generator=generator)
    c = torch.tan(math.pi * (u_c - 0.5))

    logits = a + b[:, None] + c[None, :]
    probs = torch.sigmoid(logits)

    upper_mask = torch.triu(
        torch.ones(n_nodes, n_nodes, dtype=torch.bool, device=device), diagonal=1
    )
    edges = torch.empty(n_nodes, n_nodes, device=device).uniform_(0, 1, generator=generator) < probs
    adjacency = edges & upper_mask
    return adjacency
