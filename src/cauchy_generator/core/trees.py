"""Shared utilities for Torch-native Oblivious Decision Trees."""

from __future__ import annotations

import torch


def sample_odt_splits(
    x: torch.Tensor,
    depth: int,
    generator: torch.Generator,
    *,
    feature_probs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample features and thresholds for an oblivious decision tree.

    Returns:
        split_feats: (depth,) tensor of feature indices.
        thresholds: (depth,) tensor of threshold values.
    """
    device = x.device
    n_rows, n_feats = x.shape

    if feature_probs is not None:
        split_feats = torch.multinomial(feature_probs, depth, replacement=True, generator=generator)
    else:
        split_feats = torch.randint(0, n_feats, (depth,), generator=generator, device=device)

    sample_indices = torch.randint(0, n_rows, (depth,), generator=generator, device=device)
    thresholds = x[sample_indices, split_feats]

    return split_feats, thresholds


def compute_odt_leaf_indices(
    x: torch.Tensor,
    split_feats: torch.Tensor,
    thresholds: torch.Tensor,
) -> torch.Tensor:
    """
    Compute leaf indices for all rows in x given oblivious splits.

    Returns:
        leaf_idx: (n_rows,) long tensor in [0, 2^depth - 1].
    """
    device = x.device
    depth = split_feats.shape[0]

    # bits: (n_rows, depth)
    bits = (x[:, split_feats] > thresholds.unsqueeze(0)).to(torch.int32)

    # powers: (depth,)
    powers = 2 ** torch.arange(depth, device=device, dtype=torch.int32)

    # leaf_idx: (n_rows,)
    return (bits * powers.unsqueeze(0)).sum(dim=1).to(torch.long)
