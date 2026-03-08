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


def sample_odt_splits_batch(
    x: torch.Tensor,
    depth: int,
    generator: torch.Generator,
    *,
    feature_probs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample oblivious-tree splits for a batched ``[B, rows, features]`` tensor."""

    if x.dim() != 3:
        raise ValueError(f"x must be rank-3 for batched ODT sampling, got shape={tuple(x.shape)}")

    device = x.device
    batch_size, n_rows, n_feats = x.shape
    if feature_probs is not None:
        probs = feature_probs
        if probs.dim() == 1:
            probs = probs.unsqueeze(0).expand(batch_size, -1)
        if probs.shape != (batch_size, n_feats):
            raise ValueError(
                "feature_probs must have shape (batch_size, n_features) for batched ODT "
                f"sampling, got {tuple(probs.shape)}."
            )
        split_feats = torch.multinomial(probs, depth, replacement=True, generator=generator)
    else:
        split_feats = torch.randint(
            0,
            n_feats,
            (batch_size, depth),
            generator=generator,
            device=device,
        )

    sample_indices = torch.randint(
        0,
        n_rows,
        (batch_size, depth),
        generator=generator,
        device=device,
    )
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)
    thresholds = x[batch_indices, sample_indices, split_feats]
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

    # Evaluate each split predicate per row.
    bits = (x[:, split_feats] > thresholds.unsqueeze(0)).to(torch.int32)

    # Bit weights used to encode each row's leaf index.
    powers = 2 ** torch.arange(depth, device=device, dtype=torch.int32)

    # Pack split outcomes into leaf IDs.
    return (bits * powers.unsqueeze(0)).sum(dim=1).to(torch.long)


def compute_odt_leaf_indices_batch(
    x: torch.Tensor,
    split_feats: torch.Tensor,
    thresholds: torch.Tensor,
) -> torch.Tensor:
    """Compute ODT leaf indices for a batched ``[B, rows, features]`` tensor."""

    if x.dim() != 3:
        raise ValueError(
            f"x must be rank-3 for batched leaf-index evaluation, got shape={tuple(x.shape)}"
        )
    if split_feats.dim() != 2 or thresholds.dim() != 2:
        raise ValueError(
            "split_feats and thresholds must be rank-2 batched tensors for batched ODT evaluation."
        )
    if split_feats.shape != thresholds.shape:
        raise ValueError(
            "split_feats and thresholds must have matching shapes for batched ODT evaluation."
        )
    if split_feats.shape[0] != x.shape[0]:
        raise ValueError(
            "split_feats batch dimension must match x batch dimension for batched ODT evaluation."
        )

    device = x.device
    depth = split_feats.shape[1]
    selected = torch.gather(
        x,
        2,
        split_feats.unsqueeze(1).expand(-1, x.shape[1], -1),
    )
    bits = (selected > thresholds.unsqueeze(1)).to(torch.int32)
    powers = (2 ** torch.arange(depth, device=device, dtype=torch.int32)).view(1, 1, depth)
    return (bits * powers).sum(dim=2).to(torch.long)
