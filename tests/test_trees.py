"""Tests for core/trees.py — ODT split and leaf index utilities."""

import torch

from dagzoo.core.trees import (
    compute_odt_leaf_indices,
    compute_odt_leaf_indices_batch,
    sample_odt_splits,
    sample_odt_splits_batch,
)
from conftest import make_generator as _make_generator


def test_sample_odt_splits_shapes() -> None:
    g = _make_generator()
    x = torch.randn(64, 6, generator=g)
    depth = 3
    split_feats, thresholds = sample_odt_splits(x, depth, g)
    assert split_feats.shape == (depth,)
    assert thresholds.shape == (depth,)


def test_split_feature_indices_in_range() -> None:
    g = _make_generator(7)
    n_feats = 5
    x = torch.randn(32, n_feats, generator=g)
    split_feats, _ = sample_odt_splits(x, 4, g)
    assert torch.all(split_feats >= 0)
    assert torch.all(split_feats < n_feats)


def test_leaf_indices_range() -> None:
    g = _make_generator(99)
    depth = 4
    x = torch.randn(64, 3, generator=g)
    sf, th = sample_odt_splits(x, depth, g)
    leaf_idx = compute_odt_leaf_indices(x, sf, th)
    assert leaf_idx.shape == (64,)
    assert torch.all(leaf_idx >= 0)
    assert torch.all(leaf_idx < 2**depth)


def test_leaf_indices_deterministic() -> None:
    x = torch.randn(32, 4)
    g1 = _make_generator(0)
    sf1, th1 = sample_odt_splits(x, 3, g1)
    li1 = compute_odt_leaf_indices(x, sf1, th1)

    g2 = _make_generator(0)
    sf2, th2 = sample_odt_splits(x, 3, g2)
    li2 = compute_odt_leaf_indices(x, sf2, th2)

    torch.testing.assert_close(li1, li2)


def test_custom_feature_probs() -> None:
    g = _make_generator(42)
    x = torch.randn(64, 5, generator=g)
    probs = torch.zeros(5)
    probs[2] = 1.0  # force feature 2
    sf, _ = sample_odt_splits(x, 6, _make_generator(0), feature_probs=probs)
    assert torch.all(sf == 2)


def test_sample_odt_splits_batch_shapes() -> None:
    g = _make_generator(5)
    x = torch.randn(3, 64, 6, generator=g)
    split_feats, thresholds = sample_odt_splits_batch(x, 4, g)
    assert split_feats.shape == (3, 4)
    assert thresholds.shape == (3, 4)


def test_sample_odt_splits_batch_respects_per_dataset_feature_probs() -> None:
    g = _make_generator(7)
    x = torch.randn(2, 32, 5, generator=g)
    probs = torch.zeros(2, 5)
    probs[0, 1] = 1.0
    probs[1, 4] = 1.0
    split_feats, _ = sample_odt_splits_batch(x, 6, _make_generator(11), feature_probs=probs)
    assert torch.all(split_feats[0] == 1)
    assert torch.all(split_feats[1] == 4)


def test_compute_odt_leaf_indices_batch_matches_scalar_helper() -> None:
    g = _make_generator(13)
    x = torch.randn(3, 48, 4, generator=g)
    split_feats = torch.tensor(
        [
            [0, 1, 2],
            [3, 2, 1],
            [1, 0, 3],
        ],
        dtype=torch.long,
    )
    thresholds = torch.randn(3, 3, generator=g)
    batched = compute_odt_leaf_indices_batch(x, split_feats, thresholds)
    expected = torch.stack(
        [
            compute_odt_leaf_indices(x[idx], split_feats[idx], thresholds[idx])
            for idx in range(x.shape[0])
        ],
        dim=0,
    )
    torch.testing.assert_close(batched, expected)
