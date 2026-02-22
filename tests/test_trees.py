"""Tests for core/trees.py — ODT split and leaf index utilities."""

import torch

from cauchy_generator.core.trees import compute_odt_leaf_indices, sample_odt_splits
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
