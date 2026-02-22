"""Tests for converters/categorical.py — Appendix E.6 categorical path."""

import numpy as np
import torch

from cauchy_generator.converters.categorical import (
    apply_categorical_converter,
    apply_categorical_converter_torch,
)
from conftest import make_generator as _make_generator


def test_output_shapes() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, 4)).astype(np.float32)
    x_prime, labels = apply_categorical_converter(x, rng, n_categories=5)
    assert x_prime.shape[0] == 64
    assert labels.shape == (64,)


def test_labels_in_range() -> None:
    rng = np.random.default_rng(1)
    n_cat = 6
    x = rng.normal(size=(64, 4)).astype(np.float32)
    _, labels = apply_categorical_converter(x, rng, n_categories=n_cat)
    assert np.all(labels >= 0)
    assert np.all(labels < n_cat)


def test_deterministic() -> None:
    x = np.random.default_rng(99).normal(size=(32, 3)).astype(np.float32)
    xp1, l1 = apply_categorical_converter(x.copy(), np.random.default_rng(0), n_categories=4)
    xp2, l2 = apply_categorical_converter(x.copy(), np.random.default_rng(0), n_categories=4)
    np.testing.assert_array_equal(xp1, xp2)
    np.testing.assert_array_equal(l1, l2)


def test_finite_outputs() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(64, 5)).astype(np.float32)
    x_prime, _ = apply_categorical_converter(x, rng, n_categories=3)
    assert np.all(np.isfinite(x_prime))


def test_torch_output_shapes() -> None:
    g = _make_generator()
    x = torch.randn(64, 4, generator=g)
    x_prime, labels = apply_categorical_converter_torch(x, g, n_categories=5)
    assert x_prime.shape[0] == 64
    assert labels.shape == (64,)


def test_torch_deterministic() -> None:
    x = torch.randn(32, 4, generator=_make_generator(99))
    xp1, l1 = apply_categorical_converter_torch(x.clone(), _make_generator(0), n_categories=4)
    xp2, l2 = apply_categorical_converter_torch(x.clone(), _make_generator(0), n_categories=4)
    torch.testing.assert_close(xp1, xp2)
    torch.testing.assert_close(l1, l2)


def test_torch_labels_in_range() -> None:
    g = _make_generator(1)
    n_cat = 6
    x = torch.randn(64, 4, generator=g)
    _, labels = apply_categorical_converter_torch(x, g, n_categories=n_cat)
    assert torch.all(labels >= 0)
    assert torch.all(labels < n_cat)
