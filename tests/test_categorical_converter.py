"""Tests for converters/categorical.py — Appendix E.6 categorical path."""

import torch

from cauchy_generator.converters.categorical import apply_categorical_converter
from conftest import make_generator as _make_generator


def test_output_shapes() -> None:
    g = _make_generator(0)
    x = torch.randn(64, 4, generator=g)
    x_prime, labels = apply_categorical_converter(x, g, n_categories=5)
    assert x_prime.shape[0] == 64
    assert labels.shape == (64,)


def test_labels_in_range() -> None:
    g = _make_generator(1)
    n_cat = 6
    x = torch.randn(64, 4, generator=g)
    _, labels = apply_categorical_converter(x, g, n_categories=n_cat)
    assert torch.all(labels >= 0)
    assert torch.all(labels < n_cat)


def test_deterministic() -> None:
    x = torch.randn(32, 3, generator=_make_generator(99))
    xp1, l1 = apply_categorical_converter(x.clone(), _make_generator(0), n_categories=4)
    xp2, l2 = apply_categorical_converter(x.clone(), _make_generator(0), n_categories=4)
    torch.testing.assert_close(xp1, xp2)
    torch.testing.assert_close(l1, l2)


def test_finite_outputs() -> None:
    g = _make_generator(7)
    x = torch.randn(64, 5, generator=g)
    x_prime, _ = apply_categorical_converter(x, g, n_categories=3)
    assert torch.all(torch.isfinite(x_prime))
