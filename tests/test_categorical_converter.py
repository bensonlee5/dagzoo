"""Tests for converters/categorical.py categorical path."""

import pytest
import torch

from dagzoo.converters.categorical import apply_categorical_converter
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


@pytest.mark.parametrize("n_cat", [10, 16, 24, 32])
def test_labels_in_range_many_class(n_cat: int) -> None:
    g = _make_generator(11 + n_cat)
    x = torch.randn(96, 8, generator=g)
    _, labels = apply_categorical_converter(x, g, n_categories=n_cat)
    assert labels.dtype == torch.int64
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


@pytest.mark.parametrize("method", ["neighbor", "softmax"])
def test_method_override_is_supported(method: str) -> None:
    g = _make_generator(21)
    x = torch.randn(48, 6, generator=g)
    x_prime, labels = apply_categorical_converter(x, g, n_categories=32, method=method)
    assert x_prime.shape == x.shape
    assert labels.shape == (x.shape[0],)


def test_unknown_method_raises() -> None:
    g = _make_generator(22)
    x = torch.randn(32, 4, generator=g)
    with pytest.raises(ValueError, match=r"Unsupported categorical converter method"):
        apply_categorical_converter(x, g, n_categories=8, method="unknown")
