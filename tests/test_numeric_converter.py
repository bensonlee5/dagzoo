"""Tests for converters/numeric.py numeric path."""

import torch

from dagzoo.converters.numeric import apply_numeric_converter
from conftest import make_generator as _make_generator


def test_output_shapes() -> None:
    g = _make_generator(0)
    x = torch.randn(64, 3, generator=g)
    x_prime, v = apply_numeric_converter(x, g)
    assert x_prime.shape[0] == 64
    assert v.shape == (64,)


def test_deterministic() -> None:
    x = torch.randn(32, 2, generator=_make_generator(99))
    xp1, v1 = apply_numeric_converter(x.clone(), _make_generator(0))
    xp2, v2 = apply_numeric_converter(x.clone(), _make_generator(0))
    torch.testing.assert_close(xp1, xp2)
    torch.testing.assert_close(v1, v2)


def test_finite_outputs() -> None:
    g = _make_generator(7)
    x = torch.randn(64, 4, generator=g)
    x_prime, v = apply_numeric_converter(x, g)
    assert torch.all(torch.isfinite(x_prime))
    assert torch.all(torch.isfinite(v))


def test_1d_input() -> None:
    g = _make_generator(1)
    x = torch.randn(64, generator=g)
    x_prime, v = apply_numeric_converter(x, g)
    assert x_prime.dim() == 2
    assert x_prime.shape[0] == 64
