"""Tests for converters/numeric.py — Appendix E.6 numeric path."""

import numpy as np
import torch

from cauchy_generator.converters.numeric import (
    apply_numeric_converter,
    apply_numeric_converter_torch,
)


def _make_generator(seed: int = 42) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return g


def test_output_shapes() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, 3)).astype(np.float32)
    x_prime, v = apply_numeric_converter(x, rng)
    assert x_prime.shape[0] == 64
    assert v.shape == (64,)


def test_deterministic() -> None:
    x = np.random.default_rng(99).normal(size=(32, 2)).astype(np.float32)
    xp1, v1 = apply_numeric_converter(x.copy(), np.random.default_rng(0))
    xp2, v2 = apply_numeric_converter(x.copy(), np.random.default_rng(0))
    np.testing.assert_array_equal(xp1, xp2)
    np.testing.assert_array_equal(v1, v2)


def test_finite_outputs() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(64, 4)).astype(np.float32)
    x_prime, v = apply_numeric_converter(x, rng)
    assert np.all(np.isfinite(x_prime))
    assert np.all(np.isfinite(v))


def test_1d_input() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(64,)).astype(np.float32)
    x_prime, v = apply_numeric_converter(x, rng)
    assert x_prime.ndim == 2
    assert x_prime.shape[0] == 64


def test_torch_output_shapes() -> None:
    g = _make_generator()
    x = torch.randn(64, 3, generator=g)
    x_prime, v = apply_numeric_converter_torch(x, g)
    assert x_prime.shape[0] == 64
    assert v.shape == (64,)


def test_torch_deterministic() -> None:
    x = torch.randn(32, 3, generator=_make_generator(99))
    xp1, v1 = apply_numeric_converter_torch(x.clone(), _make_generator(0))
    xp2, v2 = apply_numeric_converter_torch(x.clone(), _make_generator(0))
    torch.testing.assert_close(xp1, xp2)
    torch.testing.assert_close(v1, v2)


def test_torch_1d_input() -> None:
    g = _make_generator(1)
    x = torch.randn(64, generator=g)
    x_prime, v = apply_numeric_converter_torch(x, g)
    assert x_prime.dim() == 2
    assert x_prime.shape[0] == 64
