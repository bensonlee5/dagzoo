"""Tests for functions/multi.py — Appendix E.7."""

import numpy as np
import pytest
import torch

from cauchy_generator.functions.multi import (
    apply_multi_function,
    apply_multi_function_torch,
)


def _make_generator(seed: int = 42) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return g


def test_single_input() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, 4)).astype(np.float32)
    y = apply_multi_function([x], rng, out_dim=5)
    assert y.shape == (64, 5)


def test_multiple_inputs() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(size=(64, 3)).astype(np.float32)
    b = rng.normal(size=(64, 2)).astype(np.float32)
    y = apply_multi_function([a, b], rng, out_dim=4)
    assert y.shape == (64, 4)


def test_empty_raises() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="non-empty"):
        apply_multi_function([], rng, out_dim=3)


def test_deterministic() -> None:
    x = np.random.default_rng(99).normal(size=(32, 4)).astype(np.float32)
    y1 = apply_multi_function([x.copy()], np.random.default_rng(0), out_dim=3)
    y2 = apply_multi_function([x.copy()], np.random.default_rng(0), out_dim=3)
    np.testing.assert_array_equal(y1, y2)


def test_finite_outputs() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(64, 4)).astype(np.float32)
    y = apply_multi_function([x], rng, out_dim=3)
    assert np.all(np.isfinite(y))


def test_torch_single_input() -> None:
    g = _make_generator()
    x = torch.randn(64, 4, generator=g)
    y = apply_multi_function_torch([x], g, out_dim=5)
    assert y.shape == (64, 5)


def test_torch_multiple_inputs() -> None:
    g = _make_generator(1)
    a = torch.randn(64, 3, generator=g)
    b = torch.randn(64, 2, generator=g)
    y = apply_multi_function_torch([a, b], g, out_dim=4)
    assert y.shape == (64, 4)


def test_torch_deterministic() -> None:
    x = torch.randn(32, 4, generator=_make_generator(99))
    y1 = apply_multi_function_torch([x.clone()], _make_generator(0), out_dim=3)
    y2 = apply_multi_function_torch([x.clone()], _make_generator(0), out_dim=3)
    torch.testing.assert_close(y1, y2)


def test_torch_empty_raises() -> None:
    g = _make_generator()
    with pytest.raises(ValueError, match="non-empty"):
        apply_multi_function_torch([], g, out_dim=3)
