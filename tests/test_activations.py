"""Tests for functions/activations.py — Appendix E.9."""

import torch

from cauchy_generator.functions.activations import apply_random_activation_torch
from conftest import make_generator as _make_generator


def test_output_shape() -> None:
    g = _make_generator()
    x = torch.randn(64, 4, generator=g)
    y = apply_random_activation_torch(x, g)
    assert y.shape == (64, 4)


def test_finite_outputs() -> None:
    g = _make_generator(7)
    x = torch.randn(64, 4, generator=g)
    y = apply_random_activation_torch(x, g)
    assert torch.all(torch.isfinite(y))


def test_deterministic() -> None:
    x = torch.randn(32, 3)
    y1 = apply_random_activation_torch(x.clone(), _make_generator(0))
    y2 = apply_random_activation_torch(x.clone(), _make_generator(0))
    torch.testing.assert_close(y1, y2)


def test_1d_promoted() -> None:
    g = _make_generator(99)
    x = torch.randn(64, generator=g)
    y = apply_random_activation_torch(x, g)
    assert y.dim() == 2
    assert y.shape[0] == 64
