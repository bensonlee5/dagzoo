"""Tests for functions/activations.py."""

import torch

from dagsynth.functions.activations import _fixed_activation, apply_random_activation
from conftest import make_generator as _make_generator


def test_output_shape() -> None:
    g = _make_generator()
    x = torch.randn(64, 4, generator=g)
    y = apply_random_activation(x, g)
    assert y.shape == (64, 4)


def test_finite_outputs() -> None:
    g = _make_generator(7)
    x = torch.randn(64, 4, generator=g)
    y = apply_random_activation(x, g)
    assert torch.all(torch.isfinite(y))


def test_deterministic() -> None:
    x = torch.randn(32, 3)
    y1 = apply_random_activation(x.clone(), _make_generator(0))
    y2 = apply_random_activation(x.clone(), _make_generator(0))
    torch.testing.assert_close(y1, y2)


def test_1d_promoted() -> None:
    g = _make_generator(99)
    x = torch.randn(64, generator=g)
    y = apply_random_activation(x, g)
    assert y.dim() == 2
    assert y.shape[0] == 64


def test_fixed_relu_sq_matches_relu_squared() -> None:
    x = torch.tensor([[-2.0, -0.5, 0.0, 1.5, 3.0]], dtype=torch.float32)
    out = _fixed_activation(x, "relu_sq")
    expected = torch.square(torch.relu(x))
    torch.testing.assert_close(out, expected)
    assert torch.all(out >= 0.0)


def test_fixed_swiglu_matches_identity_projection_definition() -> None:
    x = torch.tensor([[-2.0, -0.5, 0.0, 1.5, 3.0]], dtype=torch.float32)
    out = _fixed_activation(x, "swiglu")
    expected = x * torch.nn.functional.silu(x)
    torch.testing.assert_close(out, expected)
