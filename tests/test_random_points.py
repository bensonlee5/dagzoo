"""Tests for sampling/random_points.py — Appendix E.12."""

import pytest
import torch

from cauchy_generator.sampling.random_points import sample_random_points
from conftest import make_generator as _make_generator


def test_output_shape() -> None:
    g = _make_generator(0)
    pts = sample_random_points(64, 4, g, "cpu")
    assert pts.shape == (64, 4)


def test_deterministic() -> None:
    a = sample_random_points(32, 3, _make_generator(7), "cpu")
    b = sample_random_points(32, 3, _make_generator(7), "cpu")
    torch.testing.assert_close(a, b)


def test_finite_outputs() -> None:
    g = _make_generator(99)
    pts = sample_random_points(64, 4, g, "cpu")
    assert torch.all(torch.isfinite(pts))


def test_invalid_dims_raises() -> None:
    g = _make_generator(0)
    with pytest.raises(ValueError, match="must be > 0"):
        sample_random_points(0, 4, g, "cpu")
    with pytest.raises(ValueError, match="must be > 0"):
        sample_random_points(10, 0, g, "cpu")
