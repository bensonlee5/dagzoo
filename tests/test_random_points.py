"""Tests for sampling/random_points.py — Appendix E.12."""

import numpy as np
import pytest
import torch

from cauchy_generator.sampling.random_points import (
    sample_random_points,
    sample_random_points_torch,
)
from conftest import make_generator as _make_generator


def test_output_shape() -> None:
    rng = np.random.default_rng(0)
    pts = sample_random_points(64, 4, rng)
    assert pts.shape == (64, 4)


def test_deterministic() -> None:
    a = sample_random_points(32, 3, np.random.default_rng(7))
    b = sample_random_points(32, 3, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


def test_finite_outputs() -> None:
    rng = np.random.default_rng(99)
    pts = sample_random_points(64, 4, rng)
    assert np.all(np.isfinite(pts))


def test_invalid_dims_raises() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="must be > 0"):
        sample_random_points(0, 4, rng)
    with pytest.raises(ValueError, match="must be > 0"):
        sample_random_points(10, 0, rng)


def test_torch_output_shape() -> None:
    g = _make_generator()
    pts = sample_random_points_torch(64, 4, g, "cpu")
    assert pts.shape == (64, 4)


def test_torch_deterministic() -> None:
    a = sample_random_points_torch(32, 3, _make_generator(7), "cpu")
    b = sample_random_points_torch(32, 3, _make_generator(7), "cpu")
    torch.testing.assert_close(a, b)
