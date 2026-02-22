"""Tests for linalg/random_matrices.py — Appendix E.10."""

import numpy as np
import pytest
import torch

from cauchy_generator.linalg.random_matrices import sample_random_matrix, sample_random_matrix_torch
from conftest import make_generator as _make_generator


def test_output_shape() -> None:
    rng = np.random.default_rng(0)
    m = sample_random_matrix(5, 3, rng)
    assert m.shape == (5, 3)


def test_rows_unit_normalized() -> None:
    rng = np.random.default_rng(1)
    m = sample_random_matrix(4, 6, rng)
    norms = np.linalg.norm(m, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-4)


def test_deterministic() -> None:
    a = sample_random_matrix(3, 4, np.random.default_rng(42))
    b = sample_random_matrix(3, 4, np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("kind", ["gaussian", "weights", "singular_values", "kernel", "activation"])
def test_each_kind(kind: str) -> None:
    rng = np.random.default_rng(10)
    m = sample_random_matrix(4, 5, rng, kind=kind)
    assert m.shape == (4, 5)
    assert np.all(np.isfinite(m))


def test_invalid_kind_raises() -> None:
    with pytest.raises(ValueError, match="Unknown matrix kind"):
        sample_random_matrix(2, 2, np.random.default_rng(0), kind="bogus")


def test_torch_output_shape() -> None:
    g = _make_generator()
    m = sample_random_matrix_torch(5, 3, g, "cpu")
    assert m.shape == (5, 3)


def test_torch_rows_unit_normalized() -> None:
    g = _make_generator(1)
    m = sample_random_matrix_torch(4, 6, g, "cpu")
    norms = torch.linalg.norm(m, dim=1)
    torch.testing.assert_close(norms, torch.ones(4), atol=1e-4, rtol=1e-4)
