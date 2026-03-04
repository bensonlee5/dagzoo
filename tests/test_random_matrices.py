"""Tests for linalg/random_matrices.py."""

import pytest
import torch

from dagzoo.linalg.random_matrices import sample_random_matrix
from conftest import make_generator as _make_generator


def test_output_shape() -> None:
    g = _make_generator(0)
    m = sample_random_matrix(5, 3, g, "cpu")
    assert m.shape == (5, 3)


def test_rows_unit_normalized() -> None:
    g = _make_generator(1)
    m = sample_random_matrix(4, 6, g, "cpu")
    norms = torch.linalg.norm(m, dim=1)
    torch.testing.assert_close(norms, torch.ones(4), atol=1e-4, rtol=1e-4)


def test_deterministic() -> None:
    a = sample_random_matrix(3, 4, _make_generator(42), "cpu")
    b = sample_random_matrix(3, 4, _make_generator(42), "cpu")
    torch.testing.assert_close(a, b)


@pytest.mark.parametrize("kind", ["gaussian", "weights", "singular_values", "kernel", "activation"])
def test_each_kind(kind: str) -> None:
    g = _make_generator(10)
    m = sample_random_matrix(4, 5, g, "cpu", kind=kind)
    assert m.shape == (4, 5)
    assert torch.all(torch.isfinite(m))


def test_invalid_kind_raises() -> None:
    with pytest.raises(ValueError, match="Unknown matrix kind"):
        sample_random_matrix(2, 2, _make_generator(0), "cpu", kind="bogus")
