"""Tests for functions/multi.py — Appendix E.7."""

import pytest
import torch

from cauchy_generator.functions.multi import apply_multi_function
from conftest import make_generator as _make_generator


def test_single_input() -> None:
    g = _make_generator(0)
    x = torch.randn(64, 4, generator=g)
    y = apply_multi_function([x], g, out_dim=5)
    assert y.shape == (64, 5)


def test_multiple_inputs() -> None:
    g = _make_generator(1)
    a = torch.randn(64, 3, generator=g)
    b = torch.randn(64, 2, generator=g)
    y = apply_multi_function([a, b], g, out_dim=4)
    assert y.shape == (64, 4)


def test_empty_raises() -> None:
    g = _make_generator(0)
    with pytest.raises(ValueError, match="non-empty"):
        apply_multi_function([], g, out_dim=3)


def test_deterministic() -> None:
    x = torch.randn(32, 4, generator=_make_generator(99))
    y1 = apply_multi_function([x.clone()], _make_generator(0), out_dim=3)
    y2 = apply_multi_function([x.clone()], _make_generator(0), out_dim=3)
    torch.testing.assert_close(y1, y2)


def test_finite_outputs() -> None:
    g = _make_generator(7)
    x = torch.randn(64, 4, generator=g)
    y = apply_multi_function([x], g, out_dim=3)
    assert torch.all(torch.isfinite(y))
