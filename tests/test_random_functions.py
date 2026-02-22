import pytest
import torch

from cauchy_generator.functions.random_functions import (
    _apply_tree_torch,
    apply_random_function_torch,
)
from conftest import make_generator as _make_generator


def test_apply_tree_torch_survives_nan_feature() -> None:
    """_apply_tree_torch should produce output even when one feature column is all-NaN."""
    g = _make_generator(42)
    x = torch.randn(64, 4, generator=g)
    x[:, 2] = float("nan")

    y = _apply_tree_torch(x, out_dim=2, generator=g)
    assert y.shape == (64, 2)


def test_apply_tree_torch_constant_features() -> None:
    """_apply_tree_torch should handle input where all features are constant."""
    g = _make_generator(7)
    x = torch.ones(32, 3)

    y = _apply_tree_torch(x, out_dim=1, generator=g)
    assert y.shape == (32, 1)
    assert torch.all(torch.isfinite(y))


def test_output_shape() -> None:
    g = _make_generator(0)
    x = torch.randn(64, 4, generator=g)
    y = apply_random_function_torch(x, g, out_dim=5)
    assert y.shape == (64, 5)


def test_deterministic() -> None:
    x = torch.randn(32, 4)
    y1 = apply_random_function_torch(x.clone(), _make_generator(0), out_dim=3)
    y2 = apply_random_function_torch(x.clone(), _make_generator(0), out_dim=3)
    torch.testing.assert_close(y1, y2)


@pytest.mark.parametrize(
    "family",
    ["nn", "tree", "discretization", "gp", "linear", "quadratic", "em", "product"],
)
def test_each_family(family: str) -> None:
    g = _make_generator(10)
    x = torch.randn(64, 4, generator=g)
    y = apply_random_function_torch(x, _make_generator(10), out_dim=3, function_type=family)
    assert y.shape == (64, 3)
    assert torch.all(torch.isfinite(y))


def test_invalid_type_raises() -> None:
    g = _make_generator()
    x = torch.randn(32, 4, generator=g)
    with pytest.raises(ValueError, match="Unknown random function family"):
        apply_random_function_torch(x, g, out_dim=2, function_type="bogus")
