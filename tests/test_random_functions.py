import pytest
import torch

from dagsynth.functions.random_functions import (
    MechanismFamily,
    _apply_tree,
    apply_random_function,
)
from conftest import make_generator as _make_generator


def test_apply_tree_survives_nan_feature() -> None:
    """_apply_tree should produce output even when one feature column is all-NaN."""
    g = _make_generator(42)
    x = torch.randn(64, 4, generator=g)
    x[:, 2] = float("nan")

    y = _apply_tree(x, out_dim=2, generator=g)
    assert y.shape == (64, 2)


def test_apply_tree_constant_features() -> None:
    """_apply_tree should handle input where all features are constant."""
    g = _make_generator(7)
    x = torch.ones(32, 3)

    y = _apply_tree(x, out_dim=1, generator=g)
    assert y.shape == (32, 1)
    assert torch.all(torch.isfinite(y))


def test_output_shape() -> None:
    g = _make_generator(0)
    x = torch.randn(64, 4, generator=g)
    y = apply_random_function(x, g, out_dim=5)
    assert y.shape == (64, 5)


def test_deterministic() -> None:
    x = torch.randn(32, 4)
    y1 = apply_random_function(x.clone(), _make_generator(0), out_dim=3)
    y2 = apply_random_function(x.clone(), _make_generator(0), out_dim=3)
    torch.testing.assert_close(y1, y2)


def test_deterministic_with_shift_tilt_and_noise_multiplier() -> None:
    x = torch.randn(32, 4)
    y1 = apply_random_function(
        x.clone(),
        _make_generator(8),
        out_dim=3,
        mechanism_logit_tilt=0.7,
        noise_sigma_multiplier=1.35,
    )
    y2 = apply_random_function(
        x.clone(),
        _make_generator(8),
        out_dim=3,
        mechanism_logit_tilt=0.7,
        noise_sigma_multiplier=1.35,
    )
    torch.testing.assert_close(y1, y2)


@pytest.mark.parametrize(
    "family",
    ["nn", "tree", "discretization", "gp", "linear", "quadratic", "em", "product"],
)
def test_each_family(family: MechanismFamily) -> None:
    g = _make_generator(10)
    x = torch.randn(64, 4, generator=g)
    y = apply_random_function(x, _make_generator(10), out_dim=3, function_type=family)
    assert y.shape == (64, 3)
    assert torch.all(torch.isfinite(y))


def test_invalid_type_raises() -> None:
    g = _make_generator()
    x = torch.randn(32, 4, generator=g)
    with pytest.raises(ValueError, match="Unknown random function family"):
        apply_random_function(x, g, out_dim=2, function_type="bogus")  # type: ignore[arg-type]
