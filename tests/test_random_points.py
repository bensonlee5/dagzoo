"""Tests for sampling/random_points.py."""

import pytest
import torch

from dagzoo.sampling.noise import NoiseSamplingSpec
from dagzoo.sampling.random_points import sample_random_points
from conftest import make_generator as _make_generator
import dagzoo.sampling.random_points as random_points_mod


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


def test_unit_ball_sampling_is_invariant_to_noise_family(monkeypatch: pytest.MonkeyPatch) -> None:
    def _force_unit_ball(*_args, **_kwargs) -> torch.Tensor:
        return torch.tensor([2], dtype=torch.int64)

    def _identity(x: torch.Tensor, *_args, **_kwargs) -> torch.Tensor:
        return x

    monkeypatch.setattr(random_points_mod.torch, "randint", _force_unit_ball)
    monkeypatch.setattr(random_points_mod, "apply_random_function", _identity)

    baseline = sample_random_points(128, 4, _make_generator(42), "cpu")
    laplace = sample_random_points(
        128,
        4,
        _make_generator(42),
        "cpu",
        noise_spec=NoiseSamplingSpec(family="laplace"),
    )
    student_t = sample_random_points(
        128,
        4,
        _make_generator(42),
        "cpu",
        noise_spec=NoiseSamplingSpec(family="student_t", student_t_df=7.0),
    )

    torch.testing.assert_close(laplace, baseline)
    torch.testing.assert_close(student_t, baseline)
