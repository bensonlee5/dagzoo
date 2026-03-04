import pytest
import torch

from dagzoo.sampling.noise import NoiseSamplingSpec
from dagzoo.sampling.random_weights import sample_random_weights
from conftest import make_generator as _make_generator
import dagzoo.sampling.random_weights as random_weights_mod


def _entropy(weights: torch.Tensor) -> float:
    probs = torch.clamp(weights, min=1e-12)
    return float(-(probs * torch.log(probs)).sum().item())


def test_random_weights_normalized() -> None:
    g = _make_generator(42)
    w = sample_random_weights(32, g, "cpu")
    assert w.shape == (32,)
    assert torch.all(w > 0)
    torch.testing.assert_close(w.sum(), torch.tensor(1.0), atol=1e-5, rtol=1e-5)


def test_random_weights_deterministic() -> None:
    a = sample_random_weights(16, _make_generator(0), "cpu")
    b = sample_random_weights(16, _make_generator(0), "cpu")
    torch.testing.assert_close(a, b)


def test_random_weights_positive() -> None:
    g = _make_generator(7)
    w = sample_random_weights(64, g, "cpu")
    assert torch.all(w > 0)


def test_random_weights_sigma_multiplier_increases_peakedness() -> None:
    low = sample_random_weights(
        64,
        _make_generator(314),
        "cpu",
        q=0.0,
        sigma=1.0,
        sigma_multiplier=1.0,
    )
    high = sample_random_weights(
        64,
        _make_generator(314),
        "cpu",
        q=0.0,
        sigma=1.0,
        sigma_multiplier=1.5,
    )
    assert _entropy(high) < _entropy(low)


def test_random_weights_nonlegacy_noise_remains_finite() -> None:
    w = sample_random_weights(
        128,
        _make_generator(101),
        "cpu",
        noise_spec=NoiseSamplingSpec(family="student_t", student_t_df=5.0),
    )
    assert torch.all(torch.isfinite(w))
    assert torch.all(w > 0)
    torch.testing.assert_close(w.sum(), torch.tensor(1.0), atol=1e-5, rtol=1e-5)


def test_random_weights_handles_nonfinite_noise_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _bad_noise(*_args, **_kwargs) -> torch.Tensor:
        return torch.tensor([float("inf"), float("-inf"), float("nan"), 0.0])

    monkeypatch.setattr(random_weights_mod, "sample_noise_from_spec", _bad_noise)
    w = sample_random_weights(4, _make_generator(202), "cpu", q=1.0, sigma=1.0)
    assert torch.all(torch.isfinite(w))
    assert torch.all(w > 0)
    torch.testing.assert_close(w.sum(), torch.tensor(1.0), atol=1e-5, rtol=1e-5)
