import torch

from cauchy_generator.sampling.random_weights import sample_random_weights
from conftest import make_generator as _make_generator


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
