import torch

from cauchy_generator.sampling.random_weights import sample_random_weights
from conftest import make_generator as _make_generator


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
