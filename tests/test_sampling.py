import numpy as np

from cauchy_generator.sampling.random_weights import sample_random_weights


def test_random_weights_normalized() -> None:
    rng = np.random.default_rng(42)
    w = sample_random_weights(32, rng)
    assert w.shape == (32,)
    assert np.all(w > 0)
    np.testing.assert_allclose(np.sum(w), 1.0, atol=1e-6, rtol=1e-6)
