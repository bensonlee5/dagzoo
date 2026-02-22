"""Tests for sampling/correlated.py — Appendix E.2."""

import numpy as np

from cauchy_generator.sampling.correlated import CorrelatedSampler


def test_sample_num_within_bounds() -> None:
    rng = np.random.default_rng(0)
    cs = CorrelatedSampler(rng)
    for _ in range(50):
        v = cs.sample_num("x", 2.0, 10.0)
        assert 2.0 <= v <= 10.0


def test_sample_num_log_scale() -> None:
    rng = np.random.default_rng(1)
    cs = CorrelatedSampler(rng)
    for _ in range(50):
        v = cs.sample_num("ls", 0.01, 100.0, log_scale=True)
        assert 0.01 <= v <= 100.0


def test_sample_num_as_int() -> None:
    rng = np.random.default_rng(2)
    cs = CorrelatedSampler(rng)
    for _ in range(50):
        v = cs.sample_num("i", 1.0, 20.0, as_int=True)
        assert isinstance(v, int)
        assert 1 <= v <= 20


def test_sample_num_deterministic() -> None:
    vals_a = []
    vals_b = []
    for seed_vals, vals in [(0, vals_a), (0, vals_b)]:
        rng = np.random.default_rng(seed_vals)
        cs = CorrelatedSampler(rng)
        for _ in range(10):
            vals.append(cs.sample_num("d", 0.0, 1.0))
    assert vals_a == vals_b


def test_sample_category_deterministic() -> None:
    vals_a = []
    vals_b = []
    for seed_vals, vals in [(0, vals_a), (0, vals_b)]:
        rng = np.random.default_rng(seed_vals)
        cs = CorrelatedSampler(rng)
        for _ in range(10):
            vals.append(cs.sample_category("cat", 5))
    assert vals_a == vals_b


def test_sample_category_valid_index() -> None:
    rng = np.random.default_rng(3)
    cs = CorrelatedSampler(rng)
    n = 7
    for _ in range(50):
        idx = cs.sample_category("cat", n)
        assert isinstance(idx, int)
        assert 0 <= idx < n
