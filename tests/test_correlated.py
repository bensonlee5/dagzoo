"""Tests for sampling/correlated.py."""

import torch

from dagzoo.sampling.correlated import CorrelatedSampler
from conftest import make_generator as _make_generator


def test_sample_num_within_bounds() -> None:
    g = _make_generator(0)
    cs = CorrelatedSampler(g, "cpu")
    for _ in range(50):
        v = cs.sample_num("x", 2.0, 10.0)
        assert 2.0 <= v <= 10.0


def test_sample_num_log_scale() -> None:
    g = _make_generator(1)
    cs = CorrelatedSampler(g, "cpu")
    for _ in range(50):
        v = cs.sample_num("ls", 0.01, 100.0, log_scale=True)
        assert 0.01 <= v <= 100.0


def test_sample_num_as_int() -> None:
    g = _make_generator(2)
    cs = CorrelatedSampler(g, "cpu")
    for _ in range(50):
        v = cs.sample_num("i", 1.0, 20.0, as_int=True)
        assert isinstance(v, int)
        assert 1 <= v <= 20


def test_sample_num_deterministic() -> None:
    vals_a = []
    vals_b = []
    for seed_vals, vals in [(0, vals_a), (0, vals_b)]:
        g = _make_generator(seed_vals)
        cs = CorrelatedSampler(g, "cpu")
        for _ in range(10):
            vals.append(cs.sample_num("d", 0.0, 1.0))
    assert vals_a == vals_b


def test_sample_num_does_not_reseed_global_rng(monkeypatch) -> None:
    g = _make_generator(0)
    cs = CorrelatedSampler(g, "cpu")

    def _manual_seed_forbidden(seed: int) -> torch.Generator:
        raise AssertionError(f"unexpected global reseed: {seed}")

    monkeypatch.setattr(torch, "manual_seed", _manual_seed_forbidden)
    _ = cs.sample_num("d", 0.0, 1.0)


def test_sample_category_deterministic() -> None:
    vals_a = []
    vals_b = []
    for seed_vals, vals in [(0, vals_a), (0, vals_b)]:
        g = _make_generator(seed_vals)
        cs = CorrelatedSampler(g, "cpu")
        for _ in range(10):
            vals.append(cs.sample_category("cat", 5))
    assert vals_a == vals_b


def test_sample_category_valid_index() -> None:
    g = _make_generator(3)
    cs = CorrelatedSampler(g, "cpu")
    n = 7
    for _ in range(50):
        idx = cs.sample_category("cat", n)
        assert isinstance(idx, int)
        assert 0 <= idx < n
