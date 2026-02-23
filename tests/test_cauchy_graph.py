"""Tests for graph/cauchy_graph.py — Appendix E.4."""

import torch

from cauchy_generator.graph.cauchy_graph import sample_cauchy_dag
from conftest import make_generator as _make_generator


def test_dag_shape() -> None:
    g = _make_generator(42)
    adj = sample_cauchy_dag(5, g, "cpu")
    assert adj.shape == (5, 5)
    assert adj.dtype == torch.bool


def test_dag_upper_triangular() -> None:
    g = _make_generator(7)
    adj = sample_cauchy_dag(8, g, "cpu")
    for i in range(8):
        for j in range(i + 1):
            assert adj[i, j] == False  # noqa: E712


def test_dag_deterministic() -> None:
    a = sample_cauchy_dag(6, _make_generator(99), "cpu")
    b = sample_cauchy_dag(6, _make_generator(99), "cpu")
    torch.testing.assert_close(a, b)


def test_dag_single_node() -> None:
    g = _make_generator(0)
    adj = sample_cauchy_dag(1, g, "cpu")
    assert adj.shape == (1, 1)
    assert not adj[0, 0]
