"""Tests for graph/cauchy_graph.py."""

import pytest
import torch

from cauchy_generator.graph.cauchy_graph import (
    dag_edge_density,
    dag_longest_path_nodes,
    sample_cauchy_dag,
)
from conftest import make_generator as _make_generator


def test_dag_shape() -> None:
    g = _make_generator(42)
    adj = sample_cauchy_dag(5, g)
    assert adj.shape == (5, 5)
    assert adj.dtype == torch.bool


def test_dag_upper_triangular() -> None:
    g = _make_generator(7)
    adj = sample_cauchy_dag(8, g)
    for i in range(8):
        for j in range(i + 1):
            assert adj[i, j] == False  # noqa: E712


def test_dag_deterministic() -> None:
    a = sample_cauchy_dag(6, _make_generator(99))
    b = sample_cauchy_dag(6, _make_generator(99))
    torch.testing.assert_close(a, b)


def test_dag_deterministic_with_edge_logit_bias() -> None:
    a = sample_cauchy_dag(6, _make_generator(99), edge_logit_bias=0.75)
    b = sample_cauchy_dag(6, _make_generator(99), edge_logit_bias=0.75)
    torch.testing.assert_close(a, b)


def test_dag_single_node() -> None:
    g = _make_generator(0)
    adj = sample_cauchy_dag(1, g)
    assert adj.shape == (1, 1)
    assert not adj[0, 0]


def test_edge_logit_bias_increases_edge_count_with_same_rng_stream() -> None:
    low_bias = sample_cauchy_dag(16, _make_generator(123), edge_logit_bias=-1.0)
    high_bias = sample_cauchy_dag(16, _make_generator(123), edge_logit_bias=1.0)
    assert int(high_bias.sum().item()) >= int(low_bias.sum().item())


def test_dag_longest_path_nodes_on_known_graph() -> None:
    adjacency = torch.tensor(
        [
            [0, 1, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ],
        dtype=torch.bool,
    )
    assert dag_longest_path_nodes(adjacency) == 4


def test_dag_longest_path_nodes_rejects_non_upper_triangular_input() -> None:
    adjacency = torch.tensor(
        [
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 0],
        ],
        dtype=torch.bool,
    )
    with pytest.raises(ValueError, match="upper-triangular"):
        dag_longest_path_nodes(adjacency)


def test_dag_longest_path_nodes_rejects_singleton_self_loop() -> None:
    adjacency = torch.tensor([[1]], dtype=torch.bool)
    with pytest.raises(ValueError, match="upper-triangular"):
        dag_longest_path_nodes(adjacency)


def test_dag_edge_density_on_known_graph() -> None:
    adjacency = torch.tensor(
        [
            [0, 1, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=torch.bool,
    )
    assert dag_edge_density(adjacency) == 0.5


def test_dag_edge_density_single_node_is_zero() -> None:
    adjacency = torch.zeros((1, 1), dtype=torch.bool)
    assert dag_edge_density(adjacency) == 0.0


def test_dag_edge_density_rejects_singleton_self_loop() -> None:
    adjacency = torch.tensor([[1]], dtype=torch.bool)
    with pytest.raises(ValueError, match="upper-triangular"):
        dag_edge_density(adjacency)
