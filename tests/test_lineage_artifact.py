from __future__ import annotations

import numpy as np
import pytest

from dagzoo.io.lineage_artifact import (
    pack_upper_triangle_adjacency,
    sha256_hex,
    unpack_upper_triangle_adjacency,
    upper_triangle_bit_length,
)


def test_pack_unpack_upper_triangle_roundtrip() -> None:
    adjacency = [
        [0, 1, 1, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
    ]
    n_nodes, edge_count, payload = pack_upper_triangle_adjacency(adjacency)

    assert n_nodes == 4
    assert edge_count == 5
    assert len(payload) == 1

    restored = unpack_upper_triangle_adjacency(
        payload,
        n_nodes=n_nodes,
        bit_length=upper_triangle_bit_length(n_nodes),
    )
    assert restored.tolist() == adjacency


def test_pack_upper_triangle_rejects_lower_triangle_edges() -> None:
    adjacency = [
        [0, 1, 0],
        [1, 0, 1],
        [0, 0, 0],
    ]
    with pytest.raises(ValueError, match="upper triangular"):
        pack_upper_triangle_adjacency(adjacency)


def test_pack_upper_triangle_rejects_non_integral_values() -> None:
    adjacency = np.array(
        [
            [0.0, 0.9, 0.0],
            [0.0, 0.0, 1.2],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    with pytest.raises(ValueError, match="entries must be 0 or 1"):
        pack_upper_triangle_adjacency(adjacency)


def test_unpack_upper_triangle_rejects_wrong_payload_length() -> None:
    with pytest.raises(ValueError, match="payload length does not match expected byte length"):
        unpack_upper_triangle_adjacency(b"", n_nodes=4, bit_length=6)


def test_sha256_hex_returns_expected_length() -> None:
    digest = sha256_hex(b"abc")
    assert len(digest) == 64
