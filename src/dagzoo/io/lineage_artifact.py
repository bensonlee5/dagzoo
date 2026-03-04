"""Helpers for compact lineage adjacency artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def upper_triangle_bit_length(n_nodes: int) -> int:
    """Return strict upper-triangle bit length for `n_nodes`."""

    if n_nodes < 2:
        raise ValueError(f"n_nodes must be >= 2, got {n_nodes}.")
    return (n_nodes * (n_nodes - 1)) // 2


def pack_upper_triangle_adjacency(
    adjacency: np.ndarray | list[list[int]],
) -> tuple[int, int, bytes]:
    """Pack an upper-triangular binary adjacency matrix using little-endian bit order."""

    matrix_raw = np.asarray(adjacency)
    if matrix_raw.ndim != 2 or matrix_raw.shape[0] != matrix_raw.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    n_nodes = int(matrix_raw.shape[0])
    if n_nodes < 2:
        raise ValueError(f"adjacency must have n_nodes >= 2, got {n_nodes}.")

    if np.any((matrix_raw != 0) & (matrix_raw != 1)):
        raise ValueError("adjacency entries must be 0 or 1")

    matrix = matrix_raw.astype(np.uint8, copy=False)
    if np.any(np.diag(matrix) != 0):
        raise ValueError("adjacency diagonal must be all zeros")
    if np.any(np.tril(matrix, k=-1) != 0):
        raise ValueError("adjacency must be upper triangular")

    src, dst = np.triu_indices(n_nodes, k=1)
    upper_bits = matrix[src, dst].astype(np.uint8, copy=False)
    packed = np.packbits(upper_bits, bitorder="little")
    edge_count = int(np.sum(upper_bits, dtype=np.int64))
    return n_nodes, edge_count, packed.tobytes()


def unpack_upper_triangle_adjacency(
    payload: bytes,
    *,
    n_nodes: int,
    bit_length: int | None = None,
) -> np.ndarray:
    """Unpack bit-packed strict upper-triangle adjacency into a dense matrix."""

    expected_bits = upper_triangle_bit_length(n_nodes)
    if bit_length is None:
        bit_length = expected_bits
    if bit_length != expected_bits:
        raise ValueError(f"bit_length must equal {expected_bits} for n_nodes={n_nodes}.")

    expected_bytes = (bit_length + 7) // 8
    if len(payload) != expected_bytes:
        raise ValueError(
            "payload length does not match expected byte length "
            f"{expected_bytes} for bit_length={bit_length}."
        )

    packed = np.frombuffer(payload, dtype=np.uint8)
    unpacked = np.unpackbits(packed, bitorder="little")
    bits = unpacked[:bit_length]
    if int(bits.shape[0]) != bit_length:
        raise ValueError(f"payload does not contain enough bits for bit_length={bit_length}.")

    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.uint8)
    src, dst = np.triu_indices(n_nodes, k=1)
    adjacency[src, dst] = bits.astype(np.uint8, copy=False)
    return adjacency


def sha256_hex(payload: bytes) -> str:
    """Return SHA-256 hex digest for payload bytes."""

    return hashlib.sha256(payload).hexdigest()


def resolve_lineage_path(dataset_dir: Path, path_hint: str) -> Path:
    """Resolve a lineage artifact path relative to one dataset directory."""

    hinted = Path(path_hint)
    if hinted.is_absolute():
        return hinted
    return dataset_dir / hinted


__all__ = [
    "pack_upper_triangle_adjacency",
    "resolve_lineage_path",
    "sha256_hex",
    "unpack_upper_triangle_adjacency",
    "upper_triangle_bit_length",
]
