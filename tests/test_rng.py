"""Tests for rng.py — seed derivation and SeedManager."""

import pytest

from dagzoo.rng import (
    SEED32_MAX,
    SeedManager,
    derive_seed,
    offset_seed32,
    validate_seed32,
)


def test_derive_seed_deterministic() -> None:
    assert derive_seed(42, "a", 1) == derive_seed(42, "a", 1)


def test_derive_seed_varies_with_base() -> None:
    assert derive_seed(1, "x") != derive_seed(2, "x")


def test_derive_seed_varies_with_component() -> None:
    assert derive_seed(42, "a") != derive_seed(42, "b")


def test_derive_seed_returns_valid_32bit() -> None:
    seed = derive_seed(99, "comp", 7)
    assert 0 <= seed < 2**32


def test_seed_manager_child_matches_derive() -> None:
    sm = SeedManager(seed=42)
    assert sm.child("node", 3) == derive_seed(42, "node", 3)


def test_validate_seed32_accepts_boundaries() -> None:
    assert validate_seed32(0) == 0
    assert validate_seed32(SEED32_MAX) == SEED32_MAX


@pytest.mark.parametrize("bad_seed", [-1, SEED32_MAX + 1, True])  # type: ignore[list-item]
def test_validate_seed32_rejects_out_of_range_values(bad_seed: int | bool) -> None:
    with pytest.raises(ValueError, match=r"seed must be an integer in \[0, 4294967295\]"):
        _ = validate_seed32(bad_seed)


def test_offset_seed32_wraps_on_overflow() -> None:
    assert offset_seed32(SEED32_MAX, 1) == 0
    assert offset_seed32(SEED32_MAX - 1, 2) == 0
