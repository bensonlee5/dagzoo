"""Tests for rng.py — seed derivation and SeedManager."""

import pytest
import torch

import dagzoo.rng as rng_mod
from dagzoo.rng import (
    KeyedRng,
    SEED32_MAX,
    SeedManager,
    derive_seed,
    keyed_rng_from_generator,
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


def test_keyed_rng_child_seed_matches_derive() -> None:
    rng = KeyedRng(seed=42, path=("node", 3))
    assert rng.child_seed("weights") == derive_seed(42, "node", 3, "weights")


def test_keyed_rng_normalizes_list_path_to_tuple() -> None:
    rng = KeyedRng(seed=42, path=["node", 3])  # type: ignore[arg-type]
    assert rng.path == ("node", 3)
    assert rng.keyed("weights").child_seed() == derive_seed(42, "node", 3, "weights")


def test_keyed_rng_treats_scalar_string_path_as_one_component() -> None:
    rng = KeyedRng(seed=42, path="node")  # type: ignore[arg-type]
    assert rng.path == ("node",)
    assert rng.child_seed() == KeyedRng(seed=42).keyed("node").child_seed()


def test_keyed_rng_treats_scalar_int_path_as_one_component() -> None:
    rng = KeyedRng(seed=42, path=3)  # type: ignore[arg-type]
    assert rng.path == (3,)
    assert rng.child_seed() == KeyedRng(seed=42).keyed(3).child_seed()


def test_keyed_rng_path_does_not_track_source_list_mutation() -> None:
    source_path = ["node", 3]
    rng = KeyedRng(seed=42, path=source_path)  # type: ignore[arg-type]
    before = rng.child_seed("weights")
    source_path.append("mutated")
    assert rng.path == ("node", 3)
    assert rng.child_seed("weights") == before


def test_keyed_rng_keyed_chaining_matches_flat_derivation() -> None:
    root = KeyedRng(seed=17)
    assert root.keyed("dataset", 4).child_seed("plan") == root.child_seed("dataset", 4, "plan")


def test_keyed_rng_from_generator_uses_ambient_nonce_in_child_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    words = iter(
        [
            torch.tensor([17, 101, 102, 103], dtype=torch.int64),
            torch.tensor([17, 201, 202, 203], dtype=torch.int64),
        ]
    )
    monkeypatch.setattr(rng_mod.torch, "randint", lambda *args, **kwargs: next(words))

    generator = torch.Generator(device="cpu")
    generator.manual_seed(99)
    first = keyed_rng_from_generator(generator, "root")
    second = keyed_rng_from_generator(generator, "root")

    assert first.seed == second.seed == 17
    assert first.child_seed("leaf") != second.child_seed("leaf")


def test_generator_derived_keyed_rng_preserves_nonce_across_keyed_children() -> None:
    root = KeyedRng(seed=17, _ambient_nonce=(101, 102, 103))
    assert root.keyed("dataset", 4).child_seed("plan") == root.child_seed("dataset", 4, "plan")


def test_keyed_rng_sibling_order_is_independent() -> None:
    root = KeyedRng(seed=99)
    first_a = root.keyed("a").child_seed("leaf")
    first_b = root.keyed("b").child_seed("leaf")
    second_b = root.keyed("b").child_seed("leaf")
    second_a = root.keyed("a").child_seed("leaf")
    assert first_a == second_a
    assert first_b == second_b


def test_keyed_rng_torch_rng_is_deterministic_for_same_key() -> None:
    root = KeyedRng(seed=123)
    draws_a = torch.rand(8, generator=root.torch_rng("node", 2), device="cpu")
    draws_b = torch.rand(8, generator=root.torch_rng("node", 2), device="cpu")
    torch.testing.assert_close(draws_a, draws_b)


def test_seed_manager_child_matches_keyed_rng() -> None:
    manager = SeedManager(seed=321)
    keyed = KeyedRng(seed=321)
    assert manager.child("node", 1) == keyed.child_seed("node", 1)


def test_seed_manager_torch_rng_matches_keyed_rng() -> None:
    manager = SeedManager(seed=321)
    keyed = KeyedRng(seed=321)
    manager_draws = torch.rand(8, generator=manager.torch_rng("node", 1), device="cpu")
    keyed_draws = torch.rand(8, generator=keyed.torch_rng("node", 1), device="cpu")
    torch.testing.assert_close(manager_draws, keyed_draws)


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
