"""RNG helpers for seeded, component-level reproducibility."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch

SEED32_MIN = 0
SEED32_MAX = (2**32) - 1
_SEED32_MODULUS = 2**32


def validate_seed32(seed: int, *, field_name: str = "seed") -> int:
    """Validate an external seed against the supported unsigned 32-bit range."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(
            f"{field_name} must be an integer in [{SEED32_MIN}, {SEED32_MAX}], got {seed!r}."
        )
    if seed < SEED32_MIN or seed > SEED32_MAX:
        raise ValueError(
            f"{field_name} must be an integer in [{SEED32_MIN}, {SEED32_MAX}], got {seed!r}."
        )
    return int(seed)


def offset_seed32(seed: int, offset: int) -> int:
    """Derive a wrapped child seed via 32-bit modular arithmetic."""

    return (int(seed) + int(offset)) % _SEED32_MODULUS


def derive_seed(base_seed: int, *components: str | int) -> int:
    """Derive a deterministic 32-bit seed from a base seed and components."""

    h = hashlib.blake2s(digest_size=8)
    h.update(str(base_seed).encode("utf-8"))
    for comp in components:
        h.update(b"|")
        h.update(str(comp).encode("utf-8"))
    return int.from_bytes(h.digest(), "little") % SEED32_MAX


@dataclass(slots=True)
class SeedManager:
    """Creates reproducible child seeds from a run-level seed."""

    seed: int

    def child(self, *components: str | int) -> int:
        """Return a deterministic child seed for the provided component path."""

        return derive_seed(self.seed, *components)

    def torch_rng(self, *components: str | int, device: str = "cpu") -> torch.Generator:
        """Return a torch Generator seeded from a deterministic child seed."""

        g = torch.Generator(device=device)
        g.manual_seed(self.child(*components))
        return g
