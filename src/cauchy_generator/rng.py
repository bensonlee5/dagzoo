"""RNG helpers for seeded, component-level reproducibility."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch


def derive_seed(base_seed: int, *components: str | int) -> int:
    """Derive a deterministic 32-bit seed from a base seed and components."""

    h = hashlib.blake2s(digest_size=8)
    h.update(str(base_seed).encode("utf-8"))
    for comp in components:
        h.update(b"|")
        h.update(str(comp).encode("utf-8"))
    return int.from_bytes(h.digest(), "little") % (2**32 - 1)


@dataclass(slots=True)
class SeedManager:
    """Creates reproducible child seeds from a run-level seed."""

    seed: int

    def child(self, *components: str | int) -> int:
        """Return a deterministic child seed for the provided component path."""

        return derive_seed(self.seed, *components)

    def numpy_rng(self, *components: str | int) -> np.random.Generator:
        """Return a NumPy Generator seeded from a deterministic child seed."""

        return np.random.default_rng(self.child(*components))

    def torch_rng(self, *components: str | int, device: str = "cpu") -> torch.Generator:
        """Return a torch Generator seeded from a deterministic child seed."""

        g = torch.Generator(device=device)
        g.manual_seed(self.child(*components))
        return g
