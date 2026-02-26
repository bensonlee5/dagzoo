"""Central constants for core dataset generation and steering behavior."""

from __future__ import annotations

# Deterministic seed offsets used during one-dataset generation.
NODE_SPEC_SEED_OFFSET = 1_000
SPLIT_PERMUTATION_SEED_OFFSET = 10_007

# Steering target normalization guard to avoid divide-by-zero bands.
STEERING_TARGET_BAND_MIN_WIDTH = 1e-6
