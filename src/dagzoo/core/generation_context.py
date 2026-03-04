"""Shared generation-context helpers for seed, split, and device resolution."""

from __future__ import annotations

import torch

from dagzoo.config import GeneratorConfig, validate_class_split_feasibility
from dagzoo.core.constants import (
    NODE_SPEC_SEED_OFFSET,
    SPLIT_PERMUTATION_SEED_OFFSET,
)
from dagzoo.core.layout_types import LayoutPlan
from dagzoo.rng import offset_seed32, validate_seed32


def _resolve_split_sizes(config: GeneratorConfig) -> tuple[int, int]:
    """Resolve explicit train/test split sizes from config."""

    return int(config.dataset.n_train), int(config.dataset.n_test)


def _resolve_run_seed(config: GeneratorConfig, seed_override: int | None) -> int:
    """Resolve and validate the run seed used by generation entrypoints."""

    if seed_override is None:
        return validate_seed32(config.seed, field_name="seed")
    return validate_seed32(seed_override, field_name="seed")


def _attempt_seed(run_seed: int, attempt_index: int) -> int:
    """Derive deterministic per-attempt seed from one run seed."""

    return offset_seed32(run_seed, attempt_index)


def _node_spec_seed(run_seed: int, node_index: int) -> int:
    """Derive deterministic per-node spec seed from one run seed."""

    return offset_seed32(run_seed, NODE_SPEC_SEED_OFFSET + node_index)


def _split_permutation_seed(run_seed: int, attempt_index: int) -> int:
    """Derive deterministic split/postprocess seed from one run seed."""

    return offset_seed32(run_seed, SPLIT_PERMUTATION_SEED_OFFSET + attempt_index)


def _resolve_device(config: GeneratorConfig, device_override: str | None) -> str:
    """Resolve runtime device and hard-fail on unavailable explicit accelerators."""

    requested = (device_override or config.runtime.device or "auto").lower()
    mps_ok = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if mps_ok:
            return "mps"
        return "cpu"
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        raise RuntimeError("Requested device 'cuda' but CUDA is not available.")
    if requested == "mps":
        if mps_ok:
            return "mps"
        raise RuntimeError("Requested device 'mps' but MPS is not available.")
    raise ValueError(f"Unsupported device '{requested}'. Expected one of: auto, cpu, cuda, mps.")


def _torch_dtype(config: GeneratorConfig) -> torch.dtype:
    """Map string runtime dtype configuration to a torch dtype."""

    return torch.float64 if config.runtime.torch_dtype == "float64" else torch.float32


def _validate_class_split_for_layout(
    config: GeneratorConfig,
    *,
    layout: LayoutPlan,
    n_train: int,
    n_test: int,
) -> None:
    """Validate class/split feasibility for one sampled layout/split pair."""

    if config.dataset.task != "classification":
        return
    validate_class_split_feasibility(
        n_classes=int(layout.n_classes),
        n_train=int(n_train),
        n_test=int(n_test),
        context=(
            "sampled classification split constraints "
            f"(n_train={int(n_train)}, n_test={int(n_test)})"
        ),
    )
