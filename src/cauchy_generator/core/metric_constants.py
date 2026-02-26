"""Shared constants and backend-agnostic helpers for metric extraction."""

from __future__ import annotations

import math
from typing import Any, Protocol

TASK_CLASSIFICATION = "classification"
TASK_REGRESSION = "regression"
RIDGE_LAMBDA = 1e-6
NUM_BOOTSTRAP = 200
EPS = 1e-12


class Shaped(Protocol):
    """Structural protocol for array/tensor-like inputs with `.shape`."""

    @property
    def shape(self) -> Any:
        """Return shape payload."""


def finite_or_none(value: float) -> float | None:
    """Return finite values as float; map non-finite values to None."""

    return float(value) if math.isfinite(value) else None


def resolve_task_from_metadata(metadata: dict[str, Any]) -> str | None:
    """Resolve task directly from metadata config payload when present."""

    config = metadata.get("config")
    if isinstance(config, dict):
        dataset_cfg = config.get("dataset")
        if isinstance(dataset_cfg, dict):
            raw_task = dataset_cfg.get("task")
            if raw_task in {TASK_CLASSIFICATION, TASK_REGRESSION}:
                return str(raw_task)
    return None


def validate_metric_shapes(
    *,
    feature_types: list[str],
    x_train: Shaped,
    x_test: Shaped,
    y_train: Shaped,
    y_test: Shaped,
) -> None:
    """Validate metric extraction array/tensor shapes and feature metadata size."""

    if int(x_train.shape[1]) != int(x_test.shape[1]):
        raise ValueError(
            f"X_train/X_test feature mismatch: {int(x_train.shape[1])} != {int(x_test.shape[1])}"
        )
    if int(x_train.shape[0]) != int(y_train.shape[0]):
        raise ValueError(
            f"X_train/y_train row mismatch: {int(x_train.shape[0])} != {int(y_train.shape[0])}"
        )
    if int(x_test.shape[0]) != int(y_test.shape[0]):
        raise ValueError(
            f"X_test/y_test row mismatch: {int(x_test.shape[0])} != {int(y_test.shape[0])}"
        )
    if int(x_train.shape[0]) + int(x_test.shape[0]) <= 0:
        raise ValueError("Dataset must contain at least one row across train and test splits.")
    if len(feature_types) != int(x_train.shape[1]):
        raise ValueError(
            "feature_types length must match feature count: "
            f"{len(feature_types)} != {int(x_train.shape[1])}"
        )
