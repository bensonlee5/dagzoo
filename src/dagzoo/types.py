"""Common data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class DatasetBundle:
    """Container for one generated dataset."""

    X_train: Any
    y_train: Any
    X_test: Any
    y_test: Any
    feature_types: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    runtime_metrics: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
