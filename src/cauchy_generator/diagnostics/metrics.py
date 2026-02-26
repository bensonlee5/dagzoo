"""Dataset-level meta-feature metrics extractors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import fields
from typing import Any

from cauchy_generator.core.steering_metrics import extract_steering_metrics
from cauchy_generator.types import DatasetBundle

from .types import DatasetMetrics


def extract_metrics_batch(
    bundles: Sequence[DatasetBundle], *, include_spearman: bool = False
) -> list[DatasetMetrics]:
    """Extract dataset-level metrics for a sequence of bundles in order."""

    return [
        extract_dataset_metrics(bundle, include_spearman=include_spearman) for bundle in bundles
    ]


def extract_dataset_metrics(
    bundle: DatasetBundle, *, include_spearman: bool = False
) -> DatasetMetrics:
    """Extract deterministic dataset-level metrics from one generated bundle."""

    # Derive all possible numeric metric names from the DatasetMetrics dataclass.
    metric_names = {f.name for f in fields(DatasetMetrics) if f.name != "task"}
    if not include_spearman:
        metric_names -= {"spearman_abs_mean", "spearman_abs_max"}

    raw = extract_steering_metrics(
        bundle, target_metric_names=metric_names, include_spearman=include_spearman
    )

    # Convert numeric results to expected types (int where appropriate)
    def _int_or_none(v: Any) -> int | None:
        return int(v) if v is not None else None

    return DatasetMetrics(
        task=str(raw["task"]),
        n_rows=int(raw["n_rows"]),
        n_features=int(raw["n_features"]),
        n_classes=_int_or_none(raw.get("n_classes")),
        n_categorical_features=int(raw["n_categorical_features"]),
        categorical_ratio=float(raw["categorical_ratio"]),
        linearity_proxy=raw.get("linearity_proxy"),
        nonlinearity_proxy=raw.get("nonlinearity_proxy"),
        wins_ratio_proxy=raw.get("wins_ratio_proxy"),
        pearson_abs_mean=raw.get("pearson_abs_mean"),
        pearson_abs_max=raw.get("pearson_abs_max"),
        spearman_abs_mean=raw.get("spearman_abs_mean"),
        spearman_abs_max=raw.get("spearman_abs_max"),
        class_entropy=raw.get("class_entropy"),
        majority_minority_ratio=raw.get("majority_minority_ratio"),
        snr_proxy_db=raw.get("snr_proxy_db"),
        cat_cardinality_min=_int_or_none(raw.get("cat_cardinality_min")),
        cat_cardinality_mean=raw.get("cat_cardinality_mean"),
        cat_cardinality_max=_int_or_none(raw.get("cat_cardinality_max")),
    )
