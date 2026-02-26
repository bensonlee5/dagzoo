"""Dataset-level meta-feature metrics extractors."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from cauchy_generator.core.metric_constants import (
    EPS as _EPS,
    NUM_BOOTSTRAP as _NUM_BOOTSTRAP,
    RIDGE_LAMBDA as _RIDGE_LAMBDA,
    TASK_CLASSIFICATION as _TASK_CLASSIFICATION,
    TASK_REGRESSION as _TASK_REGRESSION,
    finite_or_none as _finite_or_none,
    resolve_task_from_metadata,
    validate_metric_shapes,
)
from cauchy_generator.filtering import apply_torch_rf_filter
from cauchy_generator.math_utils import to_numpy as _to_numpy
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

    x_train = _as_2d_float(_to_numpy(bundle.X_train), name="X_train")
    x_test = _as_2d_float(_to_numpy(bundle.X_test), name="X_test")
    y_train = _as_target_array(_to_numpy(bundle.y_train), name="y_train")
    y_test = _as_target_array(_to_numpy(bundle.y_test), name="y_test")

    validate_metric_shapes(
        feature_types=bundle.feature_types,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
    )

    x_all = np.concatenate([x_train, x_test], axis=0)
    y_all = _concat_targets(y_train, y_test)
    task = _resolve_task(bundle.metadata, y_all)

    pearson_abs_mean, pearson_abs_max = _pearson_abs_stats(x_all)
    spearman_abs_mean: float | None = None
    spearman_abs_max: float | None = None
    if include_spearman:
        spearman_abs_mean, spearman_abs_max = _spearman_abs_stats(x_all)

    (
        n_categorical_features,
        categorical_ratio,
        cat_cardinality_min,
        cat_cardinality_mean,
        cat_cardinality_max,
    ) = _categorical_cardinality_stats(x_all, bundle.feature_types)

    linearity_proxy = _compute_linearity_proxy(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        task=task,
    )
    wins_ratio_proxy = _compute_wins_ratio_proxy(x=x_all, y=y_all, task=task)
    nonlinearity_proxy = None
    if linearity_proxy is not None and wins_ratio_proxy is not None:
        nonlinearity_proxy = max(0.0, wins_ratio_proxy - linearity_proxy)

    class_entropy: float | None = None
    majority_minority_ratio: float | None = None
    n_classes: int | None = None
    if task == _TASK_CLASSIFICATION:
        y_labels = _classification_labels(y_all)
        n_classes = int(np.unique(y_labels).size)
        class_entropy, majority_minority_ratio = _class_imbalance_stats(y_labels)

    snr_proxy_db = _compute_snr_proxy_db(
        x_train=x_train,
        x_all=x_all,
        y_train=y_train,
        y_all=y_all,
        task=task,
    )

    return DatasetMetrics(
        task=task,
        n_rows=int(x_all.shape[0]),
        n_features=int(x_all.shape[1]),
        n_classes=n_classes,
        n_categorical_features=n_categorical_features,
        categorical_ratio=categorical_ratio,
        linearity_proxy=linearity_proxy,
        nonlinearity_proxy=nonlinearity_proxy,
        wins_ratio_proxy=wins_ratio_proxy,
        pearson_abs_mean=pearson_abs_mean,
        pearson_abs_max=pearson_abs_max,
        spearman_abs_mean=spearman_abs_mean,
        spearman_abs_max=spearman_abs_max,
        class_entropy=class_entropy,
        majority_minority_ratio=majority_minority_ratio,
        snr_proxy_db=snr_proxy_db,
        cat_cardinality_min=cat_cardinality_min,
        cat_cardinality_mean=cat_cardinality_mean,
        cat_cardinality_max=cat_cardinality_max,
    )


def _as_2d_float(value: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 array, got shape={arr.shape!r}")
    return arr


def _as_target_array(value: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim not in {1, 2}:
        raise ValueError(f"{name} must be rank-1 or rank-2, got shape={arr.shape!r}")
    return arr


def _concat_targets(y_train: np.ndarray, y_test: np.ndarray) -> np.ndarray:
    if y_train.ndim == y_test.ndim:
        return np.concatenate([y_train, y_test], axis=0)

    if y_train.ndim == 1 and y_test.ndim == 2 and y_test.shape[1] == 1:
        return np.concatenate([y_train[:, np.newaxis], y_test], axis=0)

    if y_train.ndim == 2 and y_test.ndim == 1 and y_train.shape[1] == 1:
        return np.concatenate([y_train, y_test[:, np.newaxis]], axis=0)

    raise ValueError(
        f"y_train/y_test rank mismatch is unsupported: {y_train.shape!r} vs {y_test.shape!r}"
    )


def _resolve_task(metadata: dict[str, Any], y_all: np.ndarray) -> str:
    task = resolve_task_from_metadata(metadata)
    if task is not None:
        return task

    labels = np.ravel(np.asarray(y_all))
    if labels.size == 0:
        raise ValueError("Cannot infer task from empty target array.")
    if np.issubdtype(labels.dtype, np.integer):
        return _TASK_CLASSIFICATION

    finite_mask = np.isfinite(labels)
    if finite_mask.any():
        finite_vals = labels[finite_mask]
        rounded = np.rint(finite_vals)
        if np.all(np.abs(finite_vals - rounded) <= 1e-8):
            n_unique = int(np.unique(rounded.astype(np.int64)).size)
            max_classes = max(10, int(np.sqrt(float(finite_vals.size))) + 1)
            if 2 <= n_unique <= max_classes:
                return _TASK_CLASSIFICATION

    return _TASK_REGRESSION


def _classification_labels(y: np.ndarray) -> np.ndarray:
    arr = np.asarray(y)
    if arr.ndim == 2:
        if arr.shape[1] != 1:
            raise ValueError(
                "Classification targets must be rank-1 or single-column rank-2 arrays, "
                f"got shape={arr.shape!r}."
            )
        arr = arr[:, 0]
    if arr.ndim != 1:
        raise ValueError(f"Classification targets must be rank-1, got shape={arr.shape!r}.")
    return np.rint(arr).astype(np.int64)


def _regression_target_matrix(y: np.ndarray) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float64)
    if arr.ndim == 1:
        return arr[:, np.newaxis]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Regression targets must be rank-1 or rank-2, got shape={arr.shape!r}.")


def _classification_targets_for_splits(
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_train_labels = _classification_labels(y_train)
    y_test_labels = _classification_labels(y_test)
    all_labels = np.concatenate([y_train_labels, y_test_labels], axis=0)
    classes, inverse = np.unique(all_labels, return_inverse=True)
    n_classes = int(classes.size)
    if n_classes <= 0:
        raise ValueError("Classification targets must include at least one class.")
    eye = np.eye(n_classes, dtype=np.float64)
    y_train_targets = eye[inverse[: y_train_labels.shape[0]]]
    y_test_targets = eye[inverse[y_train_labels.shape[0] :]]
    return y_train_targets, y_test_targets, all_labels


def _ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_pred: np.ndarray,
    *,
    ridge_lambda: float = _RIDGE_LAMBDA,
) -> np.ndarray:
    train_design = np.concatenate(
        [np.ones((x_train.shape[0], 1), dtype=np.float64), x_train],
        axis=1,
    )
    pred_design = np.concatenate(
        [np.ones((x_pred.shape[0], 1), dtype=np.float64), x_pred],
        axis=1,
    )
    gram = train_design.T @ train_design
    reg = np.eye(gram.shape[0], dtype=np.float64) * float(ridge_lambda)
    reg[0, 0] = 0.0
    rhs = train_design.T @ y_train
    try:
        weights = np.linalg.solve(gram + reg, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(gram + reg) @ rhs
    return pred_design @ weights


def _bootstrap_wins_ratio(pred: np.ndarray, target: np.ndarray, baseline: np.ndarray) -> float:
    n_rows = int(target.shape[0])
    if n_rows <= 0:
        raise ValueError("Bootstrap wins ratio requires at least one row.")
    rng = np.random.default_rng(0)
    wins = 0
    for _ in range(_NUM_BOOTSTRAP):
        idx = rng.integers(0, n_rows, size=n_rows)
        sampled_target = target[idx]
        sampled_pred = pred[idx]
        mse_pred = float(np.mean((sampled_pred - sampled_target) ** 2))
        mse_base = float(np.mean((baseline - sampled_target) ** 2))
        wins += int(mse_pred < mse_base)
    return float(wins / _NUM_BOOTSTRAP)


def _compute_linearity_proxy(
    *,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    task: str,
) -> float | None:
    if x_train.shape[0] <= 0 or x_test.shape[0] <= 0:
        return None

    if task == _TASK_CLASSIFICATION:
        y_train_targets, y_test_targets, _ = _classification_targets_for_splits(y_train, y_test)
    elif task == _TASK_REGRESSION:
        y_train_targets = _regression_target_matrix(y_train)
        y_test_targets = _regression_target_matrix(y_test)
    else:
        raise ValueError(f"Unsupported task '{task}'.")

    pred_test = _ridge_predict(x_train, y_train_targets, x_test)
    baseline = np.mean(y_train_targets, axis=0, keepdims=True)
    wins_ratio = _bootstrap_wins_ratio(pred=pred_test, target=y_test_targets, baseline=baseline)
    return _finite_or_none(wins_ratio)


def _compute_wins_ratio_proxy(*, x: np.ndarray, y: np.ndarray, task: str) -> float | None:
    try:
        x_tensor = torch.as_tensor(x, dtype=torch.float32, device="cpu")
        if task == _TASK_CLASSIFICATION:
            labels = _classification_labels(y)
            _, dense = np.unique(labels, return_inverse=True)
            y_tensor = torch.as_tensor(dense.astype(np.int64), dtype=torch.int64, device="cpu")
        elif task == _TASK_REGRESSION:
            y_matrix = _regression_target_matrix(y).astype(np.float32, copy=False)
            if y_matrix.shape[1] == 1:
                y_tensor = torch.as_tensor(y_matrix[:, 0], dtype=torch.float32, device="cpu")
            else:
                y_tensor = torch.as_tensor(y_matrix, dtype=torch.float32, device="cpu")
        else:
            raise ValueError(f"Unsupported task '{task}'.")

        _, details = apply_torch_rf_filter(
            x_tensor,
            y_tensor,
            task=task,
            seed=0,
            threshold=0.0,
        )
    except Exception:
        return None

    value = details.get("wins_ratio")
    if isinstance(value, (int, float)):
        return _finite_or_none(float(value))
    return None


def _compute_snr_proxy_db(
    *,
    x_train: np.ndarray,
    x_all: np.ndarray,
    y_train: np.ndarray,
    y_all: np.ndarray,
    task: str,
) -> float | None:
    if x_train.shape[0] <= 0:
        return None

    if task == _TASK_CLASSIFICATION:
        y_train_labels = _classification_labels(y_train)
        y_all_labels = _classification_labels(y_all)
        classes, y_all_inverse = np.unique(y_all_labels, return_inverse=True)
        n_classes = int(classes.size)
        if n_classes <= 0:
            return None
        y_train_inverse = np.searchsorted(classes, y_train_labels)
        eye = np.eye(n_classes, dtype=np.float64)
        y_train_target = eye[y_train_inverse]
        y_all_target = eye[y_all_inverse]
    elif task == _TASK_REGRESSION:
        y_train_target = _regression_target_matrix(y_train)
        y_all_target = _regression_target_matrix(y_all)
    else:
        raise ValueError(f"Unsupported task '{task}'.")

    pred_all = _ridge_predict(x_train, y_train_target, x_all)
    residual = y_all_target - pred_all
    signal_var = float(np.var(pred_all))
    noise_var = float(np.var(residual))
    if not math.isfinite(signal_var) or not math.isfinite(noise_var):
        return None
    if signal_var <= 0.0:
        return None
    noise_var = max(noise_var, _EPS)
    snr = signal_var / noise_var
    if not math.isfinite(snr) or snr <= 0.0:
        return None
    return float(10.0 * math.log10(snr))


def _pearson_abs_stats(x: np.ndarray) -> tuple[float | None, float | None]:
    if x.shape[1] < 2:
        return None, None
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(x, rowvar=False)
    if corr.ndim != 2:
        return None, None
    upper = np.triu_indices(corr.shape[0], k=1)
    values = np.abs(corr[upper])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None, None
    return float(np.mean(values)), float(np.max(values))


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    ranks = np.full(arr.shape[0], np.nan, dtype=np.float64)
    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return ranks

    finite_values = arr[finite_mask]
    order = np.argsort(finite_values, kind="mergesort")
    sorted_values = finite_values[order]
    finite_ranks = np.empty(finite_values.shape[0], dtype=np.float64)

    i = 0
    n = sorted_values.shape[0]
    while i < n:
        j = i + 1
        while j < n and sorted_values[j] == sorted_values[i]:
            j += 1
        mean_rank = (i + j - 1) * 0.5 + 1.0
        finite_ranks[order[i:j]] = mean_rank
        i = j

    ranks[finite_mask] = finite_ranks
    return ranks


def _spearman_abs_stats(x: np.ndarray) -> tuple[float | None, float | None]:
    ranked = np.empty_like(x, dtype=np.float64)
    for i in range(x.shape[1]):
        ranked[:, i] = _rankdata_average(x[:, i])
    return _pearson_abs_stats(ranked)


def _class_imbalance_stats(y_labels: np.ndarray) -> tuple[float | None, float | None]:
    if y_labels.size == 0:
        return None, None
    _, counts = np.unique(y_labels, return_counts=True)
    if counts.size == 0:
        return None, None
    probs = counts.astype(np.float64) / float(np.sum(counts))
    entropy = float(-np.sum(probs * np.log(np.clip(probs, _EPS, None))))
    positive_counts = counts[counts > 0]
    if positive_counts.size == 0:
        return entropy, None
    ratio = float(np.max(positive_counts) / np.min(positive_counts))
    return entropy, ratio


def _categorical_cardinality_stats(
    x_all: np.ndarray,
    feature_types: list[str],
) -> tuple[int, float, int | None, float | None, int | None]:
    n_features = int(x_all.shape[1])
    if len(feature_types) != n_features:
        raise ValueError(
            f"feature_types length must match feature count: {len(feature_types)} != {n_features}"
        )
    cat_indices = [i for i, kind in enumerate(feature_types) if kind == "cat"]
    n_categorical = int(len(cat_indices))
    cat_ratio = float(n_categorical / max(1, n_features))
    if n_categorical == 0:
        return n_categorical, cat_ratio, None, None, None

    cardinalities: list[int] = []
    for idx in cat_indices:
        col = x_all[:, idx]
        finite = col[np.isfinite(col)]
        cardinalities.append(int(np.unique(finite).size))

    if not cardinalities:
        return n_categorical, cat_ratio, None, None, None
    return (
        n_categorical,
        cat_ratio,
        int(min(cardinalities)),
        float(np.mean(cardinalities)),
        int(max(cardinalities)),
    )
