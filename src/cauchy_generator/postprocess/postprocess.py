"""Postprocessing hooks (Appendix E.13)."""

from __future__ import annotations

import numpy as np


def _remove_constant_columns(x: np.ndarray, feature_types: list[str]) -> tuple[np.ndarray, list[str]]:
    """Drop columns with near-zero variance and align feature type metadata."""

    keep = np.std(x, axis=0) > 1e-12
    if not np.any(keep):
        raise ValueError("All columns are constant after generation.")
    kept_types = [t for t, k in zip(feature_types, keep, strict=True) if k]
    return x[:, keep], kept_types


def _clip_and_standardize(x: np.ndarray, feature_types: list[str]) -> np.ndarray:
    """Clip numeric outliers and standardize numeric columns."""

    out = x.copy()
    for i, t in enumerate(feature_types):
        if t == "cat":
            continue
        col = out[:, i]
        lo, hi = np.percentile(col, [1.0, 99.0])
        col = np.clip(col, lo, hi)
        mu = float(np.mean(col))
        sd = float(np.std(col))
        out[:, i] = (col - mu) / max(sd, 1e-6)
    return out


def _permute_classes(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute class indices while preserving class counts."""

    classes = np.unique(y)
    perm = rng.permutation(classes)
    mapping = {int(src): int(dst) for src, dst in zip(classes, perm, strict=True)}
    remapped = np.array([mapping[int(v)] for v in y], dtype=np.int64)
    return remapped


def postprocess_dataset(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    feature_types: list[str],
    task: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Apply E.13-style postprocessing to train/test splits.

    - Remove constant columns
    - Standardize non-categorical columns
    - Permute column order
    - Permute class labels for classification
    """

    x_all = np.concatenate([x_train, x_test], axis=0).astype(np.float32)
    x_all, feature_types = _remove_constant_columns(x_all, feature_types)
    x_all = _clip_and_standardize(x_all, feature_types)

    perm = rng.permutation(np.arange(x_all.shape[1]))
    x_all = x_all[:, perm]
    feature_types = [feature_types[int(i)] for i in perm]

    n_train = x_train.shape[0]
    x_train_p = x_all[:n_train]
    x_test_p = x_all[n_train:]

    if task == "regression":
        y_all = np.concatenate([y_train, y_test], axis=0).astype(np.float32)
        lo, hi = np.percentile(y_all, [1.0, 99.0])
        y_all = np.clip(y_all, lo, hi)
        mu = float(np.mean(y_all))
        sd = float(np.std(y_all))
        y_all = (y_all - mu) / max(sd, 1e-6)
        return x_train_p, y_all[:n_train], x_test_p, y_all[n_train:], feature_types

    y_all = np.concatenate([y_train, y_test], axis=0).astype(np.int64)
    y_all = _permute_classes(y_all, rng)
    y_train_p = y_all[:n_train]
    y_test_p = y_all[n_train:]

    if len(np.unique(y_train_p)) < 2 or len(np.unique(y_test_p)) < 2:
        # Fall back to original labels if permutation collapses effective class diversity.
        y_train_p = y_train.astype(np.int64)
        y_test_p = y_test.astype(np.int64)

    return x_train_p, y_train_p, x_test_p, y_test_p, feature_types
