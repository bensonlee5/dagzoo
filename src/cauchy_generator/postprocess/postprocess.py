"""Postprocessing hooks (Appendix E.13)."""

from __future__ import annotations

from typing import Any, Literal, overload

import torch

from cauchy_generator.config import (
    DatasetConfig,
    MISSINGNESS_MECHANISM_NONE,
    normalize_missing_mechanism,
)
from cauchy_generator.rng import SeedManager
from cauchy_generator.sampling import sample_missingness_mask


def _remove_constant_columns(
    x: torch.Tensor, feature_types: list[str]
) -> tuple[torch.Tensor, list[str], list[int]]:
    """Drop columns with near-zero variance and align feature type metadata."""

    keep = torch.std(x, dim=0, correction=0) > 1e-12
    if not torch.any(keep):
        raise ValueError("All columns are constant after generation.")
    keep_indices = [int(i) for i, keep_col in enumerate(keep.tolist()) if keep_col]
    kept_types = [feature_types[i] for i in keep_indices]
    return x[:, keep], kept_types, keep_indices


def _clip_and_standardize(x: torch.Tensor, feature_types: list[str]) -> torch.Tensor:
    """Clip numeric outliers and standardize numeric columns."""

    out = x.clone()
    for i, t in enumerate(feature_types):
        if t == "cat":
            continue
        col = out[:, i]
        q = torch.quantile(col.float(), torch.tensor([0.01, 0.99], device=col.device))
        lo, hi = q[0], q[1]
        col = torch.clamp(col, lo.item(), hi.item())
        mu = float(torch.mean(col))
        sd = float(torch.std(col, correction=0))
        out[:, i] = (col - mu) / max(sd, 1e-6)
    return out


def _permute_classes(y: torch.Tensor, generator: torch.Generator, device: str) -> torch.Tensor:
    """Permute class indices while preserving class counts."""

    classes = torch.unique(y)
    perm = classes[torch.randperm(classes.numel(), generator=generator, device=device)]
    remapped = torch.empty_like(y)
    for src, dst in zip(classes.tolist(), perm.tolist(), strict=True):
        remapped[y == src] = int(dst)
    return remapped


@overload
def postprocess_dataset(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    feature_types: list[str],
    task: str,
    generator: torch.Generator,
    device: str,
    *,
    return_feature_index_map: Literal[False] = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]: ...


@overload
def postprocess_dataset(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    feature_types: list[str],
    task: str,
    generator: torch.Generator,
    device: str,
    *,
    return_feature_index_map: Literal[True],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[int]]: ...


def postprocess_dataset(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    feature_types: list[str],
    task: str,
    generator: torch.Generator,
    device: str,
    *,
    return_feature_index_map: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[int]]
):
    """
    Apply E.13-style postprocessing to train/test splits.

    - Remove constant columns
    - Standardize non-categorical columns
    - Permute column order
    - Permute class labels for classification
    """

    x_all = torch.cat([x_train, x_test], dim=0).to(torch.float32)
    x_all, feature_types, feature_index_map = _remove_constant_columns(x_all, feature_types)
    x_all = _clip_and_standardize(x_all, feature_types)

    perm = torch.randperm(x_all.shape[1], generator=generator, device=device)
    perm_list = [int(i) for i in perm.tolist()]
    x_all = x_all[:, perm]
    feature_types = [feature_types[i] for i in perm_list]
    feature_index_map = [feature_index_map[i] for i in perm_list]

    n_train = x_train.shape[0]
    x_train_p = x_all[:n_train]
    x_test_p = x_all[n_train:]

    if task == "regression":
        y_all = torch.cat([y_train, y_test], dim=0).to(torch.float32)
        q = torch.quantile(y_all.float(), torch.tensor([0.01, 0.99], device=y_all.device))
        lo, hi = q[0], q[1]
        y_all = torch.clamp(y_all, lo.item(), hi.item())
        mu = float(torch.mean(y_all))
        sd = float(torch.std(y_all, correction=0))
        y_all = (y_all - mu) / max(sd, 1e-6)
        if return_feature_index_map:
            return (
                x_train_p,
                y_all[:n_train],
                x_test_p,
                y_all[n_train:],
                feature_types,
                feature_index_map,
            )
        return x_train_p, y_all[:n_train], x_test_p, y_all[n_train:], feature_types

    y_all = torch.cat([y_train, y_test], dim=0).to(torch.int64)
    y_all = _permute_classes(y_all, generator, device)
    y_train_p = y_all[:n_train]
    y_test_p = y_all[n_train:]

    if torch.unique(y_train_p).numel() < 2 or torch.unique(y_test_p).numel() < 2:
        # Fall back to original labels if permutation collapses effective class diversity.
        y_train_p = y_train.to(torch.int64)
        y_test_p = y_test.to(torch.int64)

    if return_feature_index_map:
        return x_train_p, y_train_p, x_test_p, y_test_p, feature_types, feature_index_map
    return x_train_p, y_train_p, x_test_p, y_test_p, feature_types


def inject_missingness(
    x_train: torch.Tensor,
    x_test: torch.Tensor,
    *,
    dataset_cfg: DatasetConfig,
    seed: int,
    attempt: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any] | None]:
    """
    Inject configured missingness into train/test feature tensors.

    Missing values are encoded as NaN and summary stats are returned for metadata.
    """

    missing_rate = float(dataset_cfg.missing_rate)
    mechanism = normalize_missing_mechanism(dataset_cfg.missing_mechanism)
    enabled = missing_rate > 0.0 and mechanism != MISSINGNESS_MECHANISM_NONE
    if not enabled:
        return x_train, x_test, None

    run_manager = SeedManager(seed)
    train_manager = SeedManager(run_manager.child("missingness", attempt, "train"))
    test_manager = SeedManager(run_manager.child("missingness", attempt, "test"))

    train_mask = sample_missingness_mask(
        x_train,
        dataset_cfg=dataset_cfg,
        seed_manager=train_manager,
        device=device,
    )
    test_mask = sample_missingness_mask(
        x_test,
        dataset_cfg=dataset_cfg,
        seed_manager=test_manager,
        device=device,
    )

    x_train_missing = x_train.masked_fill(train_mask, float("nan"))
    x_test_missing = x_test.masked_fill(test_mask, float("nan"))

    missing_count_train = int(train_mask.sum().item())
    missing_count_test = int(test_mask.sum().item())
    train_total = max(1, int(train_mask.numel()))
    test_total = max(1, int(test_mask.numel()))
    total = train_total + test_total
    missing_count_overall = missing_count_train + missing_count_test

    summary: dict[str, Any] = {
        "enabled": True,
        "mechanism": mechanism,
        "target_rate": float(missing_rate),
        "realized_rate_train": float(missing_count_train / train_total),
        "realized_rate_test": float(missing_count_test / test_total),
        "realized_rate_overall": float(missing_count_overall / total),
        "missing_count_train": missing_count_train,
        "missing_count_test": missing_count_test,
        "missing_count_overall": missing_count_overall,
    }
    return x_train_missing, x_test_missing, summary
