"""Postprocessing hooks."""

from __future__ import annotations

from typing import Any, Literal, overload

import torch

from dagzoo.config import (
    DatasetConfig,
    MISSINGNESS_MECHANISM_NONE,
    normalize_missing_mechanism,
)
from dagzoo.rng import SeedManager
from dagzoo.sampling import sample_missingness_mask


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
    numeric_indices = [i for i, t in enumerate(feature_types) if t != "cat"]
    if not numeric_indices:
        return out

    # Standardize all numeric columns in one batched pass.
    numeric_index = torch.tensor(numeric_indices, device=out.device, dtype=torch.long)
    numeric = out.index_select(dim=1, index=numeric_index)
    quantiles = torch.quantile(
        numeric.float(),
        torch.tensor([0.01, 0.99], device=numeric.device),
        dim=0,
    )
    lo = quantiles[0].unsqueeze(0)
    hi = quantiles[1].unsqueeze(0)
    numeric = torch.clamp(numeric, lo, hi)
    mu = torch.mean(numeric, dim=0, keepdim=True)
    sd = torch.std(numeric, dim=0, correction=0, keepdim=True).clamp_min(1e-6)
    out[:, numeric_index] = (numeric - mu) / sd
    return out


def _has_at_least_two_classes(y: torch.Tensor) -> bool:
    """Return whether a non-empty label tensor contains at least two classes."""

    y_i64 = y.to(torch.int64)
    return bool(torch.min(y_i64) != torch.max(y_i64))


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
    preserve_feature_schema: bool = False,
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
    preserve_feature_schema: bool = False,
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
    preserve_feature_schema: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[int]]
):
    """
    Apply postprocessing to train/test splits.

    - Remove constant columns
    - Standardize non-categorical columns
    - Permute column order
    - Permute class labels for classification
    """
    _ = device

    x_all = torch.cat([x_train, x_test], dim=0).to(torch.float32)
    if preserve_feature_schema:
        feature_types = list(feature_types)
        feature_index_map = [int(i) for i in range(x_all.shape[1])]
    else:
        x_all, feature_types, feature_index_map = _remove_constant_columns(x_all, feature_types)
    x_all = _clip_and_standardize(x_all, feature_types)

    if not preserve_feature_schema:
        # Keep permutation RNG on CPU while applying it on the feature tensor device.
        perm_cpu = torch.randperm(x_all.shape[1], generator=generator, device="cpu")
        perm_list = [int(i) for i in perm_cpu.tolist()]
        perm = perm_cpu.to(device=x_all.device)
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
        y_all = torch.clamp(y_all, lo, hi)
        mu = torch.mean(y_all)
        sd = torch.std(y_all, correction=0).clamp_min(1e-6)
        y_all = (y_all - mu) / sd
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

    y_all_original = torch.cat([y_train, y_test], dim=0).to(torch.int64)
    classes, inverse = torch.unique(y_all_original, sorted=True, return_inverse=True)
    y_all_original_dense = inverse.to(torch.int64)

    # Apply a class-id permutation directly in dense label space.
    perm_dense_cpu = torch.randperm(classes.numel(), generator=generator, device="cpu")
    perm_dense = perm_dense_cpu.to(device=inverse.device)
    y_all_permuted_dense = perm_dense[inverse].to(torch.int64)
    y_train_candidate = y_all_permuted_dense[:n_train]
    y_test_candidate = y_all_permuted_dense[n_train:]

    if not _has_at_least_two_classes(y_train_candidate) or not _has_at_least_two_classes(
        y_test_candidate
    ):
        # Fall back to original labels if permutation collapses effective class diversity.
        y_all_dense = y_all_original_dense
    else:
        y_all_dense = y_all_permuted_dense
    y_train_p = y_all_dense[:n_train]
    y_test_p = y_all_dense[n_train:]

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
