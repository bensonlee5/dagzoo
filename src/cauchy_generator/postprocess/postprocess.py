"""Postprocessing hooks (Appendix E.13)."""

from __future__ import annotations

import torch


def _remove_constant_columns(
    x: torch.Tensor, feature_types: list[str]
) -> tuple[torch.Tensor, list[str]]:
    """Drop columns with near-zero variance and align feature type metadata."""

    keep = torch.std(x, dim=0, correction=0) > 1e-12
    if not torch.any(keep):
        raise ValueError("All columns are constant after generation.")
    kept_types = [t for t, k in zip(feature_types, keep.tolist(), strict=True) if k]
    return x[:, keep], kept_types


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


def postprocess_dataset(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    feature_types: list[str],
    task: str,
    generator: torch.Generator,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """
    Apply E.13-style postprocessing to train/test splits.

    - Remove constant columns
    - Standardize non-categorical columns
    - Permute column order
    - Permute class labels for classification
    """

    x_all = torch.cat([x_train, x_test], dim=0).to(torch.float32)
    x_all, feature_types = _remove_constant_columns(x_all, feature_types)
    x_all = _clip_and_standardize(x_all, feature_types)

    perm = torch.randperm(x_all.shape[1], generator=generator, device=device)
    x_all = x_all[:, perm]
    feature_types = [feature_types[int(i)] for i in perm.tolist()]

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
        return x_train_p, y_all[:n_train], x_test_p, y_all[n_train:], feature_types

    y_all = torch.cat([y_train, y_test], dim=0).to(torch.int64)
    y_all = _permute_classes(y_all, generator, device)
    y_train_p = y_all[:n_train]
    y_test_p = y_all[n_train:]

    if torch.unique(y_train_p).numel() < 2 or torch.unique(y_test_p).numel() < 2:
        # Fall back to original labels if permutation collapses effective class diversity.
        y_train_p = y_train.to(torch.int64)
        y_test_p = y_test.to(torch.int64)

    return x_train_p, y_train_p, x_test_p, y_test_p, feature_types
