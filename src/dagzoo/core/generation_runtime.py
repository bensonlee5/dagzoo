"""Shared generation finalization helpers for canonical fixed-layout execution."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch

from dagzoo.config import GeneratorConfig
from dagzoo.core.generation_context import _split_permutation_seed
from dagzoo.core.layout_types import LayoutPlan
from dagzoo.core.metadata import _build_lineage_metadata, _build_shift_metadata
from dagzoo.core.noise_runtime import (
    NoiseRuntimeSelection,
    _build_noise_distribution_metadata,
)
from dagzoo.core.shift import ShiftRuntimeParams
from dagzoo.core.validation import _classification_split_valid, _stratified_split_indices
from dagzoo.postprocess import inject_missingness, postprocess_dataset
from dagzoo.types import DatasetBundle


def _classification_class_structure(
    *,
    y_train: torch.Tensor,
    y_test: torch.Tensor,
    n_classes_sampled: int,
) -> dict[str, Any]:
    """Build classification label-structure metadata for one emitted bundle."""

    y_train_i64 = y_train.to(torch.int64)
    y_test_i64 = y_test.to(torch.int64)
    y_all = torch.cat([y_train_i64, y_test_i64], dim=0)
    unique_all = torch.unique(y_all, sorted=True)
    n_classes_realized = int(unique_all.numel())
    labels_contiguous = bool(
        torch.equal(
            unique_all,
            torch.arange(n_classes_realized, dtype=unique_all.dtype, device=unique_all.device),
        )
    )
    train_classes = torch.unique(y_train_i64, sorted=True)
    test_classes = torch.unique(y_test_i64, sorted=True)

    return {
        "n_classes_sampled": int(n_classes_sampled),
        "n_classes_realized": int(n_classes_realized),
        "labels_contiguous": bool(labels_contiguous),
        "train_test_class_match": bool(torch.equal(train_classes, test_classes)),
        "min_label": int(unique_all[0].item()) if n_classes_realized > 0 else None,
        "max_label": int(unique_all[-1].item()) if n_classes_realized > 0 else None,
    }


def _parent_node_indices(adjacency: torch.Tensor, node_index: int) -> list[int]:
    """Return parent indices for node `node_index` from `adjacency[src, dst]`."""

    parent_indices = torch.where(adjacency[:, node_index])[0].tolist()
    return sorted(int(parent_index) for parent_index in parent_indices)


def _finalize_generated_tensors(
    config: GeneratorConfig,
    layout: LayoutPlan,
    *,
    seed: int,
    attempt: int,
    attempts_used: int,
    device: str,
    n_train: int,
    n_test: int,
    requested_device: str,
    resolved_device: str,
    device_fallback_reason: str | None,
    x: torch.Tensor,
    y: torch.Tensor,
    aux_meta: dict[str, Any],
    shift_params: ShiftRuntimeParams,
    noise_runtime_selection: NoiseRuntimeSelection,
    dtype: torch.dtype,
    preserve_feature_schema: bool = False,
) -> DatasetBundle:
    """Finalize one raw `x`/`y` pair into the standard dataset bundle contract."""

    split_postprocess_generator = torch.Generator(device="cpu")
    split_postprocess_generator.manual_seed(_split_permutation_seed(seed, attempt))

    if config.dataset.task == "classification":
        try:
            train_idx_cpu, test_idx_cpu = _stratified_split_indices(
                y.to(device="cpu"),
                n_train,
                split_postprocess_generator,
                "cpu",
            )
        except ValueError as exc:
            if str(exc).startswith("infeasible_stratified_split"):
                raise ValueError("invalid_class_split") from exc
            raise
        train_idx = train_idx_cpu.to(device=x.device)
        test_idx = test_idx_cpu.to(device=x.device)
        x_train_t, x_test_t = x[train_idx], x[test_idx]
        y_train_t, y_test_t = y[train_idx], y[test_idx]
    else:
        order_cpu = torch.randperm(
            x.shape[0],
            generator=split_postprocess_generator,
            device="cpu",
        )
        order = order_cpu.to(device=x.device)
        x, y = x[order], y[order]
        x_train_t, x_test_t = x[:n_train], x[n_train:]
        y_train_t, y_test_t = y[:n_train], y[n_train:]

    x_train, y_train, x_test, y_test, feature_types, feature_index_map = postprocess_dataset(
        x_train_t,
        y_train_t,
        x_test_t,
        y_test_t,
        list(layout.feature_types),
        config.dataset.task,
        split_postprocess_generator,
        device,
        return_feature_index_map=True,
        preserve_feature_schema=preserve_feature_schema,
    )
    x_train, x_test, missingness_summary = inject_missingness(
        x_train,
        x_test,
        dataset_cfg=config.dataset,
        seed=seed,
        attempt=attempt,
        device=device,
    )

    if config.dataset.task == "classification" and not _classification_split_valid(y_train, y_test):
        raise ValueError("invalid_class_split")

    x_train = x_train.to(device=device, dtype=dtype)
    x_test = x_test.to(device=device, dtype=dtype)
    y_dtype = torch.int64 if config.dataset.task == "classification" else dtype
    y_train = y_train.to(device=device, dtype=y_dtype)
    y_test = y_test.to(device=device, dtype=y_dtype)
    class_structure: dict[str, Any] | None = None
    n_classes: int | None = None
    if config.dataset.task == "classification":
        class_structure = _classification_class_structure(
            y_train=y_train,
            y_test=y_test,
            n_classes_sampled=int(layout.n_classes),
        )
        n_classes = int(class_structure["n_classes_realized"])
    shift_metadata = _build_shift_metadata(
        shift_params=shift_params,
        function_family_mix=config.mechanism.function_family_mix,
    )
    filter_metadata = aux_meta.get("filter", {})
    if isinstance(filter_metadata, dict):
        filter_metadata = dict(filter_metadata)
    else:
        filter_metadata = {"mode": "deferred", "status": "not_run"}

    config_payload = asdict(config)
    dataset_payload = config_payload.get("dataset")
    if isinstance(dataset_payload, dict):
        dataset_payload["n_train"] = int(n_train)
        dataset_payload["n_test"] = int(n_test)

    metadata = {
        "backend": "torch",
        "device": device,
        "requested_device": str(requested_device),
        "resolved_device": str(resolved_device),
        "device_fallback_reason": device_fallback_reason,
        "compute_backend": "torch_appendix_full",
        "n_features": int(x_train.shape[1]),
        "n_categorical_features": int(sum(1 for t in feature_types if t == "cat")),
        "n_classes": n_classes,
        "graph_nodes": int(layout.graph_nodes),
        "graph_edges": int(layout.graph_edges),
        "graph_depth_nodes": int(layout.graph_depth_nodes),
        "graph_edge_density": float(layout.graph_edge_density),
        "lineage": _build_lineage_metadata(layout, feature_index_map=feature_index_map),
        "seed": seed,
        "attempt_used": attempt,
        "filter": filter_metadata,
        "shift": shift_metadata,
        "noise_distribution": _build_noise_distribution_metadata(noise_runtime_selection),
        "generation_attempts": {
            "total_attempts": int(attempts_used),
            "retry_count": int(max(0, attempts_used - 1)),
            "filter_attempts": 0,
            "filter_rejections": 0,
            "filter_rejection_rate": None,
        },
        "config": config_payload,
    }
    if missingness_summary is not None:
        metadata["missingness"] = missingness_summary
    if class_structure is not None:
        metadata["class_structure"] = class_structure
    return DatasetBundle(
        X_train=x_train,
        y_train=y_train,
        X_test=x_test,
        y_test=y_test,
        feature_types=feature_types,
        metadata=metadata,
        runtime_metrics={},
    )
