"""Shared generation finalization helpers for canonical fixed-layout execution."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
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
from dagzoo.core.validation import (
    InvalidClassSplitError,
    InfeasibleStratifiedSplitError,
    _classification_split_valid,
    _stratified_split_indices,
)
from dagzoo.postprocess.postprocess import (
    inject_missingness,
    postprocess_dataset,
    postprocess_fixed_schema_batch,
)
from dagzoo.types import DatasetBundle


@dataclass(slots=True)
class _FixedSchemaFinalizationContext:
    """Cached metadata used by fixed-schema chunk finalization."""

    config_payload: dict[str, Any]
    shift_metadata: dict[str, Any]
    feature_types: list[str]
    feature_index_map: list[int]


def _config_payload_for_metadata(
    config: GeneratorConfig,
    *,
    n_train: int,
    n_test: int,
) -> dict[str, Any]:
    """Serialize config metadata while omitting unset fixed-layout target-cell overrides."""

    config_payload = asdict(config)
    dataset_payload = config_payload.get("dataset")
    if isinstance(dataset_payload, dict):
        dataset_payload["n_train"] = int(n_train)
        dataset_payload["n_test"] = int(n_test)

    runtime_payload = config_payload.get("runtime")
    if (
        isinstance(runtime_payload, dict)
        and runtime_payload.get("fixed_layout_target_cells") is None
    ):
        runtime_payload.pop("fixed_layout_target_cells", None)
    return config_payload


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


def _make_split_postprocess_generator(seed: int, attempt: int) -> torch.Generator:
    """Create the shared CPU generator used for splitting and postprocess permutation."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(_split_permutation_seed(seed, attempt))
    return generator


def _resolve_split_indices(
    y: torch.Tensor,
    *,
    task: str,
    n_train: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve one dataset's train/test split indices using the shared CPU generator."""

    if task == "classification":
        return _stratified_split_indices(
            y.to(device="cpu"),
            n_train,
            generator,
            "cpu",
        )

    total_rows = int(y.shape[0])
    order_cpu = torch.randperm(
        total_rows,
        generator=generator,
        device="cpu",
    )
    return order_cpu[:n_train], order_cpu[n_train:]


def _split_raw_tensors(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    train_idx_cpu: torch.Tensor,
    test_idx_cpu: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split one raw feature/target pair using CPU indices applied on the tensor device."""

    train_idx = train_idx_cpu.to(device=x.device)
    test_idx = test_idx_cpu.to(device=x.device)
    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]


def _normalized_filter_metadata(aux_meta: dict[str, Any]) -> dict[str, Any]:
    """Return the emitted filter metadata shape for one dataset."""

    filter_metadata = aux_meta.get("filter", {})
    if isinstance(filter_metadata, dict):
        return dict(filter_metadata)
    return {"mode": "deferred", "status": "not_run"}


def _parent_node_indices(adjacency: torch.Tensor, node_index: int) -> list[int]:
    """Return parent indices for node `node_index` from `adjacency[src, dst]`."""

    parent_indices = torch.where(adjacency[:, node_index])[0].tolist()
    return sorted(int(parent_index) for parent_index in parent_indices)


def _build_fixed_schema_finalization_context(
    config: GeneratorConfig,
    layout: LayoutPlan,
    *,
    n_train: int,
    n_test: int,
    shift_params: ShiftRuntimeParams,
) -> _FixedSchemaFinalizationContext:
    """Build cached metadata for fixed-schema bundle finalization."""

    config_payload = _config_payload_for_metadata(
        config,
        n_train=n_train,
        n_test=n_test,
    )
    feature_types = [str(feature_type) for feature_type in list(layout.feature_types)]
    return _FixedSchemaFinalizationContext(
        config_payload=config_payload,
        shift_metadata=_build_shift_metadata(
            shift_params=shift_params,
            function_family_mix=config.mechanism.function_family_mix,
        ),
        feature_types=feature_types,
        feature_index_map=[int(i) for i in range(int(layout.n_features))],
    )


def _build_bundle_metadata(
    layout: LayoutPlan,
    *,
    feature_types: list[str],
    feature_index_map: list[int],
    config_payload: dict[str, Any],
    shift_metadata: dict[str, Any],
    seed: int,
    attempt: int,
    attempts_used: int,
    device: str,
    requested_device: str,
    resolved_device: str,
    device_fallback_reason: str | None,
    aux_meta: dict[str, Any],
    noise_runtime_selection: NoiseRuntimeSelection,
    missingness_summary: dict[str, Any] | None,
    class_structure: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build emitted dataset metadata for scalar and batched finalization paths."""

    n_classes = None if class_structure is None else int(class_structure["n_classes_realized"])
    metadata = {
        "backend": "torch",
        "device": str(device),
        "requested_device": str(requested_device),
        "resolved_device": str(resolved_device),
        "device_fallback_reason": device_fallback_reason,
        "compute_backend": "torch_appendix_full",
        "n_features": int(len(feature_types)),
        "n_categorical_features": int(sum(1 for t in feature_types if t == "cat")),
        "n_classes": n_classes,
        "graph_nodes": int(layout.graph_nodes),
        "graph_edges": int(layout.graph_edges),
        "graph_depth_nodes": int(layout.graph_depth_nodes),
        "graph_edge_density": float(layout.graph_edge_density),
        "lineage": _build_lineage_metadata(layout, feature_index_map=feature_index_map),
        "seed": int(seed),
        "attempt_used": int(attempt),
        "filter": _normalized_filter_metadata(aux_meta),
        "shift": copy.deepcopy(shift_metadata),
        "noise_distribution": _build_noise_distribution_metadata(noise_runtime_selection),
        "generation_attempts": {
            "total_attempts": int(attempts_used),
            "retry_count": int(max(0, attempts_used - 1)),
            "filter_attempts": 0,
            "filter_rejections": 0,
            "filter_rejection_rate": None,
        },
        "config": copy.deepcopy(config_payload),
    }
    if missingness_summary is not None:
        metadata["missingness"] = missingness_summary
    if class_structure is not None:
        metadata["class_structure"] = class_structure
    return metadata


def _finalize_processed_bundle(
    config: GeneratorConfig,
    layout: LayoutPlan,
    *,
    seed: int,
    attempt: int,
    attempts_used: int,
    device: str,
    requested_device: str,
    resolved_device: str,
    device_fallback_reason: str | None,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    feature_types: list[str],
    feature_index_map: list[int],
    aux_meta: dict[str, Any],
    config_payload: dict[str, Any],
    shift_metadata: dict[str, Any],
    noise_runtime_selection: NoiseRuntimeSelection,
    dtype: torch.dtype,
) -> DatasetBundle:
    """Finalize one already-postprocessed dataset into the emitted bundle contract."""

    x_train, x_test, missingness_summary = inject_missingness(
        x_train,
        x_test,
        dataset_cfg=config.dataset,
        seed=seed,
        attempt=attempt,
        device=device,
    )

    if config.dataset.task == "classification" and not _classification_split_valid(y_train, y_test):
        raise InvalidClassSplitError("invalid_class_split")

    x_train = x_train.to(device=device, dtype=dtype)
    x_test = x_test.to(device=device, dtype=dtype)
    y_dtype = torch.int64 if config.dataset.task == "classification" else dtype
    y_train = y_train.to(device=device, dtype=y_dtype)
    y_test = y_test.to(device=device, dtype=y_dtype)

    class_structure: dict[str, Any] | None = None
    if config.dataset.task == "classification":
        class_structure = _classification_class_structure(
            y_train=y_train,
            y_test=y_test,
            n_classes_sampled=int(layout.n_classes),
        )

    metadata = _build_bundle_metadata(
        layout,
        feature_types=feature_types,
        feature_index_map=feature_index_map,
        config_payload=config_payload,
        shift_metadata=shift_metadata,
        seed=seed,
        attempt=attempt,
        attempts_used=attempts_used,
        device=device,
        requested_device=requested_device,
        resolved_device=resolved_device,
        device_fallback_reason=device_fallback_reason,
        aux_meta=aux_meta,
        noise_runtime_selection=noise_runtime_selection,
        missingness_summary=missingness_summary,
        class_structure=class_structure,
    )
    return DatasetBundle(
        X_train=x_train,
        y_train=y_train,
        X_test=x_test,
        y_test=y_test,
        feature_types=list(feature_types),
        metadata=metadata,
        runtime_metrics={},
    )


def _finalize_generated_chunk_preserve_schema(
    config: GeneratorConfig,
    layout: LayoutPlan,
    *,
    context: _FixedSchemaFinalizationContext,
    seeds: list[int],
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
    aux_meta_batch: list[dict[str, Any]],
    noise_runtime_selection: NoiseRuntimeSelection,
    dtype: torch.dtype,
) -> list[DatasetBundle | None]:
    """Finalize one fixed-schema raw chunk while preserving scalar retry semantics."""

    if int(x.shape[0]) != len(seeds) or int(y.shape[0]) != len(seeds):
        raise ValueError("Chunk tensors must align with provided dataset seeds.")

    results: list[DatasetBundle | None] = [None] * len(seeds)
    valid_positions: list[int] = []
    train_idx_cpu_list: list[torch.Tensor] = []
    test_idx_cpu_list: list[torch.Tensor] = []
    postprocess_generator_states: list[torch.Tensor] = []
    for batch_index, seed in enumerate(seeds):
        split_postprocess_generator = _make_split_postprocess_generator(seed, attempt)
        try:
            train_idx_cpu, test_idx_cpu = _resolve_split_indices(
                y[batch_index],
                task=config.dataset.task,
                n_train=n_train,
                generator=split_postprocess_generator,
            )
        except InfeasibleStratifiedSplitError:
            continue

        valid_positions.append(int(batch_index))
        train_idx_cpu_list.append(train_idx_cpu)
        test_idx_cpu_list.append(test_idx_cpu)
        postprocess_generator_states.append(split_postprocess_generator.get_state())

    if not valid_positions:
        return results

    valid_index = torch.as_tensor(valid_positions, dtype=torch.long, device=x.device)
    x_valid = x.index_select(0, valid_index)
    y_valid = y.index_select(0, valid_index)
    train_idx = torch.stack(train_idx_cpu_list).to(device=x.device)
    test_idx = torch.stack(test_idx_cpu_list).to(device=x.device)

    x_train_t = torch.gather(
        x_valid,
        1,
        train_idx.unsqueeze(-1).expand(-1, -1, int(x.shape[2])),
    )
    x_test_t = torch.gather(
        x_valid,
        1,
        test_idx.unsqueeze(-1).expand(-1, -1, int(x.shape[2])),
    )
    y_train_t = torch.gather(y_valid, 1, train_idx)
    y_test_t = torch.gather(y_valid, 1, test_idx)

    x_train, y_train, x_test, y_test = postprocess_fixed_schema_batch(
        x_train_t,
        y_train_t,
        x_test_t,
        y_test_t,
        list(context.feature_types),
        config.dataset.task,
        postprocess_generator_states=postprocess_generator_states,
    )
    for local_index, batch_index in enumerate(valid_positions):
        try:
            results[batch_index] = _finalize_processed_bundle(
                config,
                layout,
                seed=seeds[batch_index],
                attempt=attempt,
                attempts_used=attempts_used,
                device=device,
                requested_device=requested_device,
                resolved_device=resolved_device,
                device_fallback_reason=device_fallback_reason,
                x_train=x_train[local_index],
                y_train=y_train[local_index],
                x_test=x_test[local_index],
                y_test=y_test[local_index],
                feature_types=context.feature_types,
                feature_index_map=context.feature_index_map,
                aux_meta=aux_meta_batch[batch_index],
                config_payload=context.config_payload,
                shift_metadata=context.shift_metadata,
                noise_runtime_selection=noise_runtime_selection,
                dtype=dtype,
            )
        except InvalidClassSplitError:
            continue

    return results


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

    split_postprocess_generator = _make_split_postprocess_generator(seed, attempt)
    try:
        train_idx_cpu, test_idx_cpu = _resolve_split_indices(
            y,
            task=config.dataset.task,
            n_train=n_train,
            generator=split_postprocess_generator,
        )
    except InfeasibleStratifiedSplitError as exc:
        raise InvalidClassSplitError("invalid_class_split") from exc

    x_train_t, y_train_t, x_test_t, y_test_t = _split_raw_tensors(
        x,
        y,
        train_idx_cpu=train_idx_cpu,
        test_idx_cpu=test_idx_cpu,
    )

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
    shift_metadata = _build_shift_metadata(
        shift_params=shift_params,
        function_family_mix=config.mechanism.function_family_mix,
    )
    config_payload = _config_payload_for_metadata(
        config,
        n_train=n_train,
        n_test=n_test,
    )
    return _finalize_processed_bundle(
        config,
        layout,
        seed=seed,
        attempt=attempt,
        attempts_used=attempts_used,
        device=device,
        requested_device=requested_device,
        resolved_device=resolved_device,
        device_fallback_reason=device_fallback_reason,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        feature_types=feature_types,
        feature_index_map=feature_index_map,
        aux_meta=aux_meta,
        config_payload=config_payload,
        shift_metadata=shift_metadata,
        noise_runtime_selection=noise_runtime_selection,
        dtype=dtype,
    )
