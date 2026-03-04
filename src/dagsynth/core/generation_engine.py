"""Torch generation engine internals for dataset synthesis."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch

from dagsynth.config import GeneratorConfig
from dagsynth.core.generation_context import (
    _attempt_seed,
    _node_spec_seed,
    _resolve_split_sizes,
    _split_permutation_seed,
    _torch_dtype,
    _validate_class_split_for_layout,
)
from dagsynth.core.layout import _build_node_specs, _sample_layout
from dagsynth.core.layout_types import LayoutPlan, MechanismFamily
from dagsynth.core.metadata import (
    _build_lineage_metadata,
    _build_shift_metadata,
)
from dagsynth.core.noise_runtime import (
    NoiseRuntimeSelection,
    _build_noise_distribution_metadata,
    _noise_sampling_spec,
    _resolve_noise_runtime_selection,
)
from dagsynth.core.node_pipeline import apply_node_pipeline, parse_feature_key
from dagsynth.core.shift import ShiftRuntimeParams, resolve_shift_runtime_params
from dagsynth.core.validation import _classification_split_valid, _stratified_split_indices
from dagsynth.filtering import apply_extra_trees_filter
from dagsynth.postprocess import inject_missingness, postprocess_dataset
from dagsynth.rng import SeedManager
from dagsynth.sampling.noise import NoiseSamplingSpec, sample_noise_from_spec
from dagsynth.types import DatasetBundle


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


def _generate_graph_dataset_torch(
    config: GeneratorConfig,
    layout: LayoutPlan,
    seed: int,
    device: str,
    *,
    n_rows: int,
    mechanism_logit_tilt: float = 0.0,
    function_family_mix: dict[MechanismFamily, float] | None = None,
    noise_sigma_multiplier: float = 1.0,
    noise_spec: NoiseSamplingSpec | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Generate raw X/y tensors via the Torch graph pipeline."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    num_features = int(layout.n_features)
    task = config.dataset.task
    n_classes = int(layout.n_classes)
    adjacency = layout.adjacency
    if not isinstance(adjacency, torch.Tensor):
        adjacency = torch.as_tensor(adjacency, dtype=torch.bool, device=device)
    num_nodes = int(layout.graph_nodes)

    node_outputs: dict[int, torch.Tensor] = {}
    feature_values: list[torch.Tensor | None] = [None] * num_features
    target_values: torch.Tensor | None = None

    for node_index in range(num_nodes):
        # Adjacency convention is adjacency[src, dst] (row=source, column=sink).
        parent_indices = _parent_node_indices(adjacency, node_index)
        parent_data = [
            node_outputs[parent_index]
            for parent_index in parent_indices
            if parent_index in node_outputs
        ]

        # Build specs using a deterministic per-node generator for layout consistency.
        spec_gen = torch.Generator(device="cpu")
        spec_gen.manual_seed(_node_spec_seed(seed, node_index))
        specs = _build_node_specs(node_index, layout, task, spec_gen)

        x_node, extracted = apply_node_pipeline(
            parent_data,
            n_rows,
            specs,
            generator,
            device,
            mechanism_logit_tilt=mechanism_logit_tilt,
            function_family_mix=function_family_mix,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
        node_outputs[node_index] = x_node

        for key, values in extracted.items():
            feature_index = parse_feature_key(key)
            if feature_index is not None:
                if feature_index < 0 or feature_index >= num_features:
                    raise ValueError(
                        "Extracted feature index out of range: "
                        f"{feature_index} for n_features={num_features}."
                    )
                feature_values[feature_index] = values
            elif key == "target":
                target_values = values
            else:
                raise ValueError(
                    f"Unexpected extracted key {key!r}; expected 'feature_{{index}}' or 'target'."
                )

    dtype = _torch_dtype(config)
    x = torch.zeros((n_rows, num_features), dtype=dtype, device=device)
    feature_types = list(layout.feature_types)
    card_by_feature: dict[int, int] = layout.card_by_feature

    for i in range(num_features):
        v = feature_values[i]
        if v is None:
            if feature_types[i] == "cat":
                card = int(card_by_feature[i])
                v = torch.randint(0, card, (n_rows,), generator=generator, device=device)
            else:
                v = sample_noise_from_spec(
                    (n_rows,),
                    generator=generator,
                    device=device,
                    noise_spec=noise_spec,
                )
        x[:, i] = v.to(dtype)

    if target_values is None:
        if task == "classification":
            y = torch.randint(0, n_classes, (n_rows,), generator=generator, device=device)
        else:
            y = sample_noise_from_spec(
                (n_rows,),
                generator=generator,
                device=device,
                noise_spec=noise_spec,
            ).to(dtype)
    else:
        if task == "classification":
            y = target_values.to(torch.long) % n_classes
        else:
            y = target_values.to(dtype)

    accepted, filter_details = _apply_filter(config, x, y, seed=seed)
    return x, y, {"accepted": accepted, "filter": filter_details}


def _generate_torch(
    config: GeneratorConfig,
    layout: LayoutPlan,
    seed: int,
    device: str,
    *,
    n_train: int,
    n_test: int,
    shift_params: ShiftRuntimeParams | None = None,
    noise_runtime_selection: NoiseRuntimeSelection | None = None,
    requested_device: str,
    resolved_device: str,
    device_fallback_reason: str | None = None,
    preserve_feature_schema: bool = False,
) -> DatasetBundle:
    """Generate one dataset in Torch while preserving postprocess/filter contracts."""

    shift_params = shift_params or resolve_shift_runtime_params(config)
    attempts = max(1, int(config.filter.max_attempts))
    last_reason = "unknown"
    dtype = _torch_dtype(config)
    n_rows = n_train + n_test
    noise_runtime_selection = noise_runtime_selection or _resolve_noise_runtime_selection(
        config,
        run_seed=seed,
    )
    noise_spec = _noise_sampling_spec(noise_runtime_selection)

    for attempt in range(attempts):
        try:
            x, y, aux_meta = _generate_graph_dataset_torch(
                config,
                layout,
                _attempt_seed(seed, attempt),
                device,
                n_rows=n_rows,
                mechanism_logit_tilt=float(shift_params.mechanism_logit_tilt),
                function_family_mix=config.mechanism.function_family_mix,
                noise_sigma_multiplier=float(shift_params.variance_sigma_multiplier),
                noise_spec=noise_spec,
            )
        except Exception as exc:
            last_reason = f"generation_exception:{exc.__class__.__name__}"
            continue
        if not bool(aux_meta.get("accepted", True)):
            last_reason = "filtered_out"
            continue

        # Keep split/postprocess control-plane randomness on CPU to avoid tiny-op accelerator overhead.
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
                    last_reason = "invalid_class_split"
                    continue
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

        if config.dataset.task == "classification" and not _classification_split_valid(
            y_train, y_test
        ):
            last_reason = "invalid_class_split"
            continue

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
            "filter": aux_meta.get("filter", {}),
            "shift": shift_metadata,
            "noise_distribution": _build_noise_distribution_metadata(noise_runtime_selection),
            "config": asdict(config),
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
        )

    raise ValueError(
        f"Failed to generate a valid dataset after {attempts} attempts. Last reason: {last_reason}."
    )


def _apply_filter(
    config: GeneratorConfig,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    seed: int,
) -> tuple[bool, dict[str, Any]]:
    """Run filtering via CPU ExtraTrees."""

    details: dict[str, Any] = {"enabled": config.filter.enabled}
    if not config.filter.enabled:
        return True, details

    accepted, filter_details = apply_extra_trees_filter(
        x,
        y,
        task=config.dataset.task,
        seed=seed,
        n_estimators=config.filter.n_estimators,
        max_depth=config.filter.max_depth,
        min_samples_leaf=config.filter.min_samples_leaf,
        max_leaf_nodes=config.filter.max_leaf_nodes,
        max_features=config.filter.max_features,
        n_bootstrap=config.filter.n_bootstrap,
        threshold=config.filter.threshold,
        n_jobs=config.filter.n_jobs,
    )
    details.update(filter_details)
    details["accepted"] = accepted
    return accepted, details


def _generate_one_seeded(
    config: GeneratorConfig,
    *,
    seed: int,
    requested_device: str,
    resolved_device: str,
) -> DatasetBundle:
    """Generate one dataset for a fully resolved seed/device context."""

    manager = SeedManager(seed)
    layout_gen = manager.torch_rng("layout")
    layout = _sample_layout(config, layout_gen, "cpu")
    n_train, n_test = _resolve_split_sizes(config)
    return _generate_one_with_resolved_layout(
        config,
        seed=seed,
        requested_device=requested_device,
        resolved_device=resolved_device,
        n_train=n_train,
        n_test=n_test,
        layout=layout,
    )


def _generate_one_with_resolved_layout(
    config: GeneratorConfig,
    *,
    seed: int,
    requested_device: str,
    resolved_device: str,
    n_train: int,
    n_test: int,
    layout: LayoutPlan,
    preserve_feature_schema: bool = False,
) -> DatasetBundle:
    """Generate one dataset from an already-resolved split/layout context."""

    _validate_class_split_for_layout(config, layout=layout, n_train=n_train, n_test=n_test)
    manager = SeedManager(seed)
    data_seed = manager.child("data")
    shift_params = resolve_shift_runtime_params(config)
    noise_runtime_selection = _resolve_noise_runtime_selection(config, run_seed=data_seed)

    if requested_device == "auto" and resolved_device == "mps":
        try:
            return _generate_torch(
                config,
                layout,
                data_seed,
                resolved_device,
                n_train=n_train,
                n_test=n_test,
                shift_params=shift_params,
                noise_runtime_selection=noise_runtime_selection,
                requested_device=requested_device,
                resolved_device=resolved_device,
                device_fallback_reason=None,
                preserve_feature_schema=preserve_feature_schema,
            )
        except Exception as exc:
            # Keep auto mode robust on partially supported MPS runtimes by retrying on CPU.
            return _generate_torch(
                config,
                layout,
                data_seed,
                "cpu",
                n_train=n_train,
                n_test=n_test,
                shift_params=shift_params,
                noise_runtime_selection=noise_runtime_selection,
                requested_device=requested_device,
                resolved_device="cpu",
                device_fallback_reason=f"auto_mps_runtime_error:{exc.__class__.__name__}",
                preserve_feature_schema=preserve_feature_schema,
            )

    return _generate_torch(
        config,
        layout,
        data_seed,
        resolved_device,
        n_train=n_train,
        n_test=n_test,
        shift_params=shift_params,
        noise_runtime_selection=noise_runtime_selection,
        requested_device=requested_device,
        resolved_device=resolved_device,
        device_fallback_reason=None,
        preserve_feature_schema=preserve_feature_schema,
    )
