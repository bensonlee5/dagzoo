"""Canonical fixed-layout run preparation and internal batch helpers."""

from __future__ import annotations

import copy
from collections.abc import Iterator
from dataclasses import dataclass
import hashlib
import json
from typing import Any

import torch

from dagzoo.config import (
    DatasetRowsSpec,
    GeneratorConfig,
    dataset_rows_is_variable,
    resolve_dataset_total_rows,
)
from dagzoo.core.constants import FIXED_LAYOUT_PLAN_SEED_OFFSET, ROWS_REALIZATION_SEED_OFFSET
from dagzoo.core.fixed_layout_batched import (
    _FIXED_LAYOUT_EXECUTION_CONTRACT,
    build_fixed_layout_execution_plans,
    fixed_layout_plan_signature,
    generate_fixed_layout_graph_batch,
)
from dagzoo.core.generation_context import (
    _attempt_seed,
    _resolve_device,
    _resolve_run_seed,
    _resolve_split_sizes,
    _split_permutation_seed,
    _torch_dtype,
    _validate_class_split_for_layout,
)
from dagzoo.core.generation_runtime import _finalize_generated_tensors
from dagzoo.core.layout import _sample_layout
from dagzoo.core.layout_types import LayoutPlan
from dagzoo.core.noise_runtime import _noise_sampling_spec, _resolve_noise_runtime_selection
from dagzoo.core.shift import resolve_shift_runtime_params
from dagzoo.core.validation import _classification_split_valid, _stratified_split_indices
from dagzoo.rng import SeedManager, offset_seed32
from dagzoo.types import DatasetBundle

_FIXED_LAYOUT_METADATA_SCHEMA_VERSION = 3
_FIXED_LAYOUT_TARGET_CELLS = 4_000_000


@dataclass(slots=True)
class _FixedLayoutPlan:
    """Internal pre-sampled layout bundle for canonical fixed-layout generation."""

    layout: LayoutPlan
    requested_device: str
    resolved_device: str
    plan_seed: int
    n_train: int
    n_test: int
    layout_signature: str
    node_plans: list[dict[str, Any]] | None = None
    plan_signature: str | None = None


@dataclass(slots=True)
class CanonicalFixedLayoutRun:
    """Prepared fixed-layout run context for canonical public generation."""

    config: GeneratorConfig
    plan: _FixedLayoutPlan
    run_seed: int
    requested_device: str
    resolved_device: str
    batch_size: int


def _layout_to_dict(layout: LayoutPlan) -> dict[str, Any]:
    adjacency = layout.adjacency
    if isinstance(adjacency, torch.Tensor):
        adjacency_payload = adjacency.to(device="cpu", dtype=torch.int64).tolist()
    else:
        adjacency_payload = torch.as_tensor(adjacency, dtype=torch.int64, device="cpu").tolist()
    return {
        "n_features": int(layout.n_features),
        "n_cat": int(layout.n_cat),
        "cat_idx": [int(value) for value in layout.cat_idx],
        "cardinalities": [int(value) for value in layout.cardinalities],
        "card_by_feature": {
            str(int(key)): int(value) for key, value in layout.card_by_feature.items()
        },
        "n_classes": int(layout.n_classes),
        "feature_types": [str(value) for value in layout.feature_types],
        "graph_nodes": int(layout.graph_nodes),
        "graph_edges": int(layout.graph_edges),
        "graph_depth_nodes": int(layout.graph_depth_nodes),
        "graph_edge_density": float(layout.graph_edge_density),
        "adjacency": adjacency_payload,
        "feature_node_assignment": [int(value) for value in layout.feature_node_assignment],
        "target_node_assignment": int(layout.target_node_assignment),
    }


def _layout_signature(layout: LayoutPlan) -> str:
    encoded = json.dumps(
        _layout_to_dict(layout),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.blake2s(encoded, digest_size=16).hexdigest()


def _validate_fixed_layout_rows_mode(config: GeneratorConfig) -> None:
    if dataset_rows_is_variable(config.dataset.rows):
        raise ValueError(
            "Fixed-layout generation requires a fixed split size; variable dataset.rows "
            "modes (range/choices) are not supported."
        )


def _resolve_fixed_layout_batch_size(
    plan: _FixedLayoutPlan,
    *,
    num_datasets: int,
    batch_size: int | None,
) -> int:
    if batch_size is not None:
        return max(1, min(int(batch_size), int(num_datasets)))
    per_dataset_cells = max(
        1, int(plan.n_train + plan.n_test) * max(1, int(plan.layout.n_features))
    )
    auto_batch = max(1, _FIXED_LAYOUT_TARGET_CELLS // per_dataset_cells)
    return max(1, min(int(num_datasets), int(auto_batch)))


def realize_generation_config_for_run(
    config: GeneratorConfig,
    *,
    seed: int | None = None,
    device: str | None = None,
) -> tuple[GeneratorConfig, int, str, str]:
    """Resolve one canonical single-run config with rows fixed for the full run."""

    run_seed = _resolve_run_seed(config, seed)
    requested_device = (device or config.runtime.device or "auto").lower()
    resolved_device = _resolve_device(config, device)

    realized = copy.deepcopy(config)
    rows_seed = offset_seed32(run_seed, ROWS_REALIZATION_SEED_OFFSET)
    total_rows = resolve_dataset_total_rows(realized.dataset.rows, dataset_seed=rows_seed)
    if total_rows is not None:
        n_test = int(realized.dataset.n_test)
        n_train = int(total_rows) - n_test
        if n_train <= 0:
            raise ValueError(
                "Resolved rows split is invalid: total rows must be > dataset.n_test "
                f"(total_rows={int(total_rows)}, n_test={n_test})."
            )
        realized.dataset.n_train = int(n_train)
        realized.dataset.rows = DatasetRowsSpec(mode="fixed", value=int(total_rows))

    return realized, int(run_seed), str(requested_device), str(resolved_device)


def prepare_canonical_fixed_layout_run(
    config: GeneratorConfig,
    *,
    num_datasets: int,
    seed: int | None = None,
    device: str | None = None,
    batch_size: int | None = None,
) -> CanonicalFixedLayoutRun:
    """Prepare one internal fixed-layout run context for public generation APIs.

    Classification runs intentionally validate the requested run up front so a
    canonical batch does not emit partial output and then fail on a later
    dataset seed.
    """

    if num_datasets < 0:
        raise ValueError(f"num_datasets must be >= 0, got {num_datasets}")

    realized_config, run_seed, requested_device, resolved_device = (
        realize_generation_config_for_run(
            config,
            seed=seed,
            device=device,
        )
    )
    base_plan_seed = offset_seed32(run_seed, FIXED_LAYOUT_PLAN_SEED_OFFSET)
    plan = _sample_fixed_layout(
        realized_config,
        seed=base_plan_seed,
        device=requested_device,
    )
    effective_batch_size = _resolve_fixed_layout_batch_size(
        plan,
        num_datasets=max(1, int(num_datasets)),
        batch_size=batch_size,
    )
    if str(realized_config.dataset.task) == "classification":
        # Canonical classification batches guarantee completion for the
        # requested run. That requires validating the shared-plan replay
        # schedule before any public bundle is emitted.
        attempts = max(1, int(realized_config.filter.max_attempts))
        for attempt in range(attempts):
            effective_batch_size = _resolve_fixed_layout_batch_size(
                plan,
                num_datasets=max(1, int(num_datasets)),
                batch_size=batch_size,
            )
            if _fixed_layout_plan_supports_classification_run(
                realized_config,
                plan=plan,
                requested_device=requested_device,
                resolved_device=resolved_device,
                run_seed=run_seed,
                num_datasets=max(1, int(num_datasets)),
                batch_size=effective_batch_size,
            ):
                break
            if attempt == attempts - 1:
                raise ValueError(
                    "Failed to prepare a replayable canonical fixed-layout classification run "
                    f"after {attempts} attempts."
                )
            plan = _sample_fixed_layout(
                realized_config,
                seed=_attempt_seed(base_plan_seed, attempt + 1),
                device=requested_device,
            )
    return CanonicalFixedLayoutRun(
        config=realized_config,
        plan=plan,
        run_seed=int(run_seed),
        requested_device=str(requested_device),
        resolved_device=str(resolved_device),
        batch_size=int(effective_batch_size),
    )


def _sample_fixed_layout_once(
    config: GeneratorConfig,
    *,
    seed: int,
    requested_device: str,
    resolved_device: str,
) -> _FixedLayoutPlan:
    """Sample one fixed-layout plan candidate without replay validation retries."""

    _validate_fixed_layout_rows_mode(config)
    manager = SeedManager(seed)
    layout_gen = manager.torch_rng("layout")
    layout = _sample_layout(config, layout_gen, "cpu")
    n_train, n_test = _resolve_split_sizes(config, dataset_seed=seed)
    _validate_class_split_for_layout(config, layout=layout, n_train=n_train, n_test=n_test)
    shift_params = resolve_shift_runtime_params(config)
    node_plans = build_fixed_layout_execution_plans(
        config,
        layout,
        plan_seed=seed,
        mechanism_logit_tilt=float(shift_params.mechanism_logit_tilt),
    )
    return _FixedLayoutPlan(
        layout=layout,
        requested_device=requested_device,
        resolved_device=resolved_device,
        plan_seed=int(seed),
        n_train=int(n_train),
        n_test=int(n_test),
        layout_signature=_layout_signature(layout),
        node_plans=node_plans,
        plan_signature=fixed_layout_plan_signature(node_plans),
    )


def _raw_classification_labels_support_split(
    y: torch.Tensor,
    *,
    dataset_seed: int,
    attempt: int,
    n_train: int,
) -> bool:
    """Return whether one raw classification label vector can satisfy split constraints."""

    labels = y.to(device="cpu", dtype=torch.int64)
    split_generator = torch.Generator(device="cpu")
    split_generator.manual_seed(_split_permutation_seed(dataset_seed, attempt))
    try:
        train_idx_cpu, test_idx_cpu = _stratified_split_indices(
            labels,
            int(n_train),
            split_generator,
            "cpu",
        )
    except ValueError as exc:
        if str(exc).startswith("infeasible_stratified_split"):
            return False
        raise
    return _classification_split_valid(labels[train_idx_cpu], labels[test_idx_cpu])


def _fixed_layout_dataset_supports_classification_replay(
    config: GeneratorConfig,
    *,
    plan: _FixedLayoutPlan,
    dataset_seed: int,
    requested_device: str,
    resolved_device: str,
) -> bool:
    """Return whether one dataset seed can replay under one fixed-layout plan."""

    if plan.node_plans is None:
        raise ValueError("Fixed-layout plan must include node_plans.")

    data_seed = SeedManager(dataset_seed).child("data")
    shift_params = resolve_shift_runtime_params(config)
    noise_runtime_selection = _resolve_noise_runtime_selection(config, run_seed=data_seed)
    noise_spec = _noise_sampling_spec(noise_runtime_selection)
    attempts = max(1, int(config.filter.max_attempts))

    for attempt in range(attempts):
        _, y_batch, _, _effective_resolved_device, _device_fallback_reason = (
            _generate_fixed_layout_graph_batch_with_fallback(
                config,
                plan.layout,
                node_plans=plan.node_plans,
                dataset_seeds=[_attempt_seed(data_seed, attempt)],
                requested_device=requested_device,
                resolved_device=resolved_device,
                noise_sigma_multiplier=float(shift_params.variance_sigma_multiplier),
                noise_spec=noise_spec,
            )
        )
        if _raw_classification_labels_support_split(
            y_batch[0],
            dataset_seed=dataset_seed,
            attempt=attempt,
            n_train=int(plan.n_train),
        ):
            return True
    return False


def _fixed_layout_plan_supports_classification_run(
    config: GeneratorConfig,
    *,
    plan: _FixedLayoutPlan,
    requested_device: str,
    resolved_device: str,
    run_seed: int,
    num_datasets: int = 1,
    batch_size: int = 1,
) -> bool:
    """Return whether a classification plan can replay for the full requested run."""

    if plan.node_plans is None:
        raise ValueError("Fixed-layout plan must include node_plans.")

    manager = SeedManager(run_seed)
    shift_params = resolve_shift_runtime_params(config)
    effective_batch_size = max(1, int(batch_size))
    dataset_index = 0
    while dataset_index < num_datasets:
        chunk_size = min(effective_batch_size, num_datasets - dataset_index)
        dataset_seeds = [
            manager.child("dataset", dataset_index + offset) for offset in range(chunk_size)
        ]
        data_seeds = [SeedManager(dataset_seed).child("data") for dataset_seed in dataset_seeds]
        noise_runtime_selections = [
            _resolve_noise_runtime_selection(config, run_seed=data_seed) for data_seed in data_seeds
        ]
        if any(
            selection != noise_runtime_selections[0] for selection in noise_runtime_selections[1:]
        ):
            for dataset_seed in dataset_seeds:
                if not _fixed_layout_dataset_supports_classification_replay(
                    config,
                    plan=plan,
                    dataset_seed=dataset_seed,
                    requested_device=requested_device,
                    resolved_device=resolved_device,
                ):
                    return False
            dataset_index += chunk_size
            continue

        noise_runtime_selection = noise_runtime_selections[0]
        noise_spec = _noise_sampling_spec(noise_runtime_selection)
        _, y_batch, _aux_meta_batch, _effective_resolved_device, _device_fallback_reason = (
            _generate_fixed_layout_graph_batch_with_fallback(
                config,
                plan.layout,
                node_plans=plan.node_plans,
                dataset_seeds=data_seeds,
                requested_device=requested_device,
                resolved_device=resolved_device,
                noise_sigma_multiplier=float(shift_params.variance_sigma_multiplier),
                noise_spec=noise_spec,
            )
        )
        for offset, dataset_seed in enumerate(dataset_seeds):
            if _raw_classification_labels_support_split(
                y_batch[offset],
                dataset_seed=dataset_seed,
                attempt=0,
                n_train=int(plan.n_train),
            ):
                continue
            if not _fixed_layout_dataset_supports_classification_replay(
                config,
                plan=plan,
                dataset_seed=dataset_seed,
                requested_device=requested_device,
                resolved_device=resolved_device,
            ):
                return False
        dataset_index += chunk_size
    return True


def _fixed_layout_plan_supports_classification_replay(
    config: GeneratorConfig,
    *,
    plan: _FixedLayoutPlan,
    requested_device: str,
    resolved_device: str,
    validation_seed: int,
) -> bool:
    """Return whether a classification plan can replay under the fixed-layout engine."""

    return _fixed_layout_plan_supports_classification_run(
        config,
        plan=plan,
        requested_device=requested_device,
        resolved_device=resolved_device,
        run_seed=validation_seed,
        num_datasets=1,
        batch_size=1,
    )


def _sample_fixed_layout(
    config: GeneratorConfig,
    *,
    seed: int | None = None,
    device: str | None = None,
) -> _FixedLayoutPlan:
    """Sample one in-process layout plan for canonical fixed-layout generation."""

    run_seed = _resolve_run_seed(config, seed)
    requested_device = (device or config.runtime.device or "auto").lower()
    resolved_device = _resolve_device(config, device)
    attempts = max(1, int(config.filter.max_attempts))
    last_error = "unknown"

    for attempt in range(attempts):
        plan_seed = _attempt_seed(run_seed, attempt)
        plan = _sample_fixed_layout_once(
            config,
            seed=plan_seed,
            requested_device=requested_device,
            resolved_device=resolved_device,
        )
        if str(config.dataset.task) != "classification":
            return plan
        valid = False
        for validation_attempt in range(attempts):
            validation_seed = _attempt_seed(plan_seed, validation_attempt)
            if _fixed_layout_plan_supports_classification_replay(
                config,
                plan=plan,
                requested_device=requested_device,
                resolved_device=resolved_device,
                validation_seed=validation_seed,
            ):
                valid = True
                break
        if valid:
            return plan
        last_error = "invalid_class_split"

    raise ValueError(
        "Failed to sample a replayable fixed-layout classification plan after "
        f"{attempts} attempts. Last reason: {last_error}."
    )


def _annotate_fixed_layout_metadata(bundle: DatasetBundle, *, plan: _FixedLayoutPlan) -> None:
    bundle.metadata["layout_mode"] = "fixed"
    bundle.metadata["layout_plan_seed"] = int(plan.plan_seed)
    bundle.metadata["layout_signature"] = str(plan.layout_signature)
    bundle.metadata["layout_plan_schema_version"] = int(_FIXED_LAYOUT_METADATA_SCHEMA_VERSION)
    bundle.metadata["layout_execution_contract"] = str(_FIXED_LAYOUT_EXECUTION_CONTRACT)
    if plan.plan_signature is not None:
        bundle.metadata["layout_plan_signature"] = str(plan.plan_signature)


def _extract_emitted_schema_signature(
    bundle: DatasetBundle,
) -> tuple[int, tuple[str, ...], tuple[int, ...]]:
    n_features = int(bundle.metadata.get("n_features", int(bundle.X_train.shape[1])))
    feature_types = tuple(str(t) for t in bundle.feature_types)
    if len(feature_types) != n_features:
        raise ValueError(
            "Fixed-layout bundle emitted inconsistent feature schema metadata: "
            f"n_features={n_features}, feature_types_len={len(feature_types)}."
        )

    lineage = bundle.metadata.get("lineage")
    if not isinstance(lineage, dict):
        raise ValueError("Fixed-layout bundle is missing lineage metadata.")
    assignments = lineage.get("assignments")
    if not isinstance(assignments, dict):
        raise ValueError("Fixed-layout bundle is missing lineage assignments metadata.")
    raw_feature_to_node = assignments.get("feature_to_node")
    if not isinstance(raw_feature_to_node, list):
        raise ValueError("Fixed-layout bundle is missing lineage assignments.feature_to_node.")
    feature_to_node = tuple(int(value) for value in raw_feature_to_node)
    if len(feature_to_node) != n_features:
        raise ValueError(
            "Fixed-layout bundle emitted inconsistent lineage feature mapping: "
            f"n_features={n_features}, feature_to_node_len={len(feature_to_node)}."
        )

    return n_features, feature_types, feature_to_node


def _generate_fixed_layout_graph_batch_with_fallback(
    config: GeneratorConfig,
    layout: LayoutPlan,
    *,
    node_plans: list[dict[str, Any]],
    dataset_seeds: list[int],
    requested_device: str,
    resolved_device: str,
    noise_sigma_multiplier: float,
    noise_spec: Any,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]], str, str | None]:
    if requested_device == "auto" and resolved_device == "mps":
        try:
            x_batch, y_batch, aux_meta_batch = generate_fixed_layout_graph_batch(
                config,
                layout,
                node_plans=node_plans,
                dataset_seeds=dataset_seeds,
                device=resolved_device,
                noise_sigma_multiplier=noise_sigma_multiplier,
                noise_spec=noise_spec,
            )
            return x_batch, y_batch, aux_meta_batch, resolved_device, None
        except Exception as exc:
            x_batch, y_batch, aux_meta_batch = generate_fixed_layout_graph_batch(
                config,
                layout,
                node_plans=node_plans,
                dataset_seeds=dataset_seeds,
                device="cpu",
                noise_sigma_multiplier=noise_sigma_multiplier,
                noise_spec=noise_spec,
            )
            return (
                x_batch,
                y_batch,
                aux_meta_batch,
                "cpu",
                f"auto_mps_runtime_error:{exc.__class__.__name__}",
            )

    x_batch, y_batch, aux_meta_batch = generate_fixed_layout_graph_batch(
        config,
        layout,
        node_plans=node_plans,
        dataset_seeds=dataset_seeds,
        device=resolved_device,
        noise_sigma_multiplier=noise_sigma_multiplier,
        noise_spec=noise_spec,
    )
    return x_batch, y_batch, aux_meta_batch, resolved_device, None


def _generate_fixed_layout_bundle_with_retries(
    config: GeneratorConfig,
    *,
    plan: _FixedLayoutPlan,
    dataset_seed: int,
    requested_device: str,
    resolved_device: str,
    preserve_feature_schema: bool,
) -> DatasetBundle:
    if plan.node_plans is None:
        raise ValueError("Fixed-layout plan must include node_plans.")

    data_seed = SeedManager(dataset_seed).child("data")
    shift_params = resolve_shift_runtime_params(config)
    noise_runtime_selection = _resolve_noise_runtime_selection(config, run_seed=data_seed)
    noise_spec = _noise_sampling_spec(noise_runtime_selection)
    dtype = _torch_dtype(config)
    attempts = max(1, int(config.filter.max_attempts))
    last_error: str = "unknown"

    for attempt in range(attempts):
        (
            x_batch,
            y_batch,
            aux_meta_batch,
            effective_resolved_device,
            device_fallback_reason,
        ) = _generate_fixed_layout_graph_batch_with_fallback(
            config,
            plan.layout,
            node_plans=plan.node_plans,
            dataset_seeds=[_attempt_seed(data_seed, attempt)],
            requested_device=requested_device,
            resolved_device=resolved_device,
            noise_sigma_multiplier=float(shift_params.variance_sigma_multiplier),
            noise_spec=noise_spec,
        )
        try:
            return _finalize_generated_tensors(
                config,
                plan.layout,
                seed=dataset_seed,
                attempt=attempt,
                attempts_used=attempt + 1,
                device=effective_resolved_device,
                n_train=int(plan.n_train),
                n_test=int(plan.n_test),
                requested_device=requested_device,
                resolved_device=effective_resolved_device,
                device_fallback_reason=device_fallback_reason,
                x=x_batch[0],
                y=y_batch[0],
                aux_meta=aux_meta_batch[0],
                shift_params=shift_params,
                noise_runtime_selection=noise_runtime_selection,
                dtype=dtype,
                preserve_feature_schema=preserve_feature_schema,
            )
        except ValueError as exc:
            if str(exc) == "invalid_class_split":
                last_error = "invalid_class_split"
                continue
            raise

    raise ValueError(
        "Failed to generate a valid fixed-layout dataset after "
        f"{attempts} attempts. Last reason: {last_error}."
    )


def _generate_batch_with_plan_iter(
    config: GeneratorConfig,
    *,
    plan: _FixedLayoutPlan,
    num_datasets: int,
    seed: int | None = None,
    batch_size: int | None = None,
) -> Iterator[DatasetBundle]:
    """Yield datasets for one in-process fixed-layout plan."""

    if num_datasets < 0:
        raise ValueError(f"num_datasets must be >= 0, got {num_datasets}")
    if num_datasets == 0:
        return

    requested_device = str(plan.requested_device)
    validated_resolved_device = str(plan.resolved_device)
    run_seed = _resolve_run_seed(config, seed)
    manager = SeedManager(run_seed)
    dtype = _torch_dtype(config)
    shift_params = resolve_shift_runtime_params(config)
    expected_schema: tuple[int, tuple[str, ...], tuple[int, ...]] | None = None

    effective_batch_size = _resolve_fixed_layout_batch_size(
        plan,
        num_datasets=num_datasets,
        batch_size=batch_size,
    )
    dataset_index = 0
    while dataset_index < num_datasets:
        chunk_size = min(effective_batch_size, num_datasets - dataset_index)
        dataset_seeds = [
            manager.child("dataset", dataset_index + offset) for offset in range(chunk_size)
        ]
        data_seeds = [SeedManager(dataset_seed).child("data") for dataset_seed in dataset_seeds]
        noise_runtime_selections = [
            _resolve_noise_runtime_selection(config, run_seed=data_seed) for data_seed in data_seeds
        ]
        if any(
            selection != noise_runtime_selections[0] for selection in noise_runtime_selections[1:]
        ):
            # Noise-family runtime selection currently stays scalar when datasets in the same
            # chunk sample different runtime families.
            for dataset_seed in dataset_seeds:
                bundle = _generate_fixed_layout_bundle_with_retries(
                    config,
                    plan=plan,
                    dataset_seed=dataset_seed,
                    requested_device=requested_device,
                    resolved_device=validated_resolved_device,
                    preserve_feature_schema=True,
                )
                _annotate_fixed_layout_metadata(bundle, plan=plan)
                schema = _extract_emitted_schema_signature(bundle)
                if expected_schema is None:
                    expected_schema = schema
                elif schema != expected_schema:
                    raise ValueError(
                        "Fixed-layout schema mismatch: emitted dataset does not match "
                        "the first fixed-layout bundle schema."
                    )
                yield bundle
            dataset_index += chunk_size
            continue

        noise_runtime_selection = noise_runtime_selections[0]
        noise_spec = _noise_sampling_spec(noise_runtime_selection)
        (
            x_batch,
            y_batch,
            aux_meta_batch,
            effective_resolved_device,
            device_fallback_reason,
        ) = _generate_fixed_layout_graph_batch_with_fallback(
            config,
            plan.layout,
            node_plans=plan.node_plans or [],
            dataset_seeds=data_seeds,
            requested_device=requested_device,
            resolved_device=validated_resolved_device,
            noise_sigma_multiplier=float(shift_params.variance_sigma_multiplier),
            noise_spec=noise_spec,
        )
        for offset, dataset_seed in enumerate(dataset_seeds):
            try:
                bundle = _finalize_generated_tensors(
                    config,
                    plan.layout,
                    seed=dataset_seed,
                    attempt=0,
                    attempts_used=1,
                    device=effective_resolved_device,
                    n_train=int(plan.n_train),
                    n_test=int(plan.n_test),
                    requested_device=requested_device,
                    resolved_device=effective_resolved_device,
                    device_fallback_reason=device_fallback_reason,
                    x=x_batch[offset],
                    y=y_batch[offset],
                    aux_meta=aux_meta_batch[offset],
                    shift_params=shift_params,
                    noise_runtime_selection=noise_runtime_selection,
                    dtype=dtype,
                    preserve_feature_schema=True,
                )
            except ValueError as exc:
                if str(exc) != "invalid_class_split":
                    raise
                bundle = _generate_fixed_layout_bundle_with_retries(
                    config,
                    plan=plan,
                    dataset_seed=dataset_seed,
                    requested_device=requested_device,
                    resolved_device=validated_resolved_device,
                    preserve_feature_schema=True,
                )
            _annotate_fixed_layout_metadata(bundle, plan=plan)
            schema = _extract_emitted_schema_signature(bundle)
            if expected_schema is None:
                expected_schema = schema
            elif schema != expected_schema:
                raise ValueError(
                    "Fixed-layout schema mismatch: emitted dataset does not match "
                    "the first fixed-layout bundle schema."
                )
            yield bundle
        dataset_index += chunk_size
