"""Fixed-layout plan types, serialization, and batched generation helpers."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import hashlib
import json
from typing import Any, cast

import torch

from dagzoo.config import GeneratorConfig, dataset_rows_is_variable
from dagzoo.core.fixed_layout_batched import (
    _FIXED_LAYOUT_EXECUTION_CONTRACT,
    build_fixed_layout_execution_plans,
    fixed_layout_plan_signature,
    generate_fixed_layout_graph_batch,
    normalize_fixed_layout_node_plans,
)
from dagzoo.core.generation_context import (
    _attempt_seed,
    _resolve_device,
    _resolve_run_seed,
    _resolve_split_sizes,
    _validate_class_split_for_layout,
)
from dagzoo.core.generation_engine import _finalize_generated_tensors, _torch_dtype
from dagzoo.core.layout import _sample_layout
from dagzoo.core.layout_types import FeatureType, LayoutPlan
from dagzoo.core.noise_runtime import _noise_sampling_spec, _resolve_noise_runtime_selection
from dagzoo.core.shift import resolve_shift_runtime_params
from dagzoo.rng import SeedManager
from dagzoo.types import DatasetBundle

_FIXED_LAYOUT_PLAN_SCHEMA_NAME = "dagzoo_fixed_layout_plan"
_FIXED_LAYOUT_PLAN_SCHEMA_VERSION = 3
_FIXED_LAYOUT_TARGET_CELLS = 4_000_000


@dataclass(slots=True)
class FixedLayoutPlan:
    """Pre-sampled layout bundle for fixed-layout batch generation."""

    layout: LayoutPlan
    requested_device: str
    resolved_device: str
    plan_seed: int
    n_train: int
    n_test: int
    layout_signature: str
    compatibility_snapshot: dict[str, Any]
    node_plans: list[dict[str, Any]] | None = None
    plan_signature: str | None = None
    execution_contract: str = _FIXED_LAYOUT_EXECUTION_CONTRACT
    plan_schema_version: int = _FIXED_LAYOUT_PLAN_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Serialize one fixed-layout plan into a JSON/YAML-safe mapping."""

        return {
            "schema_name": _FIXED_LAYOUT_PLAN_SCHEMA_NAME,
            "schema_version": int(self.plan_schema_version),
            "layout": _layout_to_dict(self.layout),
            "requested_device": str(self.requested_device),
            "resolved_device": str(self.resolved_device),
            "plan_seed": int(self.plan_seed),
            "n_train": int(self.n_train),
            "n_test": int(self.n_test),
            "layout_signature": str(self.layout_signature),
            "compatibility_snapshot": dict(self.compatibility_snapshot),
            "node_plans": list(self.node_plans or []),
            "plan_signature": None if self.plan_signature is None else str(self.plan_signature),
            "execution_contract": str(self.execution_contract),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FixedLayoutPlan:
        """Deserialize one fixed-layout plan from a mapping payload."""

        if not isinstance(data, dict):
            raise ValueError("Fixed-layout plan payload must be a mapping.")
        schema_name = str(data.get("schema_name", ""))
        if schema_name and schema_name != _FIXED_LAYOUT_PLAN_SCHEMA_NAME:
            raise ValueError(
                "Unsupported fixed-layout plan schema_name "
                f"{schema_name!r}; expected {_FIXED_LAYOUT_PLAN_SCHEMA_NAME!r}."
            )
        schema_version = int(data.get("schema_version", _FIXED_LAYOUT_PLAN_SCHEMA_VERSION))
        layout_payload = data.get("layout")
        if not isinstance(layout_payload, dict):
            raise ValueError("Fixed-layout plan payload must include a layout mapping.")
        snapshot = data.get("compatibility_snapshot")
        if not isinstance(snapshot, dict):
            raise ValueError(
                "Fixed-layout plan integrity mismatch: compatibility_snapshot must be a mapping."
            )
        node_plans_raw = data.get("node_plans", [])
        if node_plans_raw is None:
            node_plans = None
        elif isinstance(node_plans_raw, list):
            node_plans = [dict(plan) for plan in node_plans_raw]
        else:
            raise ValueError("Fixed-layout plan payload must include node_plans as a list.")
        return cls(
            layout=_layout_from_dict(layout_payload),
            requested_device=str(data["requested_device"]),
            resolved_device=str(data["resolved_device"]),
            plan_seed=int(data["plan_seed"]),
            n_train=int(data["n_train"]),
            n_test=int(data["n_test"]),
            layout_signature=str(data["layout_signature"]),
            compatibility_snapshot=dict(snapshot),
            node_plans=node_plans,
            plan_signature=(
                None if data.get("plan_signature") is None else str(data["plan_signature"])
            ),
            execution_contract=str(
                data.get("execution_contract", _FIXED_LAYOUT_EXECUTION_CONTRACT)
            ),
            plan_schema_version=int(schema_version),
        )


_FIXED_LAYOUT_COMPAT_KEYS: tuple[str, ...] = (
    "dataset.task",
    "dataset.n_train",
    "dataset.n_test",
    "dataset.n_features_min",
    "dataset.n_features_max",
    "dataset.categorical_ratio_min",
    "dataset.categorical_ratio_max",
    "dataset.max_categorical_cardinality",
    "dataset.n_classes_min",
    "dataset.n_classes_max",
    "graph.n_nodes_min",
    "graph.n_nodes_max",
    "shift.edge_logit_bias_shift",
    "runtime.resolved_device",
)


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


def _layout_from_dict(payload: dict[str, Any]) -> LayoutPlan:
    adjacency = torch.as_tensor(payload["adjacency"], dtype=torch.bool, device="cpu")
    feature_types = cast("list[FeatureType]", [str(value) for value in payload["feature_types"]])
    return LayoutPlan(
        n_features=int(payload["n_features"]),
        n_cat=int(payload["n_cat"]),
        cat_idx=[int(value) for value in payload["cat_idx"]],
        cardinalities=[int(value) for value in payload["cardinalities"]],
        card_by_feature={
            int(key): int(value) for key, value in dict(payload["card_by_feature"]).items()
        },
        n_classes=int(payload["n_classes"]),
        feature_types=feature_types,
        graph_nodes=int(payload["graph_nodes"]),
        graph_edges=int(payload["graph_edges"]),
        graph_depth_nodes=int(payload["graph_depth_nodes"]),
        graph_edge_density=float(payload["graph_edge_density"]),
        adjacency=adjacency,
        feature_node_assignment=[int(value) for value in payload["feature_node_assignment"]],
        target_node_assignment=int(payload["target_node_assignment"]),
    )


def _layout_signature(layout: LayoutPlan) -> str:
    encoded = json.dumps(
        _layout_to_dict(layout),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.blake2s(encoded, digest_size=16).hexdigest()


def _build_fixed_layout_compatibility_snapshot(
    config: GeneratorConfig,
    *,
    resolved_device: str,
    n_train: int,
    n_test: int,
) -> dict[str, Any]:
    shift_params = resolve_shift_runtime_params(config)
    return {
        "dataset.task": str(config.dataset.task),
        "dataset.n_train": int(n_train),
        "dataset.n_test": int(n_test),
        "dataset.n_features_min": int(config.dataset.n_features_min),
        "dataset.n_features_max": int(config.dataset.n_features_max),
        "dataset.categorical_ratio_min": float(config.dataset.categorical_ratio_min),
        "dataset.categorical_ratio_max": float(config.dataset.categorical_ratio_max),
        "dataset.max_categorical_cardinality": int(config.dataset.max_categorical_cardinality),
        "dataset.n_classes_min": int(config.dataset.n_classes_min),
        "dataset.n_classes_max": int(config.dataset.n_classes_max),
        "graph.n_nodes_min": int(config.graph.n_nodes_min),
        "graph.n_nodes_max": int(config.graph.n_nodes_max),
        "shift.edge_logit_bias_shift": float(shift_params.edge_logit_bias_shift),
        "runtime.resolved_device": str(resolved_device),
    }


def _validate_fixed_layout_rows_mode(config: GeneratorConfig) -> None:
    if dataset_rows_is_variable(config.dataset.rows):
        raise ValueError(
            "Fixed-layout generation requires a fixed split size; variable dataset.rows "
            "modes (range/choices) are not supported."
        )


def _resolve_fixed_layout_batch_size(
    plan: FixedLayoutPlan,
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


def sample_fixed_layout(
    config: GeneratorConfig,
    *,
    seed: int | None = None,
    device: str | None = None,
) -> FixedLayoutPlan:
    """Sample one reusable layout plan for fixed-layout batch generation."""

    run_seed = _resolve_run_seed(config, seed)
    _validate_fixed_layout_rows_mode(config)
    requested_device = (device or config.runtime.device or "auto").lower()
    resolved_device = _resolve_device(config, device)
    manager = SeedManager(run_seed)
    layout_gen = manager.torch_rng("layout")
    layout = _sample_layout(config, layout_gen, "cpu")
    n_train, n_test = _resolve_split_sizes(config, dataset_seed=run_seed)
    _validate_class_split_for_layout(config, layout=layout, n_train=n_train, n_test=n_test)
    node_plans = build_fixed_layout_execution_plans(config, layout, plan_seed=run_seed)
    return FixedLayoutPlan(
        layout=layout,
        requested_device=requested_device,
        resolved_device=resolved_device,
        plan_seed=int(run_seed),
        n_train=int(n_train),
        n_test=int(n_test),
        layout_signature=_layout_signature(layout),
        compatibility_snapshot=_build_fixed_layout_compatibility_snapshot(
            config,
            resolved_device=resolved_device,
            n_train=n_train,
            n_test=n_test,
        ),
        node_plans=node_plans,
        plan_signature=fixed_layout_plan_signature(node_plans),
        execution_contract=_FIXED_LAYOUT_EXECUTION_CONTRACT,
    )


def _annotate_fixed_layout_metadata(bundle: DatasetBundle, *, plan: FixedLayoutPlan) -> None:
    bundle.metadata["layout_mode"] = "fixed"
    bundle.metadata["layout_plan_seed"] = int(plan.plan_seed)
    bundle.metadata["layout_signature"] = str(plan.layout_signature)
    bundle.metadata["layout_plan_schema_version"] = int(plan.plan_schema_version)
    bundle.metadata["layout_execution_contract"] = str(plan.execution_contract)
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


def _validate_fixed_layout_plan_compatibility(
    config: GeneratorConfig,
    *,
    plan: FixedLayoutPlan,
) -> str:
    _validate_fixed_layout_rows_mode(config)

    snapshot = plan.compatibility_snapshot
    if not isinstance(snapshot, dict):
        raise ValueError(
            "Fixed-layout plan integrity mismatch: compatibility_snapshot must be a mapping."
        )

    computed_layout_signature = _layout_signature(plan.layout)
    if str(plan.layout_signature) != computed_layout_signature:
        raise ValueError(
            "Fixed-layout plan integrity mismatch: layout_signature does not match plan.layout."
        )
    if plan.node_plans is None or not isinstance(plan.node_plans, list) or not plan.node_plans:
        raise ValueError(
            "Fixed-layout plan integrity mismatch: node_plans must be a non-empty list."
        )
    normalized_node_plans = normalize_fixed_layout_node_plans(plan.node_plans)
    computed_plan_signature = fixed_layout_plan_signature(normalized_node_plans)
    if plan.plan_signature is None or str(plan.plan_signature) != computed_plan_signature:
        raise ValueError(
            "Fixed-layout plan integrity mismatch: plan_signature does not match node_plans."
        )

    missing_keys = [key for key in _FIXED_LAYOUT_COMPAT_KEYS if key not in snapshot]
    if missing_keys:
        raise ValueError(
            "Fixed-layout plan integrity mismatch: compatibility snapshot is missing keys: "
            f"{', '.join(missing_keys)}."
        )

    if int(plan.n_train) != int(snapshot["dataset.n_train"]):
        raise ValueError(
            "Fixed-layout plan integrity mismatch: plan.n_train does not match "
            "compatibility snapshot."
        )
    if int(plan.n_test) != int(snapshot["dataset.n_test"]):
        raise ValueError(
            "Fixed-layout plan integrity mismatch: plan.n_test does not match "
            "compatibility snapshot."
        )
    if str(plan.resolved_device) != str(snapshot["runtime.resolved_device"]):
        raise ValueError(
            "Fixed-layout plan integrity mismatch: plan.resolved_device does not match "
            "compatibility snapshot."
        )

    try:
        resolved_device = _resolve_device(config, plan.requested_device)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(
            "Fixed-layout plan/config mismatch: unable to resolve the plan-requested "
            f"device '{plan.requested_device}' for the current environment."
        ) from exc
    if str(plan.resolved_device) != str(resolved_device):
        raise ValueError(
            "Fixed-layout plan/config mismatch: plan.resolved_device does not match "
            f"the currently resolved backend ({plan.resolved_device!r} != {resolved_device!r})."
        )

    current_n_train, current_n_test = _resolve_split_sizes(config, dataset_seed=int(plan.plan_seed))
    current_snapshot = _build_fixed_layout_compatibility_snapshot(
        config,
        resolved_device=resolved_device,
        n_train=current_n_train,
        n_test=current_n_test,
    )
    mismatches: list[str] = []
    for key in _FIXED_LAYOUT_COMPAT_KEYS:
        plan_value = snapshot[key]
        config_value = current_snapshot[key]
        if plan_value != config_value:
            mismatches.append(f"{key} (plan={plan_value!r}, config={config_value!r})")
    if mismatches:
        raise ValueError(
            "Fixed-layout plan/config mismatch for compatibility fields: " + "; ".join(mismatches)
        )
    return str(resolved_device)


def _generate_fixed_layout_bundle_with_retries(
    config: GeneratorConfig,
    *,
    plan: FixedLayoutPlan,
    dataset_seed: int,
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
        x_batch, y_batch, aux_meta_batch = generate_fixed_layout_graph_batch(
            config,
            plan.layout,
            node_plans=plan.node_plans,
            dataset_seeds=[_attempt_seed(data_seed, attempt)],
            device=resolved_device,
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
                device=resolved_device,
                n_train=int(plan.n_train),
                n_test=int(plan.n_test),
                requested_device=plan.requested_device,
                resolved_device=resolved_device,
                device_fallback_reason=None,
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


def generate_batch_fixed_layout_iter(
    config: GeneratorConfig,
    *,
    plan: FixedLayoutPlan,
    num_datasets: int,
    seed: int | None = None,
    batch_size: int | None = None,
) -> Iterator[DatasetBundle]:
    """Yield datasets that share one pre-sampled fixed layout and split shape."""

    if num_datasets < 0:
        raise ValueError(f"num_datasets must be >= 0, got {num_datasets}")
    if num_datasets == 0:
        return

    validated_resolved_device = _validate_fixed_layout_plan_compatibility(config, plan=plan)
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
        x_batch, y_batch, aux_meta_batch = generate_fixed_layout_graph_batch(
            config,
            plan.layout,
            node_plans=plan.node_plans or [],
            dataset_seeds=data_seeds,
            device=validated_resolved_device,
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
                    device=validated_resolved_device,
                    n_train=int(plan.n_train),
                    n_test=int(plan.n_test),
                    requested_device=plan.requested_device,
                    resolved_device=validated_resolved_device,
                    device_fallback_reason=None,
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


def generate_batch_fixed_layout(
    config: GeneratorConfig,
    *,
    plan: FixedLayoutPlan,
    num_datasets: int,
    seed: int | None = None,
    batch_size: int | None = None,
) -> list[DatasetBundle]:
    """Generate a materialized fixed-layout batch using a reusable plan."""

    return list(
        generate_batch_fixed_layout_iter(
            config,
            plan=plan,
            num_datasets=num_datasets,
            seed=seed,
            batch_size=batch_size,
        )
    )
