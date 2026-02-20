"""Seeded synthetic dataset generation with NumPy default and optional torch output."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import torch

from cauchy_generator.config import GeneratorConfig
from cauchy_generator.core.node_pipeline import (
    ConverterSpec,
    apply_node_pipeline,
    apply_node_pipeline_torch,
)
from cauchy_generator.filtering import apply_extratrees_filter
from cauchy_generator.graph import sample_cauchy_dag
from cauchy_generator.postprocess import postprocess_dataset
from cauchy_generator.rng import SeedManager
from cauchy_generator.sampling import CorrelatedSampler
from cauchy_generator.types import DatasetBundle


def _resolve_backend(
    config: GeneratorConfig, device_override: str | None
) -> tuple[str, str]:
    """Resolve runtime backend/device with NumPy default and optional torch."""

    requested = (device_override or config.runtime.device or "auto").lower()
    if torch is None:
        return "numpy", "cpu"

    # Explicit accelerator requests opt into torch when available.
    if requested == "cuda":
        return ("torch", "cuda") if torch.cuda.is_available() else ("numpy", "cpu")
    if requested == "mps":
        mps_ok = (
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        )
        return ("torch", "mps") if mps_ok else ("numpy", "cpu")

    if requested == "auto":
        if config.runtime.prefer_torch:
            if torch.cuda.is_available():
                return "torch", "cuda"
            if (
                getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
            ):
                return "torch", "mps"
            return "torch", "cpu"
        return "numpy", "cpu"

    if requested == "cpu":
        return ("torch", "cpu") if config.runtime.prefer_torch else ("numpy", "cpu")
    return "numpy", "cpu"


def _torch_dtype(config: GeneratorConfig) -> torch.dtype:
    """Map string runtime dtype configuration to a torch dtype."""

    assert torch is not None
    return torch.float64 if config.runtime.torch_dtype == "float64" else torch.float32


def _sample_node_count(config: GeneratorConfig, rng: np.random.Generator) -> int:
    """Sample graph node count using log-uniform bounds from config."""

    low = max(2.0, float(config.graph.n_nodes_min))
    high = max(low, float(config.graph.n_nodes_max))
    return max(2, int(np.exp(rng.uniform(np.log(low), np.log(high)))))


def _sample_assignments(
    n_cols: int, n_nodes: int, rng: np.random.Generator
) -> np.ndarray:
    """Assign columns to a random eligible subset of graph nodes."""

    eligible_count = int(rng.integers(1, n_nodes + 1))
    eligible_nodes = rng.choice(np.arange(n_nodes), size=eligible_count, replace=False)
    return rng.choice(eligible_nodes, size=n_cols, replace=True).astype(np.int32)


def _sample_layout(config: GeneratorConfig, rng: np.random.Generator) -> dict[str, Any]:
    """Sample dataset layout, graph, and node assignments for one dataset instance."""

    n_features = int(
        rng.integers(config.dataset.n_features_min, config.dataset.n_features_max + 1)
    )

    corr = CorrelatedSampler(rng)
    raw_ratio = corr.sample_num(
        "categorical_ratio",
        config.dataset.categorical_ratio_min,
        config.dataset.categorical_ratio_max,
        log_scale=False,
        as_int=False,
    )
    cat_ratio = float(np.clip(raw_ratio, 0.0, 1.0))
    n_cat = int(round(cat_ratio * n_features))
    n_cat = max(0, min(n_features, n_cat))
    cat_idx = (
        np.sort(rng.choice(np.arange(n_features), size=n_cat, replace=False))
        if n_cat > 0
        else np.array([], dtype=np.int32)
    )

    max_card = max(2, config.dataset.max_categorical_cardinality)
    cardinalities = [
        int(np.exp(rng.uniform(np.log(2.0), np.log(float(max_card))))) for _ in cat_idx
    ]
    cardinalities = [max(2, x) for x in cardinalities]
    card_by_feature = {
        int(idx): int(card) for idx, card in zip(cat_idx, cardinalities, strict=True)
    }

    n_classes = int(
        rng.integers(config.dataset.n_classes_min, config.dataset.n_classes_max + 1)
    )
    n_classes = max(2, n_classes)

    n_nodes = _sample_node_count(config, rng)
    adjacency = sample_cauchy_dag(n_nodes, rng)
    feature_node_assignment = _sample_assignments(n_features, n_nodes, rng)
    target_node_assignment = int(_sample_assignments(1, n_nodes, rng)[0])

    feature_types = ["num"] * n_features
    for i in cat_idx:
        feature_types[int(i)] = "cat"

    return {
        "n_features": n_features,
        "n_cat": n_cat,
        "cat_idx": cat_idx,
        "cardinalities": cardinalities,
        "card_by_feature": card_by_feature,
        "n_classes": n_classes,
        "feature_types": feature_types,
        "graph_nodes": n_nodes,
        "graph_edges": int(adjacency.sum()),
        "adjacency": adjacency,
        "feature_node_assignment": feature_node_assignment,
        "target_node_assignment": target_node_assignment,
    }


def _build_node_specs(
    node_index: int,
    layout: dict[str, Any],
    task: str,
    rng: np.random.Generator,
) -> list[ConverterSpec]:
    """Build converter specs for one node in the graph execution order."""

    specs: list[ConverterSpec] = []
    feature_assignment = np.asarray(layout["feature_node_assignment"])
    feature_types = list(layout["feature_types"])
    card_by_feature: dict[int, int] = layout["card_by_feature"]

    feature_indices = np.where(feature_assignment == node_index)[0]
    for f_idx in feature_indices:
        f = int(f_idx)
        if feature_types[f] == "cat":
            c = int(card_by_feature[f])
            if c > 2 and rng.random() >= 0.5:
                d = int(rng.integers(1, c))
            else:
                d = c
            specs.append(
                ConverterSpec(
                    key=f"feature_{f}", kind="cat", dim=max(1, d), cardinality=c
                )
            )
        else:
            specs.append(ConverterSpec(key=f"feature_{f}", kind="num", dim=1))

    if int(layout["target_node_assignment"]) == node_index:
        if task == "classification":
            n_classes = int(layout["n_classes"])
            specs.append(
                ConverterSpec(
                    key="target",
                    kind="target_cls",
                    dim=max(2, n_classes),
                    cardinality=n_classes,
                )
            )
        else:
            specs.append(ConverterSpec(key="target", kind="target_reg", dim=1))
    return specs


def _generate_graph_dataset_torch(
    config: GeneratorConfig,
    layout: dict[str, Any],
    seed: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Generate raw X/y tensors via the Torch graph pipeline."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    n_rows = config.dataset.n_train + config.dataset.n_test
    n_features = int(layout["n_features"])
    task = config.dataset.task
    n_classes = int(layout["n_classes"])
    adjacency = torch.as_tensor(layout["adjacency"], dtype=torch.bool, device=device)
    n_nodes = int(layout["graph_nodes"])

    node_outputs: dict[int, torch.Tensor] = {}
    feature_values: list[torch.Tensor | None] = [None] * n_features
    target_values: torch.Tensor | None = None

    for node_idx in range(n_nodes):
        parents = torch.where(adjacency[:, node_idx])[0].tolist()
        parent_data = [node_outputs[int(p)] for p in parents if int(p) in node_outputs]

        # Build specs using CPU RNG for layout consistency
        rng_cpu = np.random.default_rng(seed + node_idx + 1000)
        specs = _build_node_specs(node_idx, layout, task, rng_cpu)

        x_node, extracted = apply_node_pipeline_torch(
            parent_data, n_rows, specs, generator, device
        )
        node_outputs[node_idx] = x_node

        for key, values in extracted.items():
            if key.startswith("feature_"):
                col = int(key.split("_", maxsplit=1)[1])
                feature_values[col] = values
            elif key == "target":
                target_values = values

    dtype = torch.float64 if config.runtime.torch_dtype == "float64" else torch.float32
    x = torch.zeros((n_rows, n_features), dtype=dtype, device=device)
    feature_types = list(layout["feature_types"])
    card_by_feature: dict[int, int] = layout["card_by_feature"]

    for i in range(n_features):
        v = feature_values[i]
        if v is None:
            if feature_types[i] == "cat":
                card = int(card_by_feature[i])
                v = torch.randint(
                    0, card, (n_rows,), generator=generator, device=device
                )
            else:
                v = torch.randn(n_rows, generator=generator, device=device)
        x[:, i] = v.to(dtype)

    if target_values is None:
        if task == "classification":
            y = torch.randint(
                0, n_classes, (n_rows,), generator=generator, device=device
            )
        else:
            y = torch.randn(n_rows, generator=generator, device=device).to(dtype)
    else:
        if task == "classification":
            y = target_values.to(torch.long) % n_classes
        else:
            y = target_values.to(dtype)

    return (
        x,
        y,
        {
            "accepted": True,
            "filter": {"enabled": False, "reason": "skipped_in_torch_path"},
        },
    )


def _generate_torch(
    config: GeneratorConfig,
    layout: dict[str, Any],
    seed: int,
    device: str,
) -> DatasetBundle:
    """Generate one dataset natively in torch."""
    x, y, aux_meta = _generate_graph_dataset_torch(config, layout, seed, device)

    # Random permutation
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 10_007)
    order = torch.randperm(x.shape[0], generator=generator, device=device)
    x = x[order]
    y = y[order]

    n_train = config.dataset.n_train
    x_train, x_test = x[:n_train], x[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    metadata = {
        "backend": "torch",
        "device": device,
        "compute_backend": "torch_appendix_full",
        "n_features": int(x_train.shape[1]),
        "graph_nodes": int(layout["graph_nodes"]),
        "graph_edges": int(layout["graph_edges"]),
        "seed": seed,
        "config": asdict(config),
    }
    return DatasetBundle(
        X_train=x_train,
        y_train=y_train,
        X_test=x_test,
        y_test=y_test,
        feature_types=list(layout["feature_types"]),
        metadata=metadata,
    )


def _classification_split_valid(y_train: np.ndarray, y_test: np.ndarray) -> bool:
    """Validate classification split constraints from Appendix E.13."""

    train_classes = set(np.unique(y_train).tolist())
    test_classes = set(np.unique(y_test).tolist())
    return len(train_classes) >= 2 and train_classes == test_classes


def _apply_filter_numpy(
    config: GeneratorConfig,
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
) -> tuple[bool, dict[str, Any]]:
    """Run optional ExtraTrees filtering and return acceptance metadata."""

    details: dict[str, Any] = {"enabled": config.filter.enabled}
    if not config.filter.enabled:
        return True, details

    accepted, filter_details = apply_extratrees_filter(
        x,
        y,
        task=config.dataset.task,
        seed=seed,
        n_estimators=config.filter.n_estimators,
        max_depth=config.filter.max_depth,
        n_bootstrap=config.filter.n_bootstrap,
        threshold=config.filter.threshold,
    )
    details.update(filter_details)
    details["accepted"] = accepted
    return accepted, details


def _generate_graph_dataset_numpy(
    config: GeneratorConfig,
    layout: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Generate raw X/y arrays via the NumPy graph pipeline."""

    rng = np.random.default_rng(seed)
    n_rows = config.dataset.n_train + config.dataset.n_test
    n_features = int(layout["n_features"])
    task = config.dataset.task
    n_classes = int(layout["n_classes"])
    adjacency = np.asarray(layout["adjacency"], dtype=bool)
    n_nodes = int(layout["graph_nodes"])

    node_outputs: dict[int, np.ndarray] = {}
    feature_values: list[np.ndarray | None] = [None] * n_features
    target_values: np.ndarray | None = None

    for node_idx in range(n_nodes):
        parents = np.where(adjacency[:, node_idx])[0]
        parent_data = [node_outputs[int(p)] for p in parents if int(p) in node_outputs]
        specs = _build_node_specs(node_idx, layout, task, rng)
        x_node, extracted = apply_node_pipeline(parent_data, n_rows, specs, rng)
        node_outputs[node_idx] = x_node

        for key, values in extracted.items():
            if key.startswith("feature_"):
                col = int(key.split("_", maxsplit=1)[1])
                feature_values[col] = values
            elif key == "target":
                target_values = values

    x = np.zeros((n_rows, n_features), dtype=np.float32)
    feature_types = list(layout["feature_types"])
    card_by_feature: dict[int, int] = layout["card_by_feature"]
    for i in range(n_features):
        v = feature_values[i]
        if v is None:
            if feature_types[i] == "cat":
                card = int(card_by_feature[i])
                v = rng.integers(0, card, size=n_rows, dtype=np.int64)
            else:
                v = rng.normal(size=n_rows).astype(np.float32)
        x[:, i] = np.asarray(v, dtype=np.float32)

    if target_values is None:
        if task == "classification":
            y = rng.integers(0, n_classes, size=n_rows, dtype=np.int64)
        else:
            y = rng.normal(size=n_rows).astype(np.float32)
    else:
        if task == "classification":
            y = np.asarray(target_values, dtype=np.int64) % n_classes
            if np.unique(y).size < 2:
                y = rng.integers(0, n_classes, size=n_rows, dtype=np.int64)
        else:
            y = np.asarray(target_values, dtype=np.float32)

    accepted, filter_details = _apply_filter_numpy(config, x, y, seed=seed)
    return x, y, {"filter": filter_details, "accepted": accepted}


def _generate_numpy(
    config: GeneratorConfig,
    layout: dict[str, Any],
    seed: int,
) -> DatasetBundle:
    """Generate one dataset in NumPy and enforce split validity constraints."""

    attempts = max(1, int(config.filter.max_attempts))
    last_reason = "unknown"

    for attempt in range(attempts):
        try:
            x, y, aux_meta = _generate_graph_dataset_numpy(
                config, layout, seed + attempt
            )
        except Exception as exc:
            last_reason = f"generation_exception:{exc.__class__.__name__}"
            continue
        if not bool(aux_meta.get("accepted", True)):
            last_reason = "filtered_out"
            continue

        rng = np.random.default_rng(seed + 10_007 + attempt)
        order = rng.permutation(x.shape[0])
        x = x[order]
        y = y[order]

        n_train = config.dataset.n_train
        x_train, x_test = x[:n_train], x[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]

        x_train, y_train, x_test, y_test, feature_types = postprocess_dataset(
            x_train,
            y_train,
            x_test,
            y_test,
            list(layout["feature_types"]),
            config.dataset.task,
            rng,
        )

        if config.dataset.task == "classification" and not _classification_split_valid(
            y_train, y_test
        ):
            last_reason = "invalid_class_split"
            continue

        metadata = {
            "backend": "numpy",
            "device": "cpu",
            "compute_backend": "numpy_appendix_light",
            "n_features": int(x_train.shape[1]),
            "n_categorical_features": int(sum(1 for t in feature_types if t == "cat")),
            "n_classes": (
                int(layout["n_classes"])
                if config.dataset.task == "classification"
                else None
            ),
            "graph_nodes": int(layout["graph_nodes"]),
            "graph_edges": int(layout["graph_edges"]),
            "seed": seed,
            "attempt_used": attempt,
            "filter": aux_meta.get("filter", {}),
            "config": asdict(config),
        }
        return DatasetBundle(
            X_train=x_train.astype(np.float32),
            y_train=y_train,
            X_test=x_test.astype(np.float32),
            y_test=y_test,
            feature_types=feature_types,
            metadata=metadata,
        )

    raise ValueError(
        f"Failed to generate a valid dataset after {attempts} attempts. Last reason: {last_reason}."
    )


def _bundle_to_torch(
    bundle: DatasetBundle,
    config: GeneratorConfig,
    device: str,
) -> DatasetBundle:
    """Convert NumPy bundle arrays to torch tensors on the requested device."""

    assert torch is not None
    dtype = _torch_dtype(config)
    x_train = torch.as_tensor(np.asarray(bundle.X_train), device=device, dtype=dtype)
    x_test = torch.as_tensor(np.asarray(bundle.X_test), device=device, dtype=dtype)
    if config.dataset.task == "classification":
        y_train = torch.as_tensor(
            np.asarray(bundle.y_train), device=device, dtype=torch.int64
        )
        y_test = torch.as_tensor(
            np.asarray(bundle.y_test), device=device, dtype=torch.int64
        )
    else:
        y_train = torch.as_tensor(
            np.asarray(bundle.y_train), device=device, dtype=dtype
        )
        y_test = torch.as_tensor(np.asarray(bundle.y_test), device=device, dtype=dtype)

    metadata = dict(bundle.metadata)
    metadata["backend"] = "torch"
    metadata["device"] = device
    return DatasetBundle(
        X_train=x_train,
        y_train=y_train,
        X_test=x_test,
        y_test=y_test,
        feature_types=list(bundle.feature_types),
        metadata=metadata,
    )


def _bundle_to_numpy(bundle: DatasetBundle) -> DatasetBundle:
    """Convert a torch bundle to NumPy arrays for compatibility workflows."""

    if torch is None:
        return bundle
    if not isinstance(bundle.X_train, torch.Tensor):
        return bundle

    return DatasetBundle(
        X_train=bundle.X_train.detach().cpu().numpy(),
        y_train=bundle.y_train.detach().cpu().numpy(),
        X_test=bundle.X_test.detach().cpu().numpy(),
        y_test=bundle.y_test.detach().cpu().numpy(),
        feature_types=list(bundle.feature_types),
        metadata=dict(bundle.metadata),
    )


def generate_one(
    config: GeneratorConfig,
    *,
    seed: int | None = None,
    device: str | None = None,
) -> DatasetBundle:
    """Generate one dataset bundle with deterministic per-dataset randomness."""

    manager = SeedManager(seed if seed is not None else config.seed)
    layout_rng = manager.numpy_rng("layout")
    layout = _sample_layout(config, layout_rng)
    data_seed = manager.child("data")

    backend, resolved_device = _resolve_backend(config, device)

    if backend == "torch":
        bundle = _generate_torch(config, layout, data_seed, resolved_device)
        return bundle if config.runtime.torch_output else _bundle_to_numpy(bundle)

    return _generate_numpy(config, layout, data_seed)


def generate_batch(
    config: GeneratorConfig,
    *,
    num_datasets: int,
    seed: int | None = None,
    device: str | None = None,
) -> list[DatasetBundle]:
    """Generate a batch of datasets using deterministic per-dataset child seeds."""

    run_seed = seed if seed is not None else config.seed
    manager = SeedManager(run_seed)
    return [
        generate_one(config, seed=manager.child("dataset", i), device=device)
        for i in range(num_datasets)
    ]
