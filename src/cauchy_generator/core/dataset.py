"""Seeded synthetic dataset generation with Torch execution on all devices."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict
from typing import Any

import numpy as np
import torch

from cauchy_generator.config import (
    CURRICULUM_STAGE_AUTO,
    CURRICULUM_STAGE_OFF,
    DatasetConfig,
    GeneratorConfig,
    normalize_curriculum_stage,
)
from cauchy_generator.math_utils import to_numpy as _to_numpy
from cauchy_generator.core.node_pipeline import (
    ConverterSpec,
    apply_node_pipeline_torch,
)
from cauchy_generator.filtering import apply_torch_rf_filter
from cauchy_generator.graph import sample_cauchy_dag
from cauchy_generator.postprocess import postprocess_dataset
from cauchy_generator.rng import SeedManager
from cauchy_generator.sampling import CorrelatedSampler
from cauchy_generator.types import DatasetBundle

_CURRICULUM_STAGE1_ROWS = 1024
_CURRICULUM_STAGE2_MIN_ROWS = 400
_CURRICULUM_STAGE2_MAX_ROWS = 10_240
_CURRICULUM_STAGE3_MIN_ROWS = 400
_CURRICULUM_STAGE3_MAX_ROWS = 60_000
_CURRICULUM_STAGE1_TRAIN_FRACTION_MIN = 0.30
_CURRICULUM_STAGE1_TRAIN_FRACTION_MAX = 0.90
_CURRICULUM_STAGE23_TRAIN_FRACTION = 0.80
_DEFAULT_CONFIGURED_N_TRAIN = int(DatasetConfig().n_train)
_DEFAULT_CONFIGURED_N_TEST = int(DatasetConfig().n_test)


def _sample_log_uniform_int(rng: np.random.Generator, low: int, high: int) -> int:
    """Sample an integer from a log-uniform range [low, high]."""

    sampled = int(np.exp(rng.uniform(np.log(float(low)), np.log(float(high)))))
    return int(np.clip(sampled, low, high))


def _split_counts(n_total: int, train_fraction: float) -> tuple[int, int]:
    """Split total rows into train/test while ensuring both sides are non-empty."""

    n_total = max(2, int(n_total))
    n_train = int(round(float(train_fraction) * n_total))
    n_train = min(max(1, n_train), n_total - 1)
    n_test = n_total - n_train
    return n_train, n_test


def _sample_stage_rows(stage: int, rng: np.random.Generator) -> tuple[int, float]:
    """Sample total rows and train fraction for one curriculum stage."""

    if stage == 1:
        return (
            _CURRICULUM_STAGE1_ROWS,
            float(
                rng.uniform(
                    _CURRICULUM_STAGE1_TRAIN_FRACTION_MIN,
                    _CURRICULUM_STAGE1_TRAIN_FRACTION_MAX,
                )
            ),
        )
    if stage == 2:
        return (
            _sample_log_uniform_int(rng, _CURRICULUM_STAGE2_MIN_ROWS, _CURRICULUM_STAGE2_MAX_ROWS),
            _CURRICULUM_STAGE23_TRAIN_FRACTION,
        )
    if stage == 3:
        return (
            _sample_log_uniform_int(rng, _CURRICULUM_STAGE3_MIN_ROWS, _CURRICULUM_STAGE3_MAX_ROWS),
            _CURRICULUM_STAGE23_TRAIN_FRACTION,
        )
    raise ValueError(f"Unsupported curriculum stage '{stage}'. Expected 1, 2, or 3.")


def _sample_auto_stage(rng: np.random.Generator) -> int:
    """Sample a curriculum stage uniformly from 1..3."""

    return int(rng.integers(1, 4))


def _sample_curriculum(
    config: GeneratorConfig,
    manager: SeedManager,
    *,
    auto_stage: int,
) -> dict[str, Any]:
    """Resolve stage and sample row/split regime for this dataset seed."""

    mode = normalize_curriculum_stage(config.curriculum_stage)
    configured_n_train = max(1, int(config.dataset.n_train))
    configured_n_test = max(1, int(config.dataset.n_test))
    configured_total = configured_n_train + configured_n_test
    if mode == CURRICULUM_STAGE_OFF:
        return {
            "mode": "off",
            "stage": None,
            "n_rows_total": configured_total,
            "n_train": configured_n_train,
            "n_test": configured_n_test,
            "train_fraction": float(configured_n_train / configured_total),
        }

    stage = auto_stage if mode == CURRICULUM_STAGE_AUTO else int(mode)
    rows_rng = manager.numpy_rng("curriculum", "rows", stage)
    sampled_rows_total, train_fraction = _sample_stage_rows(stage, rows_rng)
    configured_total = max(2, configured_total)
    n_rows_total = sampled_rows_total
    # Preserve caller-provided split-size knobs as a workload ceiling when
    # they intentionally deviate from baseline defaults.
    has_split_override = (
        configured_n_train != _DEFAULT_CONFIGURED_N_TRAIN
        or configured_n_test != _DEFAULT_CONFIGURED_N_TEST
    )
    if has_split_override:
        n_rows_total = min(sampled_rows_total, configured_total)
    n_train, n_test = _split_counts(n_rows_total, train_fraction)
    return {
        "mode": CURRICULUM_STAGE_AUTO if mode == CURRICULUM_STAGE_AUTO else "fixed",
        "stage": stage,
        "n_rows_total": n_rows_total,
        "n_train": n_train,
        "n_test": n_test,
        "train_fraction": float(train_fraction),
    }


def _resolve_device(config: GeneratorConfig, device_override: str | None) -> str:
    """Resolve runtime device and hard-fail on unavailable explicit accelerators."""

    requested = (device_override or config.runtime.device or "auto").lower()
    mps_ok = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if mps_ok:
            return "mps"
        return "cpu"
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        raise RuntimeError("Requested device 'cuda' but CUDA is not available.")
    if requested == "mps":
        if mps_ok:
            return "mps"
        raise RuntimeError("Requested device 'mps' but MPS is not available.")
    raise ValueError(f"Unsupported device '{requested}'. Expected one of: auto, cpu, cuda, mps.")


def _torch_dtype(config: GeneratorConfig) -> torch.dtype:
    """Map string runtime dtype configuration to a torch dtype."""

    return torch.float64 if config.runtime.torch_dtype == "float64" else torch.float32


def _to_torch(arr: Any, device: str, dtype: torch.dtype) -> torch.Tensor:
    """Convert an array-like to a torch tensor on the given device."""
    return torch.as_tensor(np.asarray(arr), device=device, dtype=dtype)


def _sample_node_count(config: GeneratorConfig, rng: np.random.Generator) -> int:
    """Sample graph node count using log-uniform bounds from config."""

    low = max(2.0, float(config.graph.n_nodes_min))
    high = max(low, float(config.graph.n_nodes_max))
    return max(2, int(np.exp(rng.uniform(np.log(low), np.log(high)))))


def _sample_assignments(n_cols: int, n_nodes: int, rng: np.random.Generator) -> np.ndarray:
    """Assign columns to a random eligible subset of graph nodes."""

    eligible_count = int(rng.integers(1, n_nodes + 1))
    eligible_nodes = rng.choice(np.arange(n_nodes), size=eligible_count, replace=False)
    return rng.choice(eligible_nodes, size=n_cols, replace=True).astype(np.int32)


def _sample_layout(config: GeneratorConfig, rng: np.random.Generator) -> dict[str, Any]:
    """Sample dataset layout, graph, and node assignments for one dataset instance."""

    n_features = int(rng.integers(config.dataset.n_features_min, config.dataset.n_features_max + 1))

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

    n_classes = int(rng.integers(config.dataset.n_classes_min, config.dataset.n_classes_max + 1))
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
                ConverterSpec(key=f"feature_{f}", kind="cat", dim=max(1, d), cardinality=c)
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
    *,
    n_rows: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Generate raw X/y tensors via the Torch graph pipeline."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

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

        x_node, extracted = apply_node_pipeline_torch(parent_data, n_rows, specs, generator, device)
        node_outputs[node_idx] = x_node

        for key, values in extracted.items():
            if key.startswith("feature_"):
                col = int(key.split("_", maxsplit=1)[1])
                feature_values[col] = values
            elif key == "target":
                target_values = values

    dtype = _torch_dtype(config)
    x = torch.zeros((n_rows, n_features), dtype=dtype, device=device)
    feature_types = list(layout["feature_types"])
    card_by_feature: dict[int, int] = layout["card_by_feature"]

    for i in range(n_features):
        v = feature_values[i]
        if v is None:
            if feature_types[i] == "cat":
                card = int(card_by_feature[i])
                v = torch.randint(0, card, (n_rows,), generator=generator, device=device)
            else:
                v = torch.randn(n_rows, generator=generator, device=device)
        x[:, i] = v.to(dtype)

    if target_values is None:
        if task == "classification":
            y = torch.randint(0, n_classes, (n_rows,), generator=generator, device=device)
        else:
            y = torch.randn(n_rows, generator=generator, device=device).to(dtype)
    else:
        if task == "classification":
            y = target_values.to(torch.long) % n_classes
        else:
            y = target_values.to(dtype)

    accepted, filter_details = _apply_filter_torch(config, x, y, seed=seed)
    return x, y, {"accepted": accepted, "filter": filter_details}


def _generate_torch(
    config: GeneratorConfig,
    layout: dict[str, Any],
    seed: int,
    device: str,
    *,
    n_train: int,
    n_test: int,
    curriculum: dict[str, Any],
) -> DatasetBundle:
    """Generate one dataset in Torch while preserving Appendix E postprocess/filter contracts."""

    attempts = max(1, int(config.filter.max_attempts))
    last_reason = "unknown"
    dtype = _torch_dtype(config)
    n_rows = n_train + n_test

    for attempt in range(attempts):
        try:
            x, y, aux_meta = _generate_graph_dataset_torch(
                config,
                layout,
                seed + attempt,
                device,
                n_rows=n_rows,
            )
        except Exception as exc:
            last_reason = f"generation_exception:{exc.__class__.__name__}"
            continue
        if not bool(aux_meta.get("accepted", True)):
            last_reason = "filtered_out"
            continue

        generator = torch.Generator(device=device)
        generator.manual_seed(seed + 10_007 + attempt)
        order = torch.randperm(x.shape[0], generator=generator, device=device)
        x = x[order]
        y = y[order]

        x_train_t, x_test_t = x[:n_train], x[n_train:]
        y_train_t, y_test_t = y[:n_train], y[n_train:]

        # Reuse the validated NumPy postprocessing implementation to keep
        # Appendix E.13 behavior and split constraints consistent.
        rng = np.random.default_rng(seed + 10_007 + attempt)
        x_train_np, y_train_np, x_test_np, y_test_np, feature_types = postprocess_dataset(
            _to_numpy(x_train_t),
            _to_numpy(y_train_t),
            _to_numpy(x_test_t),
            _to_numpy(y_test_t),
            list(layout["feature_types"]),
            config.dataset.task,
            rng,
        )

        if config.dataset.task == "classification" and not _classification_split_valid(
            y_train_np, y_test_np
        ):
            last_reason = "invalid_class_split"
            continue

        x_train = _to_torch(x_train_np, device, dtype)
        x_test = _to_torch(x_test_np, device, dtype)
        y_dtype = torch.int64 if config.dataset.task == "classification" else dtype
        y_train = _to_torch(y_train_np, device, y_dtype)
        y_test = _to_torch(y_test_np, device, y_dtype)

        metadata = {
            "backend": "torch",
            "device": device,
            "compute_backend": "torch_appendix_full",
            "n_features": int(x_train.shape[1]),
            "n_categorical_features": int(sum(1 for t in feature_types if t == "cat")),
            "n_classes": (
                int(layout["n_classes"]) if config.dataset.task == "classification" else None
            ),
            "graph_nodes": int(layout["graph_nodes"]),
            "graph_edges": int(layout["graph_edges"]),
            "seed": seed,
            "attempt_used": attempt,
            "filter": aux_meta.get("filter", {}),
            "curriculum": curriculum,
            "config": asdict(config),
        }
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


def _classification_split_valid(y_train: np.ndarray, y_test: np.ndarray) -> bool:
    """Validate classification split constraints from Appendix E.13."""

    train_classes = set(np.unique(y_train).tolist())
    test_classes = set(np.unique(y_test).tolist())
    return len(train_classes) >= 2 and train_classes == test_classes


def _apply_filter_torch(
    config: GeneratorConfig,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    seed: int,
) -> tuple[bool, dict[str, Any]]:
    """Run E.14-style filtering natively in Torch using random forests."""

    details: dict[str, Any] = {"enabled": config.filter.enabled}
    if not config.filter.enabled:
        return True, details

    accepted, filter_details = apply_torch_rf_filter(
        x,
        y,
        task=config.dataset.task,
        seed=seed,
        n_trees=config.filter.n_trees,
        depth=config.filter.depth,
        min_samples_leaf=config.filter.min_samples_leaf,
        max_leaf_nodes=config.filter.max_leaf_nodes,
        max_features=config.filter.max_features,
        n_split_candidates=config.filter.n_split_candidates,
        n_bootstrap=config.filter.n_bootstrap,
        threshold=config.filter.threshold,
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
    auto_stage: int,
) -> DatasetBundle:
    """Generate one dataset for a fully resolved seed/device/stage context."""

    manager = SeedManager(seed)
    curriculum = _sample_curriculum(config, manager, auto_stage=auto_stage)
    layout_rng = manager.numpy_rng("layout")
    layout = _sample_layout(config, layout_rng)
    data_seed = manager.child("data")
    n_train = int(curriculum["n_train"])
    n_test = int(curriculum["n_test"])

    if requested_device == "auto" and resolved_device == "mps":
        try:
            return _generate_torch(
                config,
                layout,
                data_seed,
                resolved_device,
                n_train=n_train,
                n_test=n_test,
                curriculum=curriculum,
            )
        except Exception:
            # Keep auto mode robust on partially supported MPS runtimes by retrying on CPU.
            return _generate_torch(
                config,
                layout,
                data_seed,
                "cpu",
                n_train=n_train,
                n_test=n_test,
                curriculum=curriculum,
            )

    return _generate_torch(
        config,
        layout,
        data_seed,
        resolved_device,
        n_train=n_train,
        n_test=n_test,
        curriculum=curriculum,
    )


def generate_one(
    config: GeneratorConfig,
    *,
    seed: int | None = None,
    device: str | None = None,
) -> DatasetBundle:
    """Generate one dataset bundle with deterministic per-dataset randomness."""

    mode = normalize_curriculum_stage(config.curriculum_stage)
    auto_stage = 1
    if isinstance(mode, int):
        auto_stage = mode
    run_seed = seed if seed is not None else config.seed
    requested_device = (device or config.runtime.device or "auto").lower()
    resolved_device = _resolve_device(config, device)
    return _generate_one_seeded(
        config,
        seed=run_seed,
        requested_device=requested_device,
        resolved_device=resolved_device,
        auto_stage=auto_stage,
    )


def generate_batch(
    config: GeneratorConfig,
    *,
    num_datasets: int,
    seed: int | None = None,
    device: str | None = None,
) -> list[DatasetBundle]:
    """Generate a batch of datasets using deterministic per-dataset child seeds."""

    return list(
        generate_batch_iter(
            config,
            num_datasets=num_datasets,
            seed=seed,
            device=device,
        )
    )


def generate_batch_iter(
    config: GeneratorConfig,
    *,
    num_datasets: int,
    seed: int | None = None,
    device: str | None = None,
) -> Iterator[DatasetBundle]:
    """Yield datasets lazily using deterministic per-dataset child seeds."""

    if num_datasets < 0:
        raise ValueError(f"num_datasets must be >= 0, got {num_datasets}")
    if num_datasets == 0:
        return

    mode = normalize_curriculum_stage(config.curriculum_stage)
    requested_device = (device or config.runtime.device or "auto").lower()
    resolved_device = _resolve_device(config, device)
    run_seed = seed if seed is not None else config.seed
    manager = SeedManager(run_seed)
    for i in range(num_datasets):
        dataset_seed = manager.child("dataset", i)
        auto_stage = 1
        if mode == CURRICULUM_STAGE_AUTO:
            stage_rng = SeedManager(dataset_seed).numpy_rng("curriculum", "stage")
            auto_stage = _sample_auto_stage(stage_rng)

        yield _generate_one_seeded(
            config,
            seed=dataset_seed,
            requested_device=requested_device,
            resolved_device=resolved_device,
            auto_stage=auto_stage,
        )
