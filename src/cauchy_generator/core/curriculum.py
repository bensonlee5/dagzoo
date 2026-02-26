"""Curriculum and stagewise graph sampling helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from typing import Any

import torch

from cauchy_generator.config import (
    CURRICULUM_STAGE_AUTO,
    CURRICULUM_STAGE_OFF,
    DatasetConfig,
    GeneratorConfig,
    normalize_curriculum_stage,
)
from cauchy_generator.graph import dag_edge_density, dag_longest_path_nodes, sample_cauchy_dag
from cauchy_generator.rng import SeedManager

_CURRICULUM_STAGE1_ROWS = 1024
_CURRICULUM_STAGE2_MIN_ROWS = 400
_CURRICULUM_STAGE2_MAX_ROWS = 10_240
_CURRICULUM_STAGE3_MIN_ROWS = 400
_CURRICULUM_STAGE3_MAX_ROWS = 60_000
_CURRICULUM_STAGE1_TRAIN_FRACTION_MIN = 0.30
_CURRICULUM_STAGE1_TRAIN_FRACTION_MAX = 0.90
_CURRICULUM_STAGE23_TRAIN_FRACTION = 0.80
# Fixed stagewise structural prior for RD-006/RD-090 scope.
# Tuning/configurability can be promoted to a later roadmap item if needed.
_CURRICULUM_STAGE_STRUCTURE_EDGE_LOGIT_BIAS: dict[int, float] = {1: -0.75, 2: 0.0, 3: 0.75}
_CURRICULUM_GRAPH_SAMPLING_MAX_ATTEMPTS = 64
_CURRICULUM_MONOTONICITY_AXES: tuple[str, ...] = (
    "n_rows_total",
    "n_features",
    "graph_nodes",
    "graph_depth_nodes",
)
_DEFAULT_CONFIGURED_N_TRAIN = int(DatasetConfig().n_train)
_DEFAULT_CONFIGURED_N_TEST = int(DatasetConfig().n_test)


@dataclass(slots=True, frozen=True)
class _StagewiseLayoutBounds:
    feature_min: int
    feature_max: int
    node_min: int
    node_max: int
    depth_min: int | None
    depth_max: int | None
    stage: int | None


def _sample_log_uniform_int(generator: torch.Generator, device: str, low: int, high: int) -> int:
    """Sample an integer from a log-uniform range [low, high]."""

    log_low = math.log(float(low))
    log_high = math.log(float(high))
    u = torch.empty(1, device=device).uniform_(log_low, log_high, generator=generator)
    sampled = int(math.exp(u.item()))
    return max(low, min(high, sampled))


def _split_counts(n_total: int, train_fraction: float) -> tuple[int, int]:
    """Split total rows into train/test while ensuring both sides are non-empty."""

    n_total = max(2, int(n_total))
    n_train = int(round(float(train_fraction) * n_total))
    n_train = min(max(1, n_train), n_total - 1)
    n_test = n_total - n_train
    return n_train, n_test


def _sample_stage_rows(stage: int, generator: torch.Generator, device: str) -> tuple[int, float]:
    """Sample total rows and train fraction for one curriculum stage."""

    if stage == 1:
        frac = (
            torch.empty(1, device=device)
            .uniform_(
                _CURRICULUM_STAGE1_TRAIN_FRACTION_MIN,
                _CURRICULUM_STAGE1_TRAIN_FRACTION_MAX,
                generator=generator,
            )
            .item()
        )
        return (_CURRICULUM_STAGE1_ROWS, float(frac))
    if stage == 2:
        return (
            _sample_log_uniform_int(
                generator, device, _CURRICULUM_STAGE2_MIN_ROWS, _CURRICULUM_STAGE2_MAX_ROWS
            ),
            _CURRICULUM_STAGE23_TRAIN_FRACTION,
        )
    if stage == 3:
        return (
            _sample_log_uniform_int(
                generator, device, _CURRICULUM_STAGE3_MIN_ROWS, _CURRICULUM_STAGE3_MAX_ROWS
            ),
            _CURRICULUM_STAGE23_TRAIN_FRACTION,
        )
    raise ValueError(f"Unsupported curriculum stage '{stage}'. Expected 1, 2, or 3.")


def _sample_auto_stage(generator: torch.Generator) -> int:
    """Sample a curriculum stage uniformly from 1..3."""

    return int(torch.randint(1, 4, (1,), generator=generator).item())


def _sample_curriculum(
    config: GeneratorConfig,
    manager: SeedManager,
    *,
    auto_stage: int,
    sample_stage_rows_fn: Callable[
        [int, torch.Generator, str], tuple[int, float]
    ] = _sample_stage_rows,
    split_counts_fn: Callable[[int, float], tuple[int, int]] = _split_counts,
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
    rows_gen = manager.torch_rng("curriculum", "rows", stage)
    sampled_rows_total, train_fraction = sample_stage_rows_fn(stage, rows_gen, "cpu")
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
    n_train, n_test = split_counts_fn(n_rows_total, train_fraction)
    return {
        "mode": CURRICULUM_STAGE_AUTO if mode == CURRICULUM_STAGE_AUTO else "fixed",
        "stage": stage,
        "n_rows_total": n_rows_total,
        "n_train": n_train,
        "n_test": n_test,
        "train_fraction": float(train_fraction),
    }


def _resolve_stagewise_layout_bounds(
    config: GeneratorConfig, curriculum: dict[str, Any]
) -> _StagewiseLayoutBounds:
    """Resolve effective feature/node sampling bounds for a curriculum stage."""

    feature_min = int(config.dataset.n_features_min)
    feature_max = int(config.dataset.n_features_max)
    node_min = int(config.graph.n_nodes_min)
    node_max = int(config.graph.n_nodes_max)
    depth_min: int | None = None
    depth_max: int | None = None
    stage: int | None = None

    stage_raw = curriculum.get("stage")
    if stage_raw is not None:
        stage = int(stage_raw)
        stage_cfg = config.curriculum.stages.get(stage)
        if stage_cfg is not None:
            if stage_cfg.n_features_min is not None:
                feature_min = int(stage_cfg.n_features_min)
            if stage_cfg.n_features_max is not None:
                feature_max = int(stage_cfg.n_features_max)
            if stage_cfg.n_nodes_min is not None:
                node_min = int(stage_cfg.n_nodes_min)
            if stage_cfg.n_nodes_max is not None:
                node_max = int(stage_cfg.n_nodes_max)
            if stage_cfg.depth_min is not None:
                depth_min = int(stage_cfg.depth_min)
            if stage_cfg.depth_max is not None:
                depth_max = int(stage_cfg.depth_max)

    if feature_min > feature_max:
        raise ValueError(
            "Invalid effective feature bounds after curriculum stage resolution: "
            f"n_features_min={feature_min} > n_features_max={feature_max}."
        )
    if node_min > node_max:
        raise ValueError(
            "Invalid effective node bounds after curriculum stage resolution: "
            f"n_nodes_min={node_min} > n_nodes_max={node_max}."
        )
    if depth_min is not None and depth_max is not None and depth_min > depth_max:
        raise ValueError(
            "Invalid effective depth bounds after curriculum stage resolution: "
            f"depth_min={depth_min} > depth_max={depth_max}."
        )
    return _StagewiseLayoutBounds(
        feature_min=feature_min,
        feature_max=feature_max,
        node_min=node_min,
        node_max=node_max,
        depth_min=depth_min,
        depth_max=depth_max,
        stage=stage,
    )


def _sample_stagewise_graph(
    n_nodes: int,
    stage: int | None,
    depth_min: int | None,
    depth_max: int | None,
    generator: torch.Generator,
    device: str,
) -> tuple[torch.Tensor, int, float]:
    """Sample DAG adjacency with optional stage-conditioned structure/depth constraints."""

    if stage is None:
        edge_logit_bias = 0.0
    else:
        edge_logit_bias = _CURRICULUM_STAGE_STRUCTURE_EDGE_LOGIT_BIAS.get(stage, 0.0)
    effective_depth_min = int(depth_min) if depth_min is not None else 1
    effective_depth_max = int(depth_max) if depth_max is not None else int(n_nodes)
    if effective_depth_min > effective_depth_max:
        raise ValueError(
            "Invalid effective stage depth bounds during graph sampling: "
            f"depth_min={effective_depth_min} > depth_max={effective_depth_max}."
        )

    for _ in range(_CURRICULUM_GRAPH_SAMPLING_MAX_ATTEMPTS):
        adjacency = sample_cauchy_dag(
            n_nodes,
            generator,
            device,
            edge_logit_bias=edge_logit_bias,
        )
        realized_depth = dag_longest_path_nodes(adjacency)
        if effective_depth_min <= realized_depth <= effective_depth_max:
            return adjacency, realized_depth, dag_edge_density(adjacency)

    raise ValueError(
        "Unable to sample DAG satisfying depth constraints after "
        f"{_CURRICULUM_GRAPH_SAMPLING_MAX_ATTEMPTS} attempts: "
        f"stage={stage}, n_nodes={n_nodes}, depth_min={effective_depth_min}, "
        f"depth_max={effective_depth_max}."
    )
