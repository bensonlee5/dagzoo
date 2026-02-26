"""Component-level microbenchmarks for hot path sanity checks."""

from __future__ import annotations

import time
from typing import Callable

import torch

from cauchy_generator.bench.constants import (
    MICROBENCH_BASE_SEED_OFFSET,
    MICROBENCH_DEFAULT_REPEATS,
    MICROBENCH_GENERATE_ONE_SEED_OFFSET,
    MICROBENCH_LINEAR_SEED,
    MICROBENCH_NODE_PIPELINE_SEED,
    MICROBENCH_PARENT_DIM,
    MICROBENCH_SYNTH_FEATURES,
    MICROBENCH_SYNTH_ROWS,
    MICRO_CONFIG_N_FEATURES_CAP,
    MICRO_CONFIG_N_NODES_CAP,
    MICRO_CONFIG_N_TEST_CAP,
    MICRO_CONFIG_N_TRAIN_CAP,
    MILLISECONDS_PER_SECOND,
)
from cauchy_generator.config import GeneratorConfig
from cauchy_generator.core.dataset import generate_one
from cauchy_generator.core.node_pipeline import ConverterSpec, apply_node_pipeline
from cauchy_generator.functions.random_functions import apply_random_function


def _time_ms(func: Callable[[], None], repeats: int) -> float:
    """Run ``func`` repeatedly and return the average runtime in milliseconds."""

    start = time.perf_counter()
    for _ in range(max(1, repeats)):
        func()
    elapsed = time.perf_counter() - start
    return (elapsed / max(1, repeats)) * MILLISECONDS_PER_SECOND


def _micro_config(config: GeneratorConfig) -> GeneratorConfig:
    """Create a lightweight config variant for inexpensive microbenchmarks."""

    c = GeneratorConfig.from_dict(config.to_dict())
    c.dataset.n_train = min(MICRO_CONFIG_N_TRAIN_CAP, c.dataset.n_train)
    c.dataset.n_test = min(MICRO_CONFIG_N_TEST_CAP, c.dataset.n_test)
    c.dataset.n_features_min = min(MICRO_CONFIG_N_FEATURES_CAP, c.dataset.n_features_min)
    c.dataset.n_features_max = min(MICRO_CONFIG_N_FEATURES_CAP, c.dataset.n_features_max)
    c.graph.n_nodes_min = min(MICRO_CONFIG_N_NODES_CAP, c.graph.n_nodes_min)
    c.graph.n_nodes_max = min(MICRO_CONFIG_N_NODES_CAP, c.graph.n_nodes_max)
    return c


def run_microbenchmarks(
    config: GeneratorConfig,
    *,
    device: str | None = None,
    repeats: int = MICROBENCH_DEFAULT_REPEATS,
) -> dict[str, float | int]:
    """Run targeted microbenchmarks for core generation components."""

    g_rng = torch.Generator(device="cpu")
    g_rng.manual_seed(config.seed + MICROBENCH_BASE_SEED_OFFSET)
    x = torch.randn(MICROBENCH_SYNTH_ROWS, MICROBENCH_SYNTH_FEATURES, generator=g_rng)

    def run_linear() -> None:
        local_g = torch.Generator(device="cpu")
        local_g.manual_seed(MICROBENCH_LINEAR_SEED)
        _ = apply_random_function(
            x,
            local_g,
            out_dim=MICROBENCH_SYNTH_FEATURES,
            function_type="linear",
        )

    parent_data = [
        torch.randn(MICROBENCH_SYNTH_ROWS, MICROBENCH_PARENT_DIM),
        torch.randn(MICROBENCH_SYNTH_ROWS, MICROBENCH_PARENT_DIM),
    ]
    specs = [
        ConverterSpec(key="feature_0", kind="num", dim=1),
        ConverterSpec(key="feature_1", kind="cat", dim=4, cardinality=6),
        ConverterSpec(key="target", kind="target_cls", dim=3, cardinality=3),
    ]

    def run_node_pipeline() -> None:
        g = torch.Generator(device="cpu")
        g.manual_seed(MICROBENCH_NODE_PIPELINE_SEED)
        _ = apply_node_pipeline(parent_data, MICROBENCH_SYNTH_ROWS, specs, g, "cpu")

    micro_cfg = _micro_config(config)

    def run_generate_one() -> None:
        _ = generate_one(
            micro_cfg,
            seed=micro_cfg.seed + MICROBENCH_GENERATE_ONE_SEED_OFFSET,
            device=device,
        )

    return {
        "micro_repeats": int(max(1, repeats)),
        "micro_random_function_linear_ms": _time_ms(run_linear, repeats),
        "micro_node_pipeline_ms": _time_ms(run_node_pipeline, repeats),
        "micro_generate_one_ms": _time_ms(run_generate_one, repeats),
    }
