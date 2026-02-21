"""Typed configuration models and file loading."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class DatasetConfig:
    task: str = "classification"
    n_train: int = 768
    n_test: int = 256
    n_features_min: int = 16
    n_features_max: int = 64
    n_classes_min: int = 2
    n_classes_max: int = 10
    categorical_ratio_min: float = -0.5
    categorical_ratio_max: float = 1.2
    max_categorical_cardinality: int = 9


@dataclass(slots=True)
class GraphConfig:
    n_nodes_min: int = 2
    n_nodes_max: int = 32


@dataclass(slots=True)
class RuntimeConfig:
    device: str = "auto"
    torch_dtype: str = "float32"
    generation_engine: str = "appendix_light"
    hardware_aware: bool = True
    gpu_name_hint: str | None = None
    gpu_memory_gb_hint: float | None = None
    peak_flops_hint: float | None = None


@dataclass(slots=True)
class OutputConfig:
    out_dir: str = "data/run_default"
    shard_size: int = 128
    compression: str = "zstd"


@dataclass(slots=True)
class BenchmarkConfig:
    profile_name: str = "medium_cuda"
    num_datasets: int = 2000
    warmup_datasets: int = 25
    suite: str = "standard"
    warn_threshold_pct: float = 10.0
    fail_threshold_pct: float = 20.0
    collect_memory: bool = True
    collect_reproducibility: bool = False
    reproducibility_num_datasets: int = 2
    latency_num_samples: int = 20
    profiles: dict[str, dict[str, int | str]] = field(
        default_factory=lambda: {
            "cpu": {"num_datasets": 200, "warmup_datasets": 10, "device": "cpu"},
            "cuda_desktop": {"num_datasets": 2000, "warmup_datasets": 25, "device": "cuda"},
            "cuda_h100": {"num_datasets": 5000, "warmup_datasets": 50, "device": "cuda"},
        }
    )


@dataclass(slots=True)
class FilterConfig:
    enabled: bool = False
    n_trees: int = 25
    depth: int = 6
    min_samples_leaf: int = 1
    max_leaf_nodes: int | None = None
    max_features: str | int | float = "auto"
    n_split_candidates: int = 8
    n_bootstrap: int = 200
    threshold: float = 0.95
    max_attempts: int = 3


@dataclass(slots=True)
class GeneratorConfig:
    seed: int = 1
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "GeneratorConfig":
        """Construct `GeneratorConfig` from a nested dictionary payload."""

        data = data or {}
        dataset = DatasetConfig(**(data.get("dataset") or {}))
        graph_data = dict(data.get("graph") or {})
        if "n_nodes_log2_min" in graph_data and "n_nodes_min" not in graph_data:
            graph_data["n_nodes_min"] = graph_data.pop("n_nodes_log2_min")
        if "n_nodes_log2_max" in graph_data and "n_nodes_max" not in graph_data:
            graph_data["n_nodes_max"] = graph_data.pop("n_nodes_log2_max")
        graph = GraphConfig(**graph_data)
        runtime = RuntimeConfig(**(data.get("runtime") or {}))
        output = OutputConfig(**(data.get("output") or {}))
        benchmark = BenchmarkConfig(**(data.get("benchmark") or {}))
        filter_cfg = FilterConfig(**(data.get("filter") or {}))
        seed = int(data.get("seed", 1))
        return cls(
            seed=seed,
            dataset=dataset,
            graph=graph,
            runtime=runtime,
            output=output,
            benchmark=benchmark,
            filter=filter_cfg,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GeneratorConfig":
        """Load config from a YAML file path."""

        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config at {p} must be a mapping at the top level.")
        return cls.from_dict(loaded)

    def to_dict(self) -> dict[str, Any]:
        """Serialize config dataclasses into a plain nested dictionary."""

        return asdict(self)
