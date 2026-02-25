"""Typed configuration models and file loading."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

CurriculumStage = Literal["off", "auto", 1, 2, 3]
CURRICULUM_STAGE_OFF: Literal["off"] = "off"
CURRICULUM_STAGE_AUTO: Literal["auto"] = "auto"
CURRICULUM_STAGE_DEFAULT: CurriculumStage = CURRICULUM_STAGE_OFF
CURRICULUM_STAGE_CLI_CHOICES = (CURRICULUM_STAGE_AUTO, "1", "2", "3")

_CURRICULUM_STAGE_VALUE_MAP: dict[str | int, CurriculumStage] = {
    CURRICULUM_STAGE_OFF: CURRICULUM_STAGE_OFF,
    CURRICULUM_STAGE_AUTO: CURRICULUM_STAGE_AUTO,
    "1": 1,
    "2": 2,
    "3": 3,
    1: 1,
    2: 2,
    3: 3,
}

MissingnessMechanism = Literal["none", "mcar", "mar", "mnar"]
MISSINGNESS_MECHANISM_NONE: Literal["none"] = "none"
MISSINGNESS_MECHANISM_MCAR: Literal["mcar"] = "mcar"
MISSINGNESS_MECHANISM_MAR: Literal["mar"] = "mar"
MISSINGNESS_MECHANISM_MNAR: Literal["mnar"] = "mnar"

_MISSINGNESS_MECHANISM_VALUE_MAP: dict[str, MissingnessMechanism] = {
    MISSINGNESS_MECHANISM_NONE: MISSINGNESS_MECHANISM_NONE,
    MISSINGNESS_MECHANISM_MCAR: MISSINGNESS_MECHANISM_MCAR,
    MISSINGNESS_MECHANISM_MAR: MISSINGNESS_MECHANISM_MAR,
    MISSINGNESS_MECHANISM_MNAR: MISSINGNESS_MECHANISM_MNAR,
}


def normalize_curriculum_stage(value: str | int) -> CurriculumStage:
    """Normalize curriculum stage config into a validated internal value."""

    if isinstance(value, bool):
        raise ValueError(f"Unsupported curriculum_stage '{value}'. Expected off, auto, 1, 2, or 3.")
    if isinstance(value, str):
        value = value.strip().lower()
    result = _CURRICULUM_STAGE_VALUE_MAP.get(value)
    if result is None:
        raise ValueError(f"Unsupported curriculum_stage '{value}'. Expected off, auto, 1, 2, or 3.")
    return result


def normalize_missing_mechanism(value: str) -> MissingnessMechanism:
    """Normalize missingness mechanism into a validated internal value."""

    if isinstance(value, bool) or not isinstance(value, str):
        raise ValueError(
            f"Unsupported missing_mechanism '{value}'. Expected none, mcar, mar, or mnar."
        )
    normalized = value.strip().lower()
    result = _MISSINGNESS_MECHANISM_VALUE_MAP.get(normalized)
    if result is None:
        raise ValueError(
            f"Unsupported missing_mechanism '{value}'. Expected none, mcar, mar, or mnar."
        )
    return result


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
    missing_rate: float = 0.0
    missing_mechanism: MissingnessMechanism = MISSINGNESS_MECHANISM_NONE
    missing_mar_observed_fraction: float = 0.5
    missing_mar_logit_scale: float = 1.0
    missing_mnar_logit_scale: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.missing_rate, bool):
            raise ValueError(
                f"dataset.missing_rate must be a finite value in [0, 1], got {self.missing_rate!r}."
            )
        self.missing_rate = float(self.missing_rate)
        if not math.isfinite(self.missing_rate) or not (0.0 <= self.missing_rate <= 1.0):
            raise ValueError(
                f"dataset.missing_rate must be a finite value in [0, 1], got {self.missing_rate!r}."
            )

        self.missing_mechanism = normalize_missing_mechanism(self.missing_mechanism)
        if self.missing_rate > 0.0 and self.missing_mechanism == MISSINGNESS_MECHANISM_NONE:
            raise ValueError(
                "dataset.missing_mechanism must be mcar, mar, or mnar when dataset.missing_rate > 0."
            )

        if isinstance(self.missing_mar_observed_fraction, bool):
            raise ValueError(
                "dataset.missing_mar_observed_fraction must be in (0, 1], got "
                f"{self.missing_mar_observed_fraction!r}."
            )
        self.missing_mar_observed_fraction = float(self.missing_mar_observed_fraction)
        if not math.isfinite(self.missing_mar_observed_fraction) or not (
            0.0 < self.missing_mar_observed_fraction <= 1.0
        ):
            raise ValueError(
                "dataset.missing_mar_observed_fraction must be in (0, 1], got "
                f"{self.missing_mar_observed_fraction!r}."
            )

        if isinstance(self.missing_mar_logit_scale, bool):
            raise ValueError(
                "dataset.missing_mar_logit_scale must be a finite value > 0, got "
                f"{self.missing_mar_logit_scale!r}."
            )
        self.missing_mar_logit_scale = float(self.missing_mar_logit_scale)
        if not math.isfinite(self.missing_mar_logit_scale) or self.missing_mar_logit_scale <= 0.0:
            raise ValueError(
                "dataset.missing_mar_logit_scale must be a finite value > 0, got "
                f"{self.missing_mar_logit_scale!r}."
            )

        if isinstance(self.missing_mnar_logit_scale, bool):
            raise ValueError(
                "dataset.missing_mnar_logit_scale must be a finite value > 0, got "
                f"{self.missing_mnar_logit_scale!r}."
            )
        self.missing_mnar_logit_scale = float(self.missing_mnar_logit_scale)
        if not math.isfinite(self.missing_mnar_logit_scale) or self.missing_mnar_logit_scale <= 0.0:
            raise ValueError(
                "dataset.missing_mnar_logit_scale must be a finite value > 0, got "
                f"{self.missing_mnar_logit_scale!r}."
            )


@dataclass(slots=True)
class GraphConfig:
    n_nodes_min: int = 2
    n_nodes_max: int = 32


def _validate_optional_int_bound(name: str, value: object | None, *, minimum: int) -> int | None:
    """Validate an optional integer bound and normalize to int."""

    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}.")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        normalized = value.strip()
        signless = normalized[1:] if normalized.startswith(("+", "-")) else normalized
        if not signless.isdigit():
            raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}.")
        parsed = int(normalized)
    else:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}.")
    if parsed < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}.")
    return parsed


@dataclass(slots=True)
class CurriculumStageConfig:
    n_features_min: int | None = None
    n_features_max: int | None = None
    n_nodes_min: int | None = None
    n_nodes_max: int | None = None
    depth_min: int | None = None
    depth_max: int | None = None

    def __post_init__(self) -> None:
        self.n_features_min = _validate_optional_int_bound(
            "curriculum.stages.*.n_features_min", self.n_features_min, minimum=1
        )
        self.n_features_max = _validate_optional_int_bound(
            "curriculum.stages.*.n_features_max", self.n_features_max, minimum=1
        )
        self.n_nodes_min = _validate_optional_int_bound(
            "curriculum.stages.*.n_nodes_min", self.n_nodes_min, minimum=2
        )
        self.n_nodes_max = _validate_optional_int_bound(
            "curriculum.stages.*.n_nodes_max", self.n_nodes_max, minimum=2
        )
        self.depth_min = _validate_optional_int_bound(
            "curriculum.stages.*.depth_min", self.depth_min, minimum=1
        )
        self.depth_max = _validate_optional_int_bound(
            "curriculum.stages.*.depth_max", self.depth_max, minimum=1
        )

        if (
            self.n_features_min is not None
            and self.n_features_max is not None
            and self.n_features_min > self.n_features_max
        ):
            raise ValueError(
                "curriculum.stages.*.n_features_min must be <= n_features_max, got "
                f"{self.n_features_min} > {self.n_features_max}."
            )
        if (
            self.n_nodes_min is not None
            and self.n_nodes_max is not None
            and self.n_nodes_min > self.n_nodes_max
        ):
            raise ValueError(
                "curriculum.stages.*.n_nodes_min must be <= n_nodes_max, got "
                f"{self.n_nodes_min} > {self.n_nodes_max}."
            )
        if (
            self.depth_min is not None
            and self.depth_max is not None
            and self.depth_min > self.depth_max
        ):
            raise ValueError(
                "curriculum.stages.*.depth_min must be <= depth_max, got "
                f"{self.depth_min} > {self.depth_max}."
            )


def _normalize_curriculum_stage_key(value: str | int) -> int:
    """Normalize stage key into 1..3 integer stage."""

    if isinstance(value, bool):
        raise ValueError(f"Unsupported curriculum stage key '{value}'. Expected 1, 2, or 3.")
    if isinstance(value, int):
        stage = value
    elif isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            raise ValueError("Unsupported curriculum stage key ''. Expected 1, 2, or 3.")
        signless = normalized[1:] if normalized.startswith(("+", "-")) else normalized
        if not signless.isdigit():
            raise ValueError(f"Unsupported curriculum stage key '{value}'. Expected 1, 2, or 3.")
        stage = int(normalized)
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported curriculum stage key '{value}'. Expected 1, 2, or 3.")
    if stage not in (1, 2, 3):
        raise ValueError(f"Unsupported curriculum stage key '{value}'. Expected 1, 2, or 3.")
    return stage


@dataclass(slots=True)
class CurriculumConfig:
    stages: dict[int, CurriculumStageConfig] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: object | None) -> "CurriculumConfig":
        """Construct curriculum config from dictionary payload."""

        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError("curriculum must be a mapping.")
        raw_stages = data.get("stages", {})
        if raw_stages is None:
            raw_stages = {}
        if not isinstance(raw_stages, dict):
            raise ValueError("curriculum.stages must be a mapping keyed by stage (1,2,3).")
        stages: dict[int, CurriculumStageConfig] = {}
        for raw_key, raw_stage_config in raw_stages.items():
            stage = _normalize_curriculum_stage_key(raw_key)
            if stage in stages:
                raise ValueError(f"Duplicate curriculum stage '{stage}' in curriculum.stages.")
            if not isinstance(raw_stage_config, dict):
                raise ValueError(
                    f"curriculum.stages[{raw_key!r}] must be a mapping of stage bounds."
                )
            stages[stage] = CurriculumStageConfig(**raw_stage_config)
        return cls(stages=stages)


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
class DiagnosticsConfig:
    enabled: bool = False
    include_spearman: bool = False
    histogram_bins: int = 10
    quantiles: list[float] = field(default_factory=lambda: [0.05, 0.25, 0.50, 0.75, 0.95])
    underrepresented_threshold: float = 0.5
    max_values_per_metric: int | None = 50_000
    meta_feature_targets: dict[str, list[float] | tuple[float, float]] = field(default_factory=dict)
    out_dir: str | None = None


@dataclass(slots=True)
class SteeringConfig:
    enabled: bool = False
    max_attempts: int = 3
    temperature: float = 0.35


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
    curriculum_stage: str | int = CURRICULUM_STAGE_DEFAULT
    meta_feature_targets: dict[str, list[float] | tuple[float, ...]] = field(default_factory=dict)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    steering: SteeringConfig = field(default_factory=SteeringConfig)
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
        curriculum = CurriculumConfig.from_dict(data.get("curriculum"))
        runtime = RuntimeConfig(**(data.get("runtime") or {}))
        output = OutputConfig(**(data.get("output") or {}))
        diagnostics = DiagnosticsConfig(**(data.get("diagnostics") or {}))
        steering = SteeringConfig(**(data.get("steering") or {}))
        benchmark = BenchmarkConfig(**(data.get("benchmark") or {}))
        filter_cfg = FilterConfig(**(data.get("filter") or {}))
        raw_meta_targets = data.get("meta_feature_targets")
        meta_feature_targets = dict(raw_meta_targets) if isinstance(raw_meta_targets, dict) else {}
        seed = int(data.get("seed", 1))
        curriculum_stage: str | int = data.get("curriculum_stage", CURRICULUM_STAGE_DEFAULT)
        return cls(
            seed=seed,
            curriculum_stage=curriculum_stage,
            meta_feature_targets=meta_feature_targets,
            dataset=dataset,
            graph=graph,
            curriculum=curriculum,
            runtime=runtime,
            output=output,
            diagnostics=diagnostics,
            steering=steering,
            benchmark=benchmark,
            filter=filter_cfg,
        )

    def __post_init__(self) -> None:
        dataset_n_features_min = int(self.dataset.n_features_min)
        dataset_n_features_max = int(self.dataset.n_features_max)
        graph_n_nodes_min = int(self.graph.n_nodes_min)
        graph_n_nodes_max = int(self.graph.n_nodes_max)
        for stage, stage_cfg in self.curriculum.stages.items():
            _ = stage
            if stage_cfg.n_features_min is not None:
                if stage_cfg.n_features_min < dataset_n_features_min:
                    raise ValueError(
                        "curriculum.stages.*.n_features_min must be >= dataset.n_features_min "
                        f"({dataset_n_features_min}), got {stage_cfg.n_features_min}."
                    )
                if stage_cfg.n_features_min > dataset_n_features_max:
                    raise ValueError(
                        "curriculum.stages.*.n_features_min must be <= dataset.n_features_max "
                        f"({dataset_n_features_max}), got {stage_cfg.n_features_min}."
                    )
            if stage_cfg.n_features_max is not None:
                if stage_cfg.n_features_max > dataset_n_features_max:
                    raise ValueError(
                        "curriculum.stages.*.n_features_max must be <= dataset.n_features_max "
                        f"({dataset_n_features_max}), got {stage_cfg.n_features_max}."
                    )
                if stage_cfg.n_features_max < dataset_n_features_min:
                    raise ValueError(
                        "curriculum.stages.*.n_features_max must be >= dataset.n_features_min "
                        f"({dataset_n_features_min}), got {stage_cfg.n_features_max}."
                    )
            if stage_cfg.n_nodes_min is not None:
                if stage_cfg.n_nodes_min < graph_n_nodes_min:
                    raise ValueError(
                        "curriculum.stages.*.n_nodes_min must be >= graph.n_nodes_min "
                        f"({graph_n_nodes_min}), got {stage_cfg.n_nodes_min}."
                    )
                if stage_cfg.n_nodes_min > graph_n_nodes_max:
                    raise ValueError(
                        "curriculum.stages.*.n_nodes_min must be <= graph.n_nodes_max "
                        f"({graph_n_nodes_max}), got {stage_cfg.n_nodes_min}."
                    )
            if stage_cfg.n_nodes_max is not None:
                if stage_cfg.n_nodes_max > graph_n_nodes_max:
                    raise ValueError(
                        "curriculum.stages.*.n_nodes_max must be <= graph.n_nodes_max "
                        f"({graph_n_nodes_max}), got {stage_cfg.n_nodes_max}."
                    )
                if stage_cfg.n_nodes_max < graph_n_nodes_min:
                    raise ValueError(
                        "curriculum.stages.*.n_nodes_max must be >= graph.n_nodes_min "
                        f"({graph_n_nodes_min}), got {stage_cfg.n_nodes_max}."
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
