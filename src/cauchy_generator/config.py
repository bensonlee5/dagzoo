"""Typed configuration models and file loading."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from cauchy_generator.math_utils import normalize_positive_weights
from cauchy_generator.rng import SEED32_MAX, SEED32_MIN

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

ShiftProfile = Literal[
    "off",
    "graph_drift",
    "mechanism_drift",
    "noise_drift",
    "mixed",
    "custom",
]
SHIFT_PROFILE_OFF: Literal["off"] = "off"
SHIFT_PROFILE_GRAPH_DRIFT: Literal["graph_drift"] = "graph_drift"
SHIFT_PROFILE_MECHANISM_DRIFT: Literal["mechanism_drift"] = "mechanism_drift"
SHIFT_PROFILE_NOISE_DRIFT: Literal["noise_drift"] = "noise_drift"
SHIFT_PROFILE_MIXED: Literal["mixed"] = "mixed"
SHIFT_PROFILE_CUSTOM: Literal["custom"] = "custom"

_SHIFT_PROFILE_VALUE_MAP: dict[str, ShiftProfile] = {
    SHIFT_PROFILE_OFF: SHIFT_PROFILE_OFF,
    SHIFT_PROFILE_GRAPH_DRIFT: SHIFT_PROFILE_GRAPH_DRIFT,
    SHIFT_PROFILE_MECHANISM_DRIFT: SHIFT_PROFILE_MECHANISM_DRIFT,
    SHIFT_PROFILE_NOISE_DRIFT: SHIFT_PROFILE_NOISE_DRIFT,
    SHIFT_PROFILE_MIXED: SHIFT_PROFILE_MIXED,
    SHIFT_PROFILE_CUSTOM: SHIFT_PROFILE_CUSTOM,
}

NoiseFamily = Literal["legacy", "gaussian", "laplace", "student_t", "mixture"]
NOISE_FAMILY_LEGACY: Literal["legacy"] = "legacy"
NOISE_FAMILY_GAUSSIAN: Literal["gaussian"] = "gaussian"
NOISE_FAMILY_LAPLACE: Literal["laplace"] = "laplace"
NOISE_FAMILY_STUDENT_T: Literal["student_t"] = "student_t"
NOISE_FAMILY_MIXTURE: Literal["mixture"] = "mixture"

_NOISE_FAMILY_VALUE_MAP: dict[str, NoiseFamily] = {
    NOISE_FAMILY_LEGACY: NOISE_FAMILY_LEGACY,
    NOISE_FAMILY_GAUSSIAN: NOISE_FAMILY_GAUSSIAN,
    NOISE_FAMILY_LAPLACE: NOISE_FAMILY_LAPLACE,
    NOISE_FAMILY_STUDENT_T: NOISE_FAMILY_STUDENT_T,
    NOISE_FAMILY_MIXTURE: NOISE_FAMILY_MIXTURE,
}

NoiseMixtureComponent = Literal["gaussian", "laplace", "student_t"]
NOISE_MIXTURE_COMPONENT_GAUSSIAN: Literal["gaussian"] = "gaussian"
NOISE_MIXTURE_COMPONENT_LAPLACE: Literal["laplace"] = "laplace"
NOISE_MIXTURE_COMPONENT_STUDENT_T: Literal["student_t"] = "student_t"

_NOISE_MIXTURE_COMPONENT_VALUE_MAP: dict[str, NoiseMixtureComponent] = {
    NOISE_MIXTURE_COMPONENT_GAUSSIAN: NOISE_MIXTURE_COMPONENT_GAUSSIAN,
    NOISE_MIXTURE_COMPONENT_LAPLACE: NOISE_MIXTURE_COMPONENT_LAPLACE,
    NOISE_MIXTURE_COMPONENT_STUDENT_T: NOISE_MIXTURE_COMPONENT_STUDENT_T,
}

MAX_SUPPORTED_CLASS_COUNT = 32


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


def normalize_shift_profile(value: object) -> ShiftProfile:
    """Normalize shift profile into a validated internal value."""

    if value is False:
        return SHIFT_PROFILE_OFF
    if isinstance(value, bool) or not isinstance(value, str):
        raise ValueError(
            "Unsupported shift.profile "
            f"'{value}'. Expected off, graph_drift, mechanism_drift, noise_drift, mixed, or custom."
        )
    normalized = value.strip().lower()
    result = _SHIFT_PROFILE_VALUE_MAP.get(normalized)
    if result is None:
        raise ValueError(
            "Unsupported shift.profile "
            f"'{value}'. Expected off, graph_drift, mechanism_drift, noise_drift, mixed, or custom."
        )
    return result


def normalize_noise_family(value: object) -> NoiseFamily:
    """Normalize noise family into a validated internal value."""

    if isinstance(value, bool) or not isinstance(value, str):
        raise ValueError(
            "Unsupported noise.family "
            f"'{value}'. Expected legacy, gaussian, laplace, student_t, or mixture."
        )
    normalized = value.strip().lower()
    result = _NOISE_FAMILY_VALUE_MAP.get(normalized)
    if result is None:
        raise ValueError(
            "Unsupported noise.family "
            f"'{value}'. Expected legacy, gaussian, laplace, student_t, or mixture."
        )
    return result


def _normalize_noise_mixture_weights(
    value: object | None,
) -> dict[NoiseMixtureComponent, float] | None:
    """Normalize optional noise-mixture weights mapping."""

    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("noise.mixture_weights must be a mapping.")
    if not value:
        raise ValueError(
            "noise.mixture_weights must include at least one of gaussian, laplace, or student_t."
        )

    weights: dict[NoiseMixtureComponent, float] = {}
    for raw_key, raw_weight in value.items():
        if isinstance(raw_key, bool) or not isinstance(raw_key, str):
            raise ValueError(
                "noise.mixture_weights keys must be gaussian, laplace, or student_t strings."
            )
        normalized_key = raw_key.strip().lower()
        component = _NOISE_MIXTURE_COMPONENT_VALUE_MAP.get(normalized_key)
        if component is None:
            raise ValueError(
                "Unsupported noise.mixture_weights key "
                f"'{raw_key}'. Expected gaussian, laplace, or student_t."
            )
        if component in weights:
            raise ValueError(
                f"Duplicate noise.mixture_weights key '{raw_key}' after normalization."
            )
        weight = _validate_finite_float_field(
            field_name=f"noise.mixture_weights.{component}",
            value=raw_weight,
            lo=0.0,
            hi=None,
            lo_inclusive=True,
            hi_inclusive=False,
            expectation="a finite value >= 0",
        )
        weights[component] = float(weight)

    return normalize_positive_weights(weights, field_name="noise.mixture_weights")


def _validate_finite_float_field(
    *,
    field_name: str,
    value: Any,
    lo: float,
    hi: float | None,
    lo_inclusive: bool,
    hi_inclusive: bool,
    expectation: str,
) -> float:
    """Validate a float field against finite bounds and normalize it."""

    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be {expectation}, got {value!r}.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be {expectation}, got {value!r}.") from exc
    lo_ok = parsed >= lo if lo_inclusive else parsed > lo
    hi_ok = True
    if hi is not None:
        hi_ok = parsed <= hi if hi_inclusive else parsed < hi
    if not math.isfinite(parsed) or not (lo_ok and hi_ok):
        raise ValueError(f"{field_name} must be {expectation}, got {parsed!r}.")
    return parsed


def _validate_optional_finite_float_field(
    *,
    field_name: str,
    value: Any,
    lo: float,
    hi: float | None,
    lo_inclusive: bool,
    hi_inclusive: bool,
    expectation: str,
) -> float | None:
    """Validate an optional float field against finite bounds and normalize it."""

    if value is None:
        return None
    return _validate_finite_float_field(
        field_name=field_name,
        value=value,
        lo=lo,
        hi=hi,
        lo_inclusive=lo_inclusive,
        hi_inclusive=hi_inclusive,
        expectation=expectation,
    )


def _validate_int_field(
    *,
    field_name: str,
    value: Any,
    minimum: int,
    maximum: int | None = None,
) -> int:
    """Validate and normalize integer fields with optional inclusive bounds."""

    if isinstance(value, bool):
        expectation = (
            f"an integer in [{minimum}, {maximum}]"
            if maximum is not None
            else f"an integer >= {minimum}"
        )
        raise ValueError(f"{field_name} must be {expectation}, got {value!r}.")

    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        normalized = value.strip()
        signless = normalized[1:] if normalized.startswith(("+", "-")) else normalized
        if not signless.isdigit():
            expectation = (
                f"an integer in [{minimum}, {maximum}]"
                if maximum is not None
                else f"an integer >= {minimum}"
            )
            raise ValueError(f"{field_name} must be {expectation}, got {value!r}.")
        parsed = int(normalized)
    else:
        expectation = (
            f"an integer in [{minimum}, {maximum}]"
            if maximum is not None
            else f"an integer >= {minimum}"
        )
        raise ValueError(f"{field_name} must be {expectation}, got {value!r}.")

    if parsed < minimum or (maximum is not None and parsed > maximum):
        expectation = (
            f"an integer in [{minimum}, {maximum}]"
            if maximum is not None
            else f"an integer >= {minimum}"
        )
        raise ValueError(f"{field_name} must be {expectation}, got {parsed!r}.")
    return parsed


def validate_class_split_feasibility(
    *,
    n_classes: int,
    n_train: int,
    n_test: int,
    context: str,
) -> None:
    """Validate whether split sizes can represent all classes in both train and test."""

    if n_classes > n_train or n_classes > n_test:
        raise ValueError(
            f"{context}: infeasible class/split combination for classification "
            f"(n_classes={n_classes}, n_train={n_train}, n_test={n_test}). "
            "Require n_train and n_test to each be >= n_classes."
        )


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
        self.n_train = _validate_int_field(
            field_name="dataset.n_train",
            value=self.n_train,
            minimum=1,
        )
        self.n_test = _validate_int_field(
            field_name="dataset.n_test",
            value=self.n_test,
            minimum=1,
        )
        self.n_classes_min = _validate_int_field(
            field_name="dataset.n_classes_min",
            value=self.n_classes_min,
            minimum=2,
            maximum=MAX_SUPPORTED_CLASS_COUNT,
        )
        self.n_classes_max = _validate_int_field(
            field_name="dataset.n_classes_max",
            value=self.n_classes_max,
            minimum=2,
            maximum=MAX_SUPPORTED_CLASS_COUNT,
        )
        _validate_min_max_pair(
            name="dataset.n_classes_min",
            min_value=self.n_classes_min,
            max_value=self.n_classes_max,
            max_label="n_classes_max",
        )
        if str(self.task).strip().lower() == "classification":
            validate_class_split_feasibility(
                n_classes=int(self.n_classes_min),
                n_train=int(self.n_train),
                n_test=int(self.n_test),
                context="dataset classification split constraints for n_classes_min",
            )

        self.missing_rate = _validate_finite_float_field(
            field_name="dataset.missing_rate",
            value=self.missing_rate,
            lo=0.0,
            hi=1.0,
            lo_inclusive=True,
            hi_inclusive=True,
            expectation="a finite value in [0, 1]",
        )

        self.missing_mechanism = normalize_missing_mechanism(self.missing_mechanism)
        if self.missing_rate > 0.0 and self.missing_mechanism == MISSINGNESS_MECHANISM_NONE:
            raise ValueError(
                "dataset.missing_mechanism must be mcar, mar, or mnar when dataset.missing_rate > 0."
            )

        self.missing_mar_observed_fraction = _validate_finite_float_field(
            field_name="dataset.missing_mar_observed_fraction",
            value=self.missing_mar_observed_fraction,
            lo=0.0,
            hi=1.0,
            lo_inclusive=False,
            hi_inclusive=True,
            expectation="in (0, 1]",
        )
        self.missing_mar_logit_scale = _validate_finite_float_field(
            field_name="dataset.missing_mar_logit_scale",
            value=self.missing_mar_logit_scale,
            lo=0.0,
            hi=None,
            lo_inclusive=False,
            hi_inclusive=False,
            expectation="a finite value > 0",
        )
        self.missing_mnar_logit_scale = _validate_finite_float_field(
            field_name="dataset.missing_mnar_logit_scale",
            value=self.missing_mnar_logit_scale,
            lo=0.0,
            hi=None,
            lo_inclusive=False,
            hi_inclusive=False,
            expectation="a finite value > 0",
        )


@dataclass(slots=True)
class GraphConfig:
    n_nodes_min: int = 2
    n_nodes_max: int = 32


def _validate_min_max_pair(
    *,
    name: str,
    min_value: int | None,
    max_value: int | None,
    max_label: str,
) -> None:
    """Validate that optional min/max values are ordered when both are provided."""

    if min_value is None or max_value is None:
        return
    if min_value > max_value:
        raise ValueError(f"{name} must be <= {max_label}, got {min_value} > {max_value}.")


@dataclass(slots=True)
class ShiftConfig:
    enabled: bool = False
    profile: ShiftProfile = SHIFT_PROFILE_OFF
    graph_scale: float | None = None
    mechanism_scale: float | None = None
    noise_scale: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError(f"shift.enabled must be a boolean, got {self.enabled!r}.")
        self.profile = normalize_shift_profile(self.profile)
        self.graph_scale = _validate_optional_finite_float_field(
            field_name="shift.graph_scale",
            value=self.graph_scale,
            lo=0.0,
            hi=1.0,
            lo_inclusive=True,
            hi_inclusive=True,
            expectation="a finite value in [0, 1]",
        )
        self.mechanism_scale = _validate_optional_finite_float_field(
            field_name="shift.mechanism_scale",
            value=self.mechanism_scale,
            lo=0.0,
            hi=1.0,
            lo_inclusive=True,
            hi_inclusive=True,
            expectation="a finite value in [0, 1]",
        )
        self.noise_scale = _validate_optional_finite_float_field(
            field_name="shift.noise_scale",
            value=self.noise_scale,
            lo=0.0,
            hi=1.0,
            lo_inclusive=True,
            hi_inclusive=True,
            expectation="a finite value in [0, 1]",
        )

        has_overrides = any(
            scale is not None
            for scale in (self.graph_scale, self.mechanism_scale, self.noise_scale)
        )
        if not self.enabled:
            if self.profile != SHIFT_PROFILE_OFF:
                raise ValueError("shift.profile must be 'off' when shift.enabled is false.")
            if has_overrides:
                raise ValueError("shift override scales must be unset when shift.enabled is false.")
            return

        if self.profile == SHIFT_PROFILE_OFF:
            raise ValueError("shift.profile must not be 'off' when shift.enabled is true.")

        if self.profile == SHIFT_PROFILE_CUSTOM and not has_overrides:
            raise ValueError("shift.profile 'custom' requires at least one override scale.")

        if self.profile == SHIFT_PROFILE_GRAPH_DRIFT and (
            self.mechanism_scale is not None or self.noise_scale is not None
        ):
            raise ValueError("shift.profile 'graph_drift' only allows shift.graph_scale override.")
        if self.profile == SHIFT_PROFILE_MECHANISM_DRIFT and (
            self.graph_scale is not None or self.noise_scale is not None
        ):
            raise ValueError(
                "shift.profile 'mechanism_drift' only allows shift.mechanism_scale override."
            )
        if self.profile == SHIFT_PROFILE_NOISE_DRIFT and (
            self.graph_scale is not None or self.mechanism_scale is not None
        ):
            raise ValueError("shift.profile 'noise_drift' only allows shift.noise_scale override.")


@dataclass(slots=True)
class NoiseConfig:
    family: NoiseFamily = NOISE_FAMILY_LEGACY
    scale: float = 1.0
    student_t_df: float = 5.0
    mixture_weights: dict[NoiseMixtureComponent, float] | None = None

    def __post_init__(self) -> None:
        self.family = normalize_noise_family(self.family)
        self.scale = _validate_finite_float_field(
            field_name="noise.scale",
            value=self.scale,
            lo=0.0,
            hi=None,
            lo_inclusive=False,
            hi_inclusive=False,
            expectation="a finite value > 0",
        )
        self.student_t_df = _validate_finite_float_field(
            field_name="noise.student_t_df",
            value=self.student_t_df,
            lo=2.0,
            hi=None,
            lo_inclusive=False,
            hi_inclusive=False,
            expectation="a finite value > 2",
        )
        self.mixture_weights = _normalize_noise_mixture_weights(self.mixture_weights)
        if self.family != NOISE_FAMILY_MIXTURE and self.mixture_weights is not None:
            raise ValueError(
                "noise.mixture_weights is only allowed when noise.family is 'mixture'."
            )


@dataclass(slots=True)
class RuntimeConfig:
    device: str = "auto"
    torch_dtype: str = "float32"
    hardware_aware: bool = True


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
            "cuda_desktop": {
                "num_datasets": 2000,
                "warmup_datasets": 25,
                "device": "cuda",
            },
            "cuda_h100": {
                "num_datasets": 5000,
                "warmup_datasets": 50,
                "device": "cuda",
            },
        }
    )


@dataclass(slots=True)
class FilterConfig:
    enabled: bool = False
    n_estimators: int = 25
    max_depth: int = 6
    min_samples_leaf: int = 1
    max_leaf_nodes: int | None = None
    max_features: str | int | float = "auto"
    n_bootstrap: int = 200
    threshold: float = 0.95
    max_attempts: int = 3
    n_jobs: int = -1

    def __post_init__(self) -> None:
        self.n_jobs = _validate_int_field(
            field_name="filter.n_jobs",
            value=self.n_jobs,
            minimum=-1,
        )
        if self.n_jobs == 0:
            raise ValueError("filter.n_jobs must be -1 or an integer >= 1, got 0.")


@dataclass(slots=True)
class GeneratorConfig:
    seed: int = 1
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    shift: ShiftConfig = field(default_factory=ShiftConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)

    def __post_init__(self) -> None:
        self.seed = _validate_int_field(
            field_name="seed",
            value=self.seed,
            minimum=SEED32_MIN,
            maximum=SEED32_MAX,
        )

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
        shift = ShiftConfig(**(data.get("shift") or {}))
        noise = NoiseConfig(**(data.get("noise") or {}))
        runtime = RuntimeConfig(**(data.get("runtime") or {}))
        output = OutputConfig(**(data.get("output") or {}))
        diagnostics = DiagnosticsConfig(**(data.get("diagnostics") or {}))
        benchmark = BenchmarkConfig(**(data.get("benchmark") or {}))
        filter_cfg = FilterConfig(**(data.get("filter") or {}))
        seed = data.get("seed", 1)
        return cls(
            seed=seed,
            dataset=dataset,
            graph=graph,
            shift=shift,
            noise=noise,
            runtime=runtime,
            output=output,
            diagnostics=diagnostics,
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
