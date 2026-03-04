"""Typed configuration models and file loading."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml

from dagzoo.math_utils import normalize_positive_weights
from dagzoo.rng import SEED32_MAX, SEED32_MIN

MechanismFamily = Literal[
    "nn",
    "tree",
    "discretization",
    "gp",
    "linear",
    "quadratic",
    "em",
    "product",
]

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

ShiftMode = Literal[
    "off",
    "graph_drift",
    "mechanism_drift",
    "noise_drift",
    "mixed",
    "custom",
]
SHIFT_MODE_OFF: Literal["off"] = "off"
SHIFT_MODE_GRAPH_DRIFT: Literal["graph_drift"] = "graph_drift"
SHIFT_MODE_MECHANISM_DRIFT: Literal["mechanism_drift"] = "mechanism_drift"
SHIFT_MODE_NOISE_DRIFT: Literal["noise_drift"] = "noise_drift"
SHIFT_MODE_MIXED: Literal["mixed"] = "mixed"
SHIFT_MODE_CUSTOM: Literal["custom"] = "custom"

_SHIFT_MODE_VALUE_MAP: dict[str, ShiftMode] = {
    SHIFT_MODE_OFF: SHIFT_MODE_OFF,
    SHIFT_MODE_GRAPH_DRIFT: SHIFT_MODE_GRAPH_DRIFT,
    SHIFT_MODE_MECHANISM_DRIFT: SHIFT_MODE_MECHANISM_DRIFT,
    SHIFT_MODE_NOISE_DRIFT: SHIFT_MODE_NOISE_DRIFT,
    SHIFT_MODE_MIXED: SHIFT_MODE_MIXED,
    SHIFT_MODE_CUSTOM: SHIFT_MODE_CUSTOM,
}

NoiseFamily = Literal["gaussian", "laplace", "student_t", "mixture"]
NOISE_FAMILY_GAUSSIAN: Literal["gaussian"] = "gaussian"
NOISE_FAMILY_LAPLACE: Literal["laplace"] = "laplace"
NOISE_FAMILY_STUDENT_T: Literal["student_t"] = "student_t"
NOISE_FAMILY_MIXTURE: Literal["mixture"] = "mixture"

_NOISE_FAMILY_VALUE_MAP: dict[str, NoiseFamily] = {
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

_MECHANISM_FAMILY_VALUE_MAP: dict[str, MechanismFamily] = {
    "nn": "nn",
    "tree": "tree",
    "discretization": "discretization",
    "gp": "gp",
    "linear": "linear",
    "quadratic": "quadratic",
    "em": "em",
    "product": "product",
}
_PRODUCT_COMPONENT_FAMILIES: frozenset[MechanismFamily] = frozenset(
    {"tree", "discretization", "gp", "linear", "quadratic"}
)

MAX_SUPPORTED_CLASS_COUNT = 32
_SectionT = TypeVar("_SectionT")


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


def normalize_shift_mode(value: object) -> ShiftMode:
    """Normalize shift mode into a validated internal value."""

    if isinstance(value, bool) or not isinstance(value, str):
        raise ValueError(
            "Unsupported shift.mode "
            f"'{value}'. Expected off, graph_drift, mechanism_drift, noise_drift, mixed, or custom."
        )
    normalized = value.strip().lower()
    result = _SHIFT_MODE_VALUE_MAP.get(normalized)
    if result is None:
        raise ValueError(
            "Unsupported shift.mode "
            f"'{value}'. Expected off, graph_drift, mechanism_drift, noise_drift, mixed, or custom."
        )
    return result


def normalize_noise_family(value: object) -> NoiseFamily:
    """Normalize noise family into a validated internal value."""

    if isinstance(value, bool) or not isinstance(value, str):
        raise ValueError(
            "Unsupported noise.family "
            f"'{value}'. Expected gaussian, laplace, student_t, or mixture."
        )
    normalized = value.strip().lower()
    result = _NOISE_FAMILY_VALUE_MAP.get(normalized)
    if result is None:
        raise ValueError(
            "Unsupported noise.family "
            f"'{value}'. Expected gaussian, laplace, student_t, or mixture."
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


def _normalize_function_family_mix(
    value: object | None,
) -> dict[MechanismFamily, float] | None:
    """Normalize optional mechanism family-mix weights mapping."""

    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("mechanism.function_family_mix must be a mapping.")
    if not value:
        raise ValueError(
            "mechanism.function_family_mix must include at least one supported family."
        )

    weights: dict[MechanismFamily, float] = {}
    for raw_key, raw_weight in value.items():
        if isinstance(raw_key, bool) or not isinstance(raw_key, str):
            raise ValueError("mechanism.function_family_mix keys must be mechanism family names.")
        normalized_key = raw_key.strip().lower()
        family = _MECHANISM_FAMILY_VALUE_MAP.get(normalized_key)
        if family is None:
            supported = ", ".join(sorted(_MECHANISM_FAMILY_VALUE_MAP))
            raise ValueError(
                "Unsupported mechanism.function_family_mix key "
                f"'{raw_key}'. Expected one of: {supported}."
            )
        if family in weights:
            raise ValueError(
                f"Duplicate mechanism.function_family_mix key '{raw_key}' after normalization."
            )
        weight = _validate_finite_float_field(
            field_name=f"mechanism.function_family_mix.{family}",
            value=raw_weight,
            lo=0.0,
            hi=None,
            lo_inclusive=True,
            hi_inclusive=False,
            expectation="a finite value >= 0",
        )
        weights[family] = float(weight)
    return normalize_positive_weights(
        weights,
        field_name="mechanism.function_family_mix",
    )


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


def _normalize_dataset_fields(dataset: DatasetConfig) -> None:
    """Stage 1: normalize individual dataset fields and scalar bounds."""

    normalized_task = str(dataset.task).strip().lower()
    if normalized_task not in {"classification", "regression"}:
        raise ValueError(
            f"dataset.task must be 'classification' or 'regression', got {dataset.task!r}."
        )
    dataset.task = normalized_task

    dataset.n_train = _validate_int_field(
        field_name="dataset.n_train",
        value=dataset.n_train,
        minimum=1,
    )
    dataset.n_test = _validate_int_field(
        field_name="dataset.n_test",
        value=dataset.n_test,
        minimum=1,
    )
    dataset.n_features_min = _validate_int_field(
        field_name="dataset.n_features_min",
        value=dataset.n_features_min,
        minimum=1,
    )
    dataset.n_features_max = _validate_int_field(
        field_name="dataset.n_features_max",
        value=dataset.n_features_max,
        minimum=1,
    )
    dataset.max_categorical_cardinality = _validate_int_field(
        field_name="dataset.max_categorical_cardinality",
        value=dataset.max_categorical_cardinality,
        minimum=2,
    )

    dataset.categorical_ratio_min = _validate_finite_float_field(
        field_name="dataset.categorical_ratio_min",
        value=dataset.categorical_ratio_min,
        lo=0.0,
        hi=1.0,
        lo_inclusive=True,
        hi_inclusive=True,
        expectation="a finite value in [0, 1]",
    )
    dataset.categorical_ratio_max = _validate_finite_float_field(
        field_name="dataset.categorical_ratio_max",
        value=dataset.categorical_ratio_max,
        lo=0.0,
        hi=1.0,
        lo_inclusive=True,
        hi_inclusive=True,
        expectation="a finite value in [0, 1]",
    )

    dataset.n_classes_min = _validate_int_field(
        field_name="dataset.n_classes_min",
        value=dataset.n_classes_min,
        minimum=2,
        maximum=MAX_SUPPORTED_CLASS_COUNT,
    )
    dataset.n_classes_max = _validate_int_field(
        field_name="dataset.n_classes_max",
        value=dataset.n_classes_max,
        minimum=2,
        maximum=MAX_SUPPORTED_CLASS_COUNT,
    )

    dataset.missing_rate = _validate_finite_float_field(
        field_name="dataset.missing_rate",
        value=dataset.missing_rate,
        lo=0.0,
        hi=1.0,
        lo_inclusive=True,
        hi_inclusive=True,
        expectation="a finite value in [0, 1]",
    )
    dataset.missing_mechanism = normalize_missing_mechanism(dataset.missing_mechanism)
    dataset.missing_mar_observed_fraction = _validate_finite_float_field(
        field_name="dataset.missing_mar_observed_fraction",
        value=dataset.missing_mar_observed_fraction,
        lo=0.0,
        hi=1.0,
        lo_inclusive=False,
        hi_inclusive=True,
        expectation="in (0, 1]",
    )
    dataset.missing_mar_logit_scale = _validate_finite_float_field(
        field_name="dataset.missing_mar_logit_scale",
        value=dataset.missing_mar_logit_scale,
        lo=0.0,
        hi=None,
        lo_inclusive=False,
        hi_inclusive=False,
        expectation="a finite value > 0",
    )
    dataset.missing_mnar_logit_scale = _validate_finite_float_field(
        field_name="dataset.missing_mnar_logit_scale",
        value=dataset.missing_mnar_logit_scale,
        lo=0.0,
        hi=None,
        lo_inclusive=False,
        hi_inclusive=False,
        expectation="a finite value > 0",
    )


def _normalize_graph_fields(graph: GraphConfig) -> None:
    """Stage 1: normalize graph scalar fields."""

    graph.n_nodes_min = _validate_int_field(
        field_name="graph.n_nodes_min",
        value=graph.n_nodes_min,
        minimum=2,
    )
    graph.n_nodes_max = _validate_int_field(
        field_name="graph.n_nodes_max",
        value=graph.n_nodes_max,
        minimum=2,
    )


def _normalize_mechanism_fields(mechanism: MechanismConfig) -> None:
    """Stage 1: normalize mechanism section fields."""

    mechanism.function_family_mix = _normalize_function_family_mix(mechanism.function_family_mix)


def _normalize_shift_fields(shift: ShiftConfig) -> None:
    """Stage 1: normalize shift fields."""

    if not isinstance(shift.enabled, bool):
        raise ValueError(f"shift.enabled must be a boolean, got {shift.enabled!r}.")
    shift.mode = normalize_shift_mode(shift.mode)
    shift.graph_scale = _validate_optional_finite_float_field(
        field_name="shift.graph_scale",
        value=shift.graph_scale,
        lo=0.0,
        hi=1.0,
        lo_inclusive=True,
        hi_inclusive=True,
        expectation="a finite value in [0, 1]",
    )
    shift.mechanism_scale = _validate_optional_finite_float_field(
        field_name="shift.mechanism_scale",
        value=shift.mechanism_scale,
        lo=0.0,
        hi=1.0,
        lo_inclusive=True,
        hi_inclusive=True,
        expectation="a finite value in [0, 1]",
    )
    shift.variance_scale = _validate_optional_finite_float_field(
        field_name="shift.variance_scale",
        value=shift.variance_scale,
        lo=0.0,
        hi=1.0,
        lo_inclusive=True,
        hi_inclusive=True,
        expectation="a finite value in [0, 1]",
    )


def _normalize_noise_fields(noise: NoiseConfig) -> None:
    """Stage 1: normalize noise family and scalar fields."""

    noise.family = normalize_noise_family(noise.family)
    noise.base_scale = _validate_finite_float_field(
        field_name="noise.base_scale",
        value=noise.base_scale,
        lo=0.0,
        hi=None,
        lo_inclusive=False,
        hi_inclusive=False,
        expectation="a finite value > 0",
    )
    noise.student_t_df = _validate_finite_float_field(
        field_name="noise.student_t_df",
        value=noise.student_t_df,
        lo=2.0,
        hi=None,
        lo_inclusive=False,
        hi_inclusive=False,
        expectation="a finite value > 2",
    )
    noise.mixture_weights = _normalize_noise_mixture_weights(noise.mixture_weights)


def _normalize_runtime_fields(runtime: RuntimeConfig) -> None:
    """Stage 1: normalize runtime selector fields."""

    if runtime.device is None:
        runtime.device = "auto"
    elif isinstance(runtime.device, bool):
        raise ValueError(f"runtime.device must be a string or null, got {runtime.device!r}.")
    else:
        runtime.device = str(runtime.device).strip().lower() or "auto"

    if isinstance(runtime.torch_dtype, bool):
        raise ValueError(f"runtime.torch_dtype must be a string, got {runtime.torch_dtype!r}.")
    runtime.torch_dtype = str(runtime.torch_dtype).strip().lower()
    if not runtime.torch_dtype:
        raise ValueError("runtime.torch_dtype must be a non-empty string.")


def _normalize_output_fields(_output: OutputConfig) -> None:
    """Stage 1: output section has no additional field normalization."""


def _normalize_diagnostics_fields(_diagnostics: DiagnosticsConfig) -> None:
    """Stage 1: diagnostics section has no additional field normalization."""


def _normalize_benchmark_fields(_benchmark: BenchmarkConfig) -> None:
    """Stage 1: benchmark section has no additional field normalization."""


def _normalize_filter_fields(filter_cfg: FilterConfig) -> None:
    """Stage 1: normalize filter scalar fields."""

    filter_cfg.n_jobs = _validate_int_field(
        field_name="filter.n_jobs",
        value=filter_cfg.n_jobs,
        minimum=-1,
    )
    if filter_cfg.n_jobs == 0:
        raise ValueError("filter.n_jobs must be -1 or an integer >= 1, got 0.")


def _coerce_section(
    *,
    section_name: str,
    value: object,
    section_type: type[_SectionT],
) -> _SectionT:
    """Coerce one top-level section into its canonical dataclass type."""

    if isinstance(value, section_type):
        return value
    if isinstance(value, dict):
        return section_type(**value)
    if is_dataclass(value) and not isinstance(value, type):
        payload = asdict(value)
        if isinstance(payload, dict):
            return section_type(**payload)
    raise TypeError(
        f"{section_name} must be a {section_type.__name__} or mapping, got {type(value).__name__}."
    )


def _stage1_normalize_generation_sections(config: GeneratorConfig) -> None:
    """Stage 1: field-level normalization/typing for all config sections."""

    config.dataset = _coerce_section(
        section_name="dataset",
        value=config.dataset,
        section_type=DatasetConfig,
    )
    config.graph = _coerce_section(
        section_name="graph",
        value=config.graph,
        section_type=GraphConfig,
    )
    config.mechanism = _coerce_section(
        section_name="mechanism",
        value=config.mechanism,
        section_type=MechanismConfig,
    )
    config.shift = _coerce_section(
        section_name="shift",
        value=config.shift,
        section_type=ShiftConfig,
    )
    config.noise = _coerce_section(
        section_name="noise",
        value=config.noise,
        section_type=NoiseConfig,
    )
    config.runtime = _coerce_section(
        section_name="runtime",
        value=config.runtime,
        section_type=RuntimeConfig,
    )
    config.output = _coerce_section(
        section_name="output",
        value=config.output,
        section_type=OutputConfig,
    )
    config.diagnostics = _coerce_section(
        section_name="diagnostics",
        value=config.diagnostics,
        section_type=DiagnosticsConfig,
    )
    config.benchmark = _coerce_section(
        section_name="benchmark",
        value=config.benchmark,
        section_type=BenchmarkConfig,
    )
    config.filter = _coerce_section(
        section_name="filter",
        value=config.filter,
        section_type=FilterConfig,
    )

    _normalize_dataset_fields(config.dataset)
    _normalize_graph_fields(config.graph)
    _normalize_mechanism_fields(config.mechanism)
    _normalize_shift_fields(config.shift)
    _normalize_noise_fields(config.noise)
    _normalize_runtime_fields(config.runtime)
    _normalize_output_fields(config.output)
    _normalize_diagnostics_fields(config.diagnostics)
    _normalize_benchmark_fields(config.benchmark)
    _normalize_filter_fields(config.filter)


def _stage2_validate_dataset_constraints(dataset: DatasetConfig) -> None:
    """Stage 2: validate dataset cross-field constraints."""

    _validate_min_max_pair(
        name="dataset.n_features_min",
        min_value=dataset.n_features_min,
        max_value=dataset.n_features_max,
        max_label="n_features_max",
    )
    if dataset.categorical_ratio_min > dataset.categorical_ratio_max:
        raise ValueError(
            "dataset.categorical_ratio_min must be <= categorical_ratio_max, "
            f"got {dataset.categorical_ratio_min} > {dataset.categorical_ratio_max}."
        )
    _validate_min_max_pair(
        name="dataset.n_classes_min",
        min_value=dataset.n_classes_min,
        max_value=dataset.n_classes_max,
        max_label="n_classes_max",
    )

    if dataset.task == "classification":
        validate_class_split_feasibility(
            n_classes=int(dataset.n_classes_min),
            n_train=int(dataset.n_train),
            n_test=int(dataset.n_test),
            context="dataset classification split constraints for n_classes_min",
        )
        validate_class_split_feasibility(
            n_classes=int(dataset.n_classes_max),
            n_train=int(dataset.n_train),
            n_test=int(dataset.n_test),
            context="dataset classification split constraints for n_classes_max",
        )

    if dataset.missing_rate > 0.0 and dataset.missing_mechanism == MISSINGNESS_MECHANISM_NONE:
        raise ValueError(
            "dataset.missing_mechanism must be mcar, mar, or mnar when dataset.missing_rate > 0."
        )


def _stage2_validate_graph_constraints(graph: GraphConfig) -> None:
    """Stage 2: validate graph cross-field constraints."""

    _validate_min_max_pair(
        name="graph.n_nodes_min",
        min_value=graph.n_nodes_min,
        max_value=graph.n_nodes_max,
        max_label="n_nodes_max",
    )


def _stage2_validate_mechanism_constraints(mechanism: MechanismConfig) -> None:
    """Stage 2: validate mechanism-family dependent relationships."""

    family_mix = mechanism.function_family_mix
    if family_mix is None:
        return
    if "product" not in family_mix:
        return
    has_product_component = any(family in family_mix for family in _PRODUCT_COMPONENT_FAMILIES)
    if not has_product_component:
        supported = ", ".join(sorted(_PRODUCT_COMPONENT_FAMILIES))
        raise ValueError(
            "mechanism.function_family_mix assigns positive weight to 'product' but none of its "
            f"component families are enabled. Add one of: {supported}."
        )


def _stage2_validate_shift_constraints(shift: ShiftConfig) -> None:
    """Stage 2: validate shift mode and override compatibility."""

    has_overrides = any(
        scale is not None
        for scale in (shift.graph_scale, shift.mechanism_scale, shift.variance_scale)
    )
    if not shift.enabled:
        if shift.mode != SHIFT_MODE_OFF:
            raise ValueError("shift.mode must be 'off' when shift.enabled is false.")
        if has_overrides:
            raise ValueError("shift override scales must be unset when shift.enabled is false.")
        return

    if shift.mode == SHIFT_MODE_OFF:
        raise ValueError("shift.mode must not be 'off' when shift.enabled is true.")

    if shift.mode == SHIFT_MODE_CUSTOM and not has_overrides:
        raise ValueError("shift.mode 'custom' requires at least one override scale.")

    if shift.mode == SHIFT_MODE_GRAPH_DRIFT and (
        shift.mechanism_scale is not None or shift.variance_scale is not None
    ):
        raise ValueError("shift.mode 'graph_drift' only allows shift.graph_scale override.")
    if shift.mode == SHIFT_MODE_MECHANISM_DRIFT and (
        shift.graph_scale is not None or shift.variance_scale is not None
    ):
        raise ValueError("shift.mode 'mechanism_drift' only allows shift.mechanism_scale override.")
    if shift.mode == SHIFT_MODE_NOISE_DRIFT and (
        shift.graph_scale is not None or shift.mechanism_scale is not None
    ):
        raise ValueError("shift.mode 'noise_drift' only allows shift.variance_scale override.")


def _stage2_validate_noise_constraints(noise: NoiseConfig) -> None:
    """Stage 2: validate noise-family dependent relationships."""

    if noise.family != NOISE_FAMILY_MIXTURE and noise.mixture_weights is not None:
        raise ValueError("noise.mixture_weights is only allowed when noise.family is 'mixture'.")


def _stage2_validate_generation_constraints(config: GeneratorConfig) -> None:
    """Stage 2: validate cross-field constraints after section normalization."""

    _stage2_validate_dataset_constraints(config.dataset)
    _stage2_validate_graph_constraints(config.graph)
    _stage2_validate_mechanism_constraints(config.mechanism)
    _stage2_validate_shift_constraints(config.shift)
    _stage2_validate_noise_constraints(config.noise)


def _run_generation_validation_stages(config: GeneratorConfig) -> None:
    """Run staged generation validation (stage1 normalize -> stage2 cross-field)."""

    _stage1_normalize_generation_sections(config)
    _stage2_validate_generation_constraints(config)


@dataclass(slots=True)
class DatasetConfig:
    task: str = "classification"
    n_train: int = 768
    n_test: int = 256
    n_features_min: int = 16
    n_features_max: int = 64
    n_classes_min: int = 2
    n_classes_max: int = 10
    categorical_ratio_min: float = 0.0
    categorical_ratio_max: float = 1.0
    max_categorical_cardinality: int = 9
    missing_rate: float = 0.0
    missing_mechanism: MissingnessMechanism = MISSINGNESS_MECHANISM_NONE
    missing_mar_observed_fraction: float = 0.5
    missing_mar_logit_scale: float = 1.0
    missing_mnar_logit_scale: float = 1.0

    def __post_init__(self) -> None:
        _normalize_dataset_fields(self)


@dataclass(slots=True)
class GraphConfig:
    n_nodes_min: int = 2
    n_nodes_max: int = 32

    def __post_init__(self) -> None:
        _normalize_graph_fields(self)


@dataclass(slots=True)
class MechanismConfig:
    function_family_mix: dict[MechanismFamily, float] | None = None

    def __post_init__(self) -> None:
        _normalize_mechanism_fields(self)


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
    mode: ShiftMode = SHIFT_MODE_OFF
    graph_scale: float | None = None
    mechanism_scale: float | None = None
    variance_scale: float | None = None

    def __post_init__(self) -> None:
        _normalize_shift_fields(self)


@dataclass(slots=True)
class NoiseConfig:
    family: NoiseFamily = NOISE_FAMILY_GAUSSIAN
    base_scale: float = 1.0
    student_t_df: float = 5.0
    mixture_weights: dict[NoiseMixtureComponent, float] | None = None

    def __post_init__(self) -> None:
        _normalize_noise_fields(self)


@dataclass(slots=True)
class RuntimeConfig:
    device: str = "auto"
    torch_dtype: str = "float32"


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
    preset_name: str = "medium_cuda"
    num_datasets: int = 2000
    warmup_datasets: int = 25
    suite: str = "standard"
    warn_threshold_pct: float = 10.0
    fail_threshold_pct: float = 20.0
    collect_memory: bool = True
    collect_reproducibility: bool = False
    reproducibility_num_datasets: int = 2
    latency_num_samples: int = 20
    presets: dict[str, dict[str, int | str]] = field(
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
        _normalize_filter_fields(self)


@dataclass(slots=True)
class GeneratorConfig:
    seed: int = 1
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    mechanism: MechanismConfig = field(default_factory=MechanismConfig)
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
        _run_generation_validation_stages(self)

    def validate_generation_constraints(self) -> None:
        """Stage 3: re-run staged validation after runtime/CLI overrides."""

        _run_generation_validation_stages(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> GeneratorConfig:
        """Construct `GeneratorConfig` from a nested dictionary payload."""

        data = data or {}
        dataset = DatasetConfig(**(data.get("dataset") or {}))
        graph = GraphConfig(**(data.get("graph") or {}))
        mechanism = MechanismConfig(**(data.get("mechanism") or {}))
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
            mechanism=mechanism,
            shift=shift,
            noise=noise,
            runtime=runtime,
            output=output,
            diagnostics=diagnostics,
            benchmark=benchmark,
            filter=filter_cfg,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> GeneratorConfig:
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
