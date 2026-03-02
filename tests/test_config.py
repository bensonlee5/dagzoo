import pytest

from cauchy_generator.config import (
    MAX_SUPPORTED_CLASS_COUNT,
    MISSINGNESS_MECHANISM_MAR,
    MISSINGNESS_MECHANISM_MCAR,
    MISSINGNESS_MECHANISM_MNAR,
    MISSINGNESS_MECHANISM_NONE,
    GeneratorConfig,
)
from cauchy_generator.io.lineage_schema import validate_metadata_lineage


def test_load_default_config() -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    assert cfg.dataset.n_train > 0
    assert cfg.dataset.n_features_min <= cfg.dataset.n_features_max
    assert cfg.output.shard_size > 0
    assert cfg.diagnostics.enabled is False
    assert cfg.diagnostics.histogram_bins > 0
    assert cfg.diagnostics.quantiles
    assert cfg.diagnostics.underrepresented_threshold >= 0
    assert (
        cfg.diagnostics.max_values_per_metric is None or cfg.diagnostics.max_values_per_metric > 0
    )
    assert cfg.diagnostics.meta_feature_targets == {}
    assert cfg.diagnostics.out_dir is None
    assert cfg.filter.n_estimators > 0
    assert cfg.filter.max_depth >= 0
    assert cfg.filter.max_features == "auto"
    assert cfg.filter.n_jobs == -1
    assert cfg.dataset.missing_rate == 0.0
    assert cfg.dataset.missing_mechanism == MISSINGNESS_MECHANISM_NONE
    assert cfg.dataset.missing_mar_observed_fraction == 0.5
    assert cfg.dataset.missing_mar_logit_scale == 1.0
    assert cfg.dataset.missing_mnar_logit_scale == 1.0


def test_default_config_metadata_is_compatible_with_optional_lineage() -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    metadata = {
        "seed": int(cfg.seed),
        "config": cfg.to_dict(),
    }
    validate_metadata_lineage(metadata, required=False)


def test_seed_range_accepts_32bit_boundaries() -> None:
    cfg_min = GeneratorConfig.from_dict({"seed": 0})
    cfg_max = GeneratorConfig.from_dict({"seed": 4294967295})
    assert cfg_min.seed == 0
    assert cfg_max.seed == 4294967295


@pytest.mark.parametrize("bad_seed", [-1, 4294967296, True])
def test_seed_range_rejects_out_of_bounds_values(bad_seed: int | bool) -> None:
    with pytest.raises(ValueError, match=r"seed must be an integer in \[0, 4294967295\]"):
        GeneratorConfig.from_dict({"seed": bad_seed})


def test_class_range_accepts_many_class_envelope_limit() -> None:
    cfg = GeneratorConfig.from_dict(
        {
            "dataset": {
                "task": "classification",
                "n_classes_min": 12,
                "n_classes_max": MAX_SUPPORTED_CLASS_COUNT,
            }
        }
    )
    assert cfg.dataset.n_classes_min == 12
    assert cfg.dataset.n_classes_max == MAX_SUPPORTED_CLASS_COUNT


def test_class_range_rejects_values_above_many_class_envelope_limit() -> None:
    with pytest.raises(ValueError, match=r"dataset\.n_classes_max must be an integer in \[2, 32\]"):
        GeneratorConfig.from_dict({"dataset": {"task": "classification", "n_classes_max": 33}})


def test_class_range_rejects_min_greater_than_max() -> None:
    with pytest.raises(ValueError, match=r"dataset\.n_classes_min must be <= n_classes_max"):
        GeneratorConfig.from_dict(
            {"dataset": {"task": "classification", "n_classes_min": 16, "n_classes_max": 12}}
        )


def test_classification_rejects_impossible_class_split_bounds() -> None:
    with pytest.raises(ValueError, match=r"infeasible class/split combination"):
        GeneratorConfig.from_dict(
            {
                "dataset": {
                    "task": "classification",
                    "n_train": 24,
                    "n_test": 24,
                    "n_classes_min": 32,
                    "n_classes_max": 32,
                }
            }
        )


def test_classification_allows_partial_class_range_when_lower_bound_is_feasible() -> None:
    cfg = GeneratorConfig.from_dict(
        {
            "dataset": {
                "task": "classification",
                "n_train": 32,
                "n_test": 8,
                "n_classes_min": 2,
                "n_classes_max": 10,
            }
        }
    )
    assert cfg.dataset.n_classes_min == 2
    assert cfg.dataset.n_classes_max == 10


def test_regression_allows_class_fields_without_split_feasibility_check() -> None:
    cfg = GeneratorConfig.from_dict(
        {
            "dataset": {
                "task": "regression",
                "n_train": 24,
                "n_test": 24,
                "n_classes_min": 2,
                "n_classes_max": 32,
            }
        }
    )
    assert cfg.dataset.n_classes_max == 32


def test_load_cuda_presets() -> None:
    cfg_h100 = GeneratorConfig.from_yaml("configs/preset_cuda_h100.yaml")
    assert cfg_h100.runtime.device == "cuda"
    assert cfg_h100.dataset.n_features_max >= 128


def test_load_diagnostics_preset() -> None:
    cfg_diag = GeneratorConfig.from_yaml("configs/preset_diagnostics_on.yaml")
    assert cfg_diag.diagnostics.enabled is True
    assert cfg_diag.diagnostics.histogram_bins >= 8
    assert cfg_diag.diagnostics.quantiles


def test_load_missingness_presets() -> None:
    cfg_mcar = GeneratorConfig.from_yaml("configs/preset_missingness_mcar.yaml")
    assert cfg_mcar.dataset.missing_mechanism == MISSINGNESS_MECHANISM_MCAR
    assert cfg_mcar.dataset.missing_rate > 0.0

    cfg_mar = GeneratorConfig.from_yaml("configs/preset_missingness_mar.yaml")
    assert cfg_mar.dataset.missing_mechanism == MISSINGNESS_MECHANISM_MAR
    assert cfg_mar.dataset.missing_rate > 0.0
    assert cfg_mar.dataset.missing_mar_observed_fraction > 0.0

    cfg_mnar = GeneratorConfig.from_yaml("configs/preset_missingness_mnar.yaml")
    assert cfg_mnar.dataset.missing_mechanism == MISSINGNESS_MECHANISM_MNAR
    assert cfg_mnar.dataset.missing_rate > 0.0
    assert cfg_mnar.dataset.missing_mnar_logit_scale > 0.0


def test_load_benchmark_profiles() -> None:
    cfg_cpu = GeneratorConfig.from_yaml("configs/benchmark_cpu.yaml")
    cfg_desktop = GeneratorConfig.from_yaml("configs/benchmark_cuda_desktop.yaml")
    cfg_h100 = GeneratorConfig.from_yaml("configs/benchmark_cuda_h100.yaml")

    assert cfg_cpu.runtime.device == "cpu"
    assert cfg_desktop.runtime.device == "cuda"
    assert cfg_h100.runtime.device == "cuda"
    assert "cpu" in cfg_h100.benchmark.profiles


def test_load_lineage_benchmark_smoke_preset() -> None:
    cfg = GeneratorConfig.from_yaml("configs/preset_lineage_benchmark_smoke.yaml")
    assert cfg.benchmark.suite == "smoke"
    assert cfg.benchmark.profile_name == "lineage_smoke"
    assert "lineage_smoke" in cfg.benchmark.profiles
    assert cfg.benchmark.latency_num_samples >= 5


def test_runtime_config_from_dict() -> None:
    cfg = GeneratorConfig.from_dict(
        {
            "runtime": {
                "device": "cpu",
                "torch_dtype": "float64",
            },
            "diagnostics": {
                "enabled": True,
                "histogram_bins": 12,
                "max_values_per_metric": 1234,
                "meta_feature_targets": {
                    "linearity_proxy": [0.2, 0.8],
                },
            },
        }
    )
    assert cfg.runtime.device == "cpu"
    assert cfg.runtime.torch_dtype == "float64"
    assert cfg.diagnostics.enabled is True
    assert cfg.diagnostics.histogram_bins == 12
    assert cfg.diagnostics.max_values_per_metric == 1234
    assert "linearity_proxy" in cfg.diagnostics.meta_feature_targets


def test_runtime_config_rejects_generation_engine_key() -> None:
    with pytest.raises(TypeError, match="generation_engine"):
        GeneratorConfig.from_dict({"runtime": {"generation_engine": "appendix_light"}})


def test_legacy_filter_keys_are_rejected() -> None:
    with pytest.raises(TypeError, match="n_trees"):
        GeneratorConfig.from_dict(
            {
                "filter": {
                    "enabled": True,
                    "n_trees": 25,
                    "depth": 6,
                    "n_split_candidates": 8,
                }
            }
        )


def test_filter_n_jobs_accepts_minus_one_and_positive_values() -> None:
    cfg_auto = GeneratorConfig.from_dict({"filter": {"n_jobs": -1}})
    cfg_single = GeneratorConfig.from_dict({"filter": {"n_jobs": 1}})
    cfg_multi = GeneratorConfig.from_dict({"filter": {"n_jobs": 8}})

    assert cfg_auto.filter.n_jobs == -1
    assert cfg_single.filter.n_jobs == 1
    assert cfg_multi.filter.n_jobs == 8


@pytest.mark.parametrize("value", [0, -2, True])
def test_filter_n_jobs_rejects_invalid_values(value: int | bool) -> None:
    with pytest.raises(ValueError, match=r"filter\.n_jobs must"):
        GeneratorConfig.from_dict({"filter": {"n_jobs": value}})


def test_missingness_mechanism_normalization_is_case_insensitive() -> None:
    cfg = GeneratorConfig.from_dict(
        {"dataset": {"missing_rate": 0.25, "missing_mechanism": "MCAR"}}
    )
    assert cfg.dataset.missing_mechanism == MISSINGNESS_MECHANISM_MCAR


def test_missingness_rejects_none_mechanism_when_rate_is_positive() -> None:
    with pytest.raises(
        ValueError,
        match="dataset.missing_mechanism must be mcar, mar, or mnar when dataset.missing_rate > 0",
    ):
        GeneratorConfig.from_dict({"dataset": {"missing_rate": 0.1, "missing_mechanism": "none"}})


@pytest.mark.parametrize("value", [-0.1, 1.1, float("inf"), float("nan"), True])
def test_missing_rate_bounds_are_validated(value: float | bool) -> None:
    with pytest.raises(
        ValueError, match="dataset.missing_rate must be a finite value in \\[0, 1\\]"
    ):
        GeneratorConfig.from_dict({"dataset": {"missing_rate": value}})


@pytest.mark.parametrize("value", [0.0, -0.1, 1.1, float("inf"), float("nan"), True])
def test_missing_mar_observed_fraction_bounds_are_validated(value: float | bool) -> None:
    with pytest.raises(
        ValueError, match="dataset.missing_mar_observed_fraction must be in \\(0, 1\\]"
    ):
        GeneratorConfig.from_dict({"dataset": {"missing_mar_observed_fraction": value}})


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("missing_mar_logit_scale", 0.0),
        ("missing_mar_logit_scale", -1.0),
        ("missing_mar_logit_scale", float("inf")),
        ("missing_mar_logit_scale", float("nan")),
        ("missing_mar_logit_scale", True),
        ("missing_mnar_logit_scale", 0.0),
        ("missing_mnar_logit_scale", -1.0),
        ("missing_mnar_logit_scale", float("inf")),
        ("missing_mnar_logit_scale", float("nan")),
        ("missing_mnar_logit_scale", True),
    ],
)
def test_missing_logit_scale_bounds_are_validated(field_name: str, bad_value: float | bool) -> None:
    with pytest.raises(ValueError, match=rf"dataset.{field_name} must be a finite value > 0"):
        GeneratorConfig.from_dict({"dataset": {field_name: bad_value}})


def test_invalid_missing_mechanism_string_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported missing_mechanism"):
        GeneratorConfig.from_dict({"dataset": {"missing_mechanism": "garbage"}})


def test_unused_missingness_parameters_are_allowed_with_disabled_mechanism() -> None:
    cfg = GeneratorConfig.from_dict(
        {
            "dataset": {
                "missing_rate": 0.0,
                "missing_mechanism": "none",
                "missing_mar_observed_fraction": 0.8,
                "missing_mar_logit_scale": 2.5,
                "missing_mnar_logit_scale": 3.5,
            }
        }
    )
    assert cfg.dataset.missing_mechanism == MISSINGNESS_MECHANISM_NONE
    assert cfg.dataset.missing_mar_observed_fraction == 0.8
    assert cfg.dataset.missing_mar_logit_scale == 2.5
    assert cfg.dataset.missing_mnar_logit_scale == 3.5


def test_shift_defaults_are_backward_compatible_when_shift_block_is_absent() -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    assert cfg.shift.enabled is False
    assert cfg.shift.profile == "off"
    assert cfg.shift.graph_scale is None
    assert cfg.shift.mechanism_scale is None
    assert cfg.shift.noise_scale is None


@pytest.mark.parametrize("config_path", ("configs/default.yaml", "configs/benchmark_cpu.yaml"))
def test_existing_config_files_parse_with_shift_schema(config_path: str) -> None:
    cfg = GeneratorConfig.from_yaml(config_path)
    assert cfg.shift.enabled is False
    assert cfg.shift.profile == "off"


def test_shift_schema_accepts_profile_with_optional_override() -> None:
    cfg = GeneratorConfig.from_dict(
        {
            "shift": {
                "enabled": True,
                "profile": "Graph_Drift",
                "graph_scale": "0.75",
            }
        }
    )
    assert cfg.shift.enabled is True
    assert cfg.shift.profile == "graph_drift"
    assert cfg.shift.graph_scale == 0.75
    assert cfg.shift.mechanism_scale is None
    assert cfg.shift.noise_scale is None


def test_shift_schema_accepts_mixed_profile_with_multiple_overrides() -> None:
    cfg = GeneratorConfig.from_dict(
        {
            "shift": {
                "enabled": True,
                "profile": "mixed",
                "graph_scale": 0.2,
                "mechanism_scale": 0.4,
                "noise_scale": 0.6,
            }
        }
    )
    assert cfg.shift.profile == "mixed"
    assert cfg.shift.graph_scale == 0.2
    assert cfg.shift.mechanism_scale == 0.4
    assert cfg.shift.noise_scale == 0.6


def test_shift_schema_accepts_custom_profile_with_at_least_one_override() -> None:
    cfg = GeneratorConfig.from_dict(
        {
            "shift": {
                "enabled": True,
                "profile": "custom",
                "mechanism_scale": 0.35,
            }
        }
    )
    assert cfg.shift.profile == "custom"
    assert cfg.shift.mechanism_scale == 0.35


def test_shift_schema_rejects_invalid_profile_value() -> None:
    with pytest.raises(ValueError, match="Unsupported shift.profile"):
        GeneratorConfig.from_dict({"shift": {"enabled": True, "profile": "unknown_mode"}})


def test_shift_schema_accepts_boolean_false_alias_for_off_profile() -> None:
    cfg = GeneratorConfig.from_dict({"shift": {"enabled": False, "profile": False}})
    assert cfg.shift.profile == "off"


def test_shift_schema_rejects_boolean_true_profile_alias() -> None:
    with pytest.raises(ValueError, match="Unsupported shift.profile"):
        GeneratorConfig.from_dict({"shift": {"enabled": False, "profile": True}})


def test_shift_schema_from_yaml_accepts_unquoted_off_profile(tmp_path) -> None:
    cfg_path = tmp_path / "shift_off.yaml"
    cfg_path.write_text(
        "shift:\n  enabled: false\n  profile: off\n",
        encoding="utf-8",
    )
    cfg = GeneratorConfig.from_yaml(cfg_path)
    assert cfg.shift.enabled is False
    assert cfg.shift.profile == "off"


def test_shift_schema_rejects_non_boolean_enabled_value() -> None:
    with pytest.raises(ValueError, match="shift.enabled must be a boolean"):
        GeneratorConfig.from_dict({"shift": {"enabled": "true", "profile": "off"}})


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("graph_scale", -0.1),
        ("graph_scale", 1.1),
        ("graph_scale", float("inf")),
        ("graph_scale", float("nan")),
        ("graph_scale", True),
        ("mechanism_scale", -0.1),
        ("mechanism_scale", 1.1),
        ("mechanism_scale", float("inf")),
        ("mechanism_scale", float("nan")),
        ("mechanism_scale", True),
        ("noise_scale", -0.1),
        ("noise_scale", 1.1),
        ("noise_scale", float("inf")),
        ("noise_scale", float("nan")),
        ("noise_scale", True),
    ],
)
def test_shift_override_scales_enforce_finite_unit_interval_bounds(
    field_name: str, bad_value: float | bool
) -> None:
    with pytest.raises(
        ValueError, match=rf"shift\.{field_name} must be a finite value in \[0, 1\]"
    ):
        GeneratorConfig.from_dict(
            {
                "shift": {
                    "enabled": True,
                    "profile": "mixed",
                    field_name: bad_value,
                }
            }
        )


def test_shift_schema_rejects_non_off_profile_when_disabled() -> None:
    with pytest.raises(
        ValueError, match=r"shift\.profile must be 'off' when shift\.enabled is false"
    ):
        GeneratorConfig.from_dict({"shift": {"enabled": False, "profile": "mixed"}})


def test_shift_schema_rejects_overrides_when_disabled() -> None:
    with pytest.raises(
        ValueError, match=r"shift override scales must be unset when shift\.enabled is false"
    ):
        GeneratorConfig.from_dict(
            {
                "shift": {
                    "enabled": False,
                    "profile": "off",
                    "graph_scale": 0.2,
                }
            }
        )


def test_shift_schema_rejects_off_profile_when_enabled() -> None:
    with pytest.raises(
        ValueError, match=r"shift\.profile must not be 'off' when shift\.enabled is true"
    ):
        GeneratorConfig.from_dict({"shift": {"enabled": True, "profile": "off"}})


def test_shift_schema_requires_custom_profile_to_set_at_least_one_override() -> None:
    with pytest.raises(
        ValueError, match=r"shift\.profile 'custom' requires at least one override scale"
    ):
        GeneratorConfig.from_dict({"shift": {"enabled": True, "profile": "custom"}})


@pytest.mark.parametrize(
    ("profile", "payload", "error_pattern"),
    [
        (
            "graph_drift",
            {"noise_scale": 0.25},
            r"shift\.profile 'graph_drift' only allows shift\.graph_scale override",
        ),
        (
            "mechanism_drift",
            {"graph_scale": 0.25},
            r"shift\.profile 'mechanism_drift' only allows shift\.mechanism_scale override",
        ),
        (
            "noise_drift",
            {"mechanism_scale": 0.25},
            r"shift\.profile 'noise_drift' only allows shift\.noise_scale override",
        ),
    ],
)
def test_shift_schema_rejects_incompatible_profile_override_combinations(
    profile: str,
    payload: dict[str, float],
    error_pattern: str,
) -> None:
    with pytest.raises(ValueError, match=error_pattern):
        GeneratorConfig.from_dict(
            {
                "shift": {
                    "enabled": True,
                    "profile": profile,
                    **payload,
                }
            }
        )
