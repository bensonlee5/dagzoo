import pytest

from cauchy_generator.config import (
    MISSINGNESS_MECHANISM_MAR,
    MISSINGNESS_MECHANISM_MCAR,
    MISSINGNESS_MECHANISM_MNAR,
    MISSINGNESS_MECHANISM_NONE,
    GeneratorConfig,
)
from cauchy_generator.io.lineage_schema import validate_metadata_lineage


def test_load_default_config() -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    assert cfg.curriculum_stage == "off"
    assert cfg.meta_feature_targets == {}
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
    assert cfg.steering.enabled is False
    assert cfg.steering.max_attempts == 3
    assert cfg.steering.temperature > 0
    assert cfg.filter.n_trees > 0
    assert cfg.filter.depth >= 0
    assert cfg.filter.max_features == "auto"
    assert cfg.filter.n_split_candidates > 0
    assert cfg.dataset.missing_rate == 0.0
    assert cfg.dataset.missing_mechanism == MISSINGNESS_MECHANISM_NONE
    assert cfg.dataset.missing_mar_observed_fraction == 0.5
    assert cfg.dataset.missing_mar_logit_scale == 1.0
    assert cfg.dataset.missing_mnar_logit_scale == 1.0
    assert cfg.curriculum.stages == {}


def test_default_config_metadata_is_compatible_with_optional_lineage() -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    metadata = {
        "seed": int(cfg.seed),
        "config": cfg.to_dict(),
    }
    validate_metadata_lineage(metadata, required=False)


def test_load_cuda_presets() -> None:
    cfg_h100 = GeneratorConfig.from_yaml("configs/preset_cuda_h100.yaml")
    assert cfg_h100.runtime.device == "cuda"
    assert cfg_h100.dataset.n_features_max >= 128


def test_load_diagnostics_and_steering_presets() -> None:
    cfg_diag = GeneratorConfig.from_yaml("configs/preset_diagnostics_on.yaml")
    assert cfg_diag.diagnostics.enabled is True
    assert cfg_diag.steering.enabled is False
    assert cfg_diag.diagnostics.histogram_bins >= 8
    assert cfg_diag.diagnostics.quantiles

    cfg_steering = GeneratorConfig.from_yaml("configs/preset_steering_conservative.yaml")
    assert cfg_steering.steering.enabled is True
    assert cfg_steering.steering.max_attempts >= 2
    assert cfg_steering.steering.temperature > 0
    assert cfg_steering.meta_feature_targets
    for band in cfg_steering.meta_feature_targets.values():
        assert len(band) in {2, 3}


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


def test_load_curriculum_preset() -> None:
    cfg = GeneratorConfig.from_yaml("configs/curriculum_tabiclv2.yaml")
    assert cfg.curriculum_stage == "auto"
    assert cfg.dataset.n_train > 0
    assert cfg.dataset.n_test > 0
    assert cfg.curriculum.stages == {}


def test_curriculum_stage_schema_parses_with_string_stage_keys() -> None:
    cfg = GeneratorConfig.from_dict(
        {
            "dataset": {"n_features_min": 8, "n_features_max": 64},
            "graph": {"n_nodes_min": 2, "n_nodes_max": 32},
            "curriculum": {
                "stages": {
                    "1": {
                        "n_features_min": 8,
                        "n_features_max": 16,
                        "n_nodes_min": 2,
                        "n_nodes_max": 6,
                    },
                    "2": {
                        "n_features_min": 16,
                        "n_features_max": 32,
                        "depth_min": 2,
                        "depth_max": 5,
                    },
                    "3": {
                        "n_features_min": 24,
                        "n_features_max": 64,
                        "n_nodes_min": 6,
                        "n_nodes_max": 32,
                    },
                }
            },
        }
    )
    assert set(cfg.curriculum.stages) == {1, 2, 3}
    stage2 = cfg.curriculum.stages[2]
    assert stage2.depth_min == 2
    assert stage2.depth_max == 5


def test_curriculum_stage_schema_rejects_invalid_stage_key() -> None:
    with pytest.raises(ValueError, match="Unsupported curriculum stage key '4'"):
        GeneratorConfig.from_dict({"curriculum": {"stages": {"4": {"n_features_min": 8}}}})


@pytest.mark.parametrize("stage_key", (1.9, "1.9"))
def test_curriculum_stage_schema_rejects_decimal_stage_key(stage_key: object) -> None:
    with pytest.raises(ValueError, match="Unsupported curriculum stage key"):
        GeneratorConfig.from_dict({"curriculum": {"stages": {stage_key: {"n_features_min": 8}}}})


def test_curriculum_stage_schema_rejects_non_mapping_stage_payload() -> None:
    with pytest.raises(ValueError, match=r"curriculum\.stages\['1'\] must be a mapping"):
        GeneratorConfig.from_dict({"curriculum": {"stages": {"1": 12}}})


@pytest.mark.parametrize("bad_curriculum", ([1, 2], "bad"))
def test_curriculum_schema_rejects_non_mapping_curriculum_payload(
    bad_curriculum: object,
) -> None:
    with pytest.raises(ValueError, match=r"curriculum must be a mapping"):
        GeneratorConfig.from_dict({"curriculum": bad_curriculum})


def test_curriculum_stage_schema_rejects_invalid_stage_bounds() -> None:
    with pytest.raises(ValueError, match=r"n_features_min must be <= n_features_max"):
        GeneratorConfig.from_dict(
            {
                "curriculum": {"stages": {"1": {"n_features_min": 16, "n_features_max": 8}}},
            }
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    (("n_features_min", 16.9), ("n_nodes_max", 12.1), ("depth_min", "2.4")),
)
def test_curriculum_stage_schema_rejects_non_integral_bounds(
    field_name: str, value: float | str
) -> None:
    with pytest.raises(
        ValueError, match=rf"curriculum\.stages\.\*\.{field_name} must be an integer"
    ):
        GeneratorConfig.from_dict({"curriculum": {"stages": {"1": {field_name: value}}}})


def test_curriculum_stage_schema_rejects_stage_bounds_outside_global_ranges() -> None:
    with pytest.raises(ValueError, match=r"n_nodes_max must be <= graph\.n_nodes_max"):
        GeneratorConfig.from_dict(
            {
                "graph": {"n_nodes_min": 2, "n_nodes_max": 10},
                "curriculum": {"stages": {"2": {"n_nodes_min": 2, "n_nodes_max": 12}}},
            }
        )


def test_curriculum_stage_schema_rejects_features_min_above_global_max() -> None:
    with pytest.raises(ValueError, match=r"n_features_min must be <= dataset\.n_features_max"):
        GeneratorConfig.from_dict(
            {
                "dataset": {"n_features_min": 8, "n_features_max": 32},
                "curriculum": {"stages": {"2": {"n_features_min": 64}}},
            }
        )


def test_curriculum_stage_schema_rejects_features_max_below_global_min() -> None:
    with pytest.raises(ValueError, match=r"n_features_max must be >= dataset\.n_features_min"):
        GeneratorConfig.from_dict(
            {
                "dataset": {"n_features_min": 16, "n_features_max": 64},
                "curriculum": {"stages": {"2": {"n_features_max": 8}}},
            }
        )


def test_curriculum_stage_schema_rejects_nodes_min_above_global_max() -> None:
    with pytest.raises(ValueError, match=r"n_nodes_min must be <= graph\.n_nodes_max"):
        GeneratorConfig.from_dict(
            {
                "graph": {"n_nodes_min": 2, "n_nodes_max": 10},
                "curriculum": {"stages": {"2": {"n_nodes_min": 12}}},
            }
        )


def test_curriculum_stage_schema_rejects_nodes_max_below_global_min() -> None:
    with pytest.raises(ValueError, match=r"n_nodes_max must be >= graph\.n_nodes_min"):
        GeneratorConfig.from_dict(
            {
                "graph": {"n_nodes_min": 4, "n_nodes_max": 12},
                "curriculum": {"stages": {"2": {"n_nodes_max": 2}}},
            }
        )


def test_curriculum_stage_schema_rejects_depth_min_above_effective_nodes_max() -> None:
    with pytest.raises(ValueError, match=r"depth_min must be <= effective graph\.n_nodes_max"):
        GeneratorConfig.from_dict(
            {
                "graph": {"n_nodes_min": 2, "n_nodes_max": 8},
                "curriculum": {"stages": {"2": {"n_nodes_max": 6, "depth_min": 7}}},
            }
        )


def test_curriculum_stage_schema_rejects_depth_max_above_effective_nodes_max() -> None:
    with pytest.raises(ValueError, match=r"depth_max must be <= effective graph\.n_nodes_max"):
        GeneratorConfig.from_dict(
            {
                "graph": {"n_nodes_min": 2, "n_nodes_max": 8},
                "curriculum": {"stages": {"2": {"n_nodes_max": 5, "depth_max": 6}}},
            }
        )


def test_runtime_config_from_dict() -> None:
    cfg = GeneratorConfig.from_dict(
        {
            "curriculum_stage": 2,
            "meta_feature_targets": {
                "linearity_proxy": [0.1, 0.9, 1.5],
            },
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
            "steering": {
                "enabled": True,
                "max_attempts": 4,
                "temperature": 0.25,
            },
        }
    )
    assert cfg.curriculum_stage == 2
    assert cfg.meta_feature_targets["linearity_proxy"] == [0.1, 0.9, 1.5]
    assert cfg.runtime.device == "cpu"
    assert cfg.runtime.torch_dtype == "float64"
    assert cfg.diagnostics.enabled is True
    assert cfg.diagnostics.histogram_bins == 12
    assert cfg.diagnostics.max_values_per_metric == 1234
    assert "linearity_proxy" in cfg.diagnostics.meta_feature_targets
    assert cfg.steering.enabled is True
    assert cfg.steering.max_attempts == 4


def test_legacy_filter_keys_are_rejected() -> None:
    with pytest.raises(TypeError, match="n_estimators"):
        GeneratorConfig.from_dict(
            {
                "filter": {
                    "enabled": True,
                    "n_estimators": 25,
                    "max_depth": 6,
                }
            }
        )


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
