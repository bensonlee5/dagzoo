import pytest

from cauchy_generator.config import GeneratorConfig


def test_load_default_config() -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    assert cfg.curriculum_stage == "off"
    assert cfg.dataset.n_train > 0
    assert cfg.dataset.n_features_min <= cfg.dataset.n_features_max
    assert cfg.output.shard_size > 0
    assert cfg.filter.n_trees > 0
    assert cfg.filter.depth >= 0
    assert cfg.filter.max_features == "auto"
    assert cfg.filter.n_split_candidates > 0


def test_load_cuda_presets() -> None:
    cfg_h100 = GeneratorConfig.from_yaml("configs/preset_cuda_h100.yaml")
    assert cfg_h100.runtime.device == "cuda"
    assert cfg_h100.dataset.n_features_max >= 128


def test_load_benchmark_profiles() -> None:
    cfg_cpu = GeneratorConfig.from_yaml("configs/benchmark_cpu.yaml")
    cfg_desktop = GeneratorConfig.from_yaml("configs/benchmark_cuda_desktop.yaml")
    cfg_h100 = GeneratorConfig.from_yaml("configs/benchmark_cuda_h100.yaml")

    assert cfg_cpu.runtime.device == "cpu"
    assert cfg_desktop.runtime.device == "cuda"
    assert cfg_h100.runtime.device == "cuda"
    assert "cpu" in cfg_h100.benchmark.profiles


def test_load_curriculum_preset() -> None:
    cfg = GeneratorConfig.from_yaml("configs/curriculum_tabiclv2.yaml")
    assert cfg.curriculum_stage == "auto"
    assert cfg.dataset.n_train > 0
    assert cfg.dataset.n_test > 0


def test_runtime_config_from_dict() -> None:
    cfg = GeneratorConfig.from_dict(
        {
            "curriculum_stage": 2,
            "runtime": {
                "device": "cpu",
                "torch_dtype": "float64",
            },
        }
    )
    assert cfg.curriculum_stage == 2
    assert cfg.runtime.device == "cpu"
    assert cfg.runtime.torch_dtype == "float64"


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
