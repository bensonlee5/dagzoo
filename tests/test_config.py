from cauchy_generator.config import GeneratorConfig


def test_load_default_config() -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    assert cfg.dataset.n_train > 0
    assert cfg.dataset.n_features_min <= cfg.dataset.n_features_max
    assert cfg.output.shard_size > 0


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
