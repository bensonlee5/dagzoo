from __future__ import annotations

import pytest

from dagzoo.config import GeneratorConfig


@pytest.mark.parametrize(
    ("config_path", "expected_family"),
    [
        ("configs/preset_noise_gaussian_generate_smoke.yaml", "gaussian"),
        ("configs/preset_noise_laplace_generate_smoke.yaml", "laplace"),
        ("configs/preset_noise_student_t_generate_smoke.yaml", "student_t"),
        ("configs/preset_noise_mixture_generate_smoke.yaml", "mixture"),
    ],
)
def test_noise_generate_presets_load_with_expected_family(
    config_path: str,
    expected_family: str,
) -> None:
    cfg = GeneratorConfig.from_yaml(config_path)
    assert cfg.noise.family == expected_family
    assert cfg.filter.enabled is False
    if expected_family == "mixture":
        assert cfg.noise.mixture_weights is not None
        assert sum(cfg.noise.mixture_weights.values()) == pytest.approx(1.0)
    else:
        assert cfg.noise.mixture_weights is None


def test_noise_benchmark_preset_loads_with_expected_profile() -> None:
    cfg = GeneratorConfig.from_yaml("configs/preset_noise_benchmark_smoke.yaml")
    assert cfg.noise.family == "mixture"
    assert cfg.benchmark.preset_name == "noise_smoke"
    assert "noise_smoke" in cfg.benchmark.presets
    assert cfg.filter.enabled is False
