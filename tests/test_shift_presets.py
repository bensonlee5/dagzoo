from __future__ import annotations

import pytest

from cauchy_generator.config import GeneratorConfig


@pytest.mark.parametrize(
    ("config_path", "expected_profile"),
    [
        ("configs/preset_shift_graph_drift_generate_smoke.yaml", "graph_drift"),
        ("configs/preset_shift_mechanism_drift_generate_smoke.yaml", "mechanism_drift"),
        ("configs/preset_shift_noise_drift_generate_smoke.yaml", "noise_drift"),
        ("configs/preset_shift_mixed_generate_smoke.yaml", "mixed"),
    ],
)
def test_shift_generate_presets_load_with_expected_profile(
    config_path: str,
    expected_profile: str,
) -> None:
    cfg = GeneratorConfig.from_yaml(config_path)
    assert cfg.shift.enabled is True
    assert cfg.shift.profile == expected_profile
    assert cfg.filter.enabled is False


def test_shift_benchmark_preset_loads_with_expected_profile() -> None:
    cfg = GeneratorConfig.from_yaml("configs/preset_shift_benchmark_smoke.yaml")
    assert cfg.shift.enabled is True
    assert cfg.shift.profile == "mixed"
    assert cfg.benchmark.profile_name == "shift_smoke"
    assert "shift_smoke" in cfg.benchmark.profiles
