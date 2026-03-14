import pytest

from dagzoo.config import GeneratorConfig
from dagzoo.diagnostics.effective_diversity import run_filter_calibration
from dagzoo.filtering import run_deferred_filter


def test_run_deferred_filter_raises_not_implemented(tmp_path) -> None:
    with pytest.raises(NotImplementedError, match="Deferred filtering is temporarily disabled"):
        run_deferred_filter(in_dir=tmp_path / "generated", out_dir=tmp_path / "filter")


def test_run_filter_calibration_raises_not_implemented() -> None:
    cfg = GeneratorConfig.from_yaml("configs/preset_filter_benchmark_smoke.yaml")

    with pytest.raises(NotImplementedError, match="Deferred filtering is temporarily disabled"):
        run_filter_calibration(
            config=cfg,
            config_path="configs/preset_filter_benchmark_smoke.yaml",
            thresholds=None,
            suite="smoke",
            num_datasets=10,
            warmup=0,
            device="cpu",
            warn_threshold_pct=2.5,
            fail_threshold_pct=5.0,
        )
