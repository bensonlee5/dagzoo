import pytest
import yaml

from cauchy_generator.cli import main
from cauchy_generator.config import GeneratorConfig


def test_generate_cli_rejects_invalid_device() -> None:
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "generate",
                "--config",
                "configs/default.yaml",
                "--device",
                "cud",
                "--num-datasets",
                "1",
                "--no-write",
            ]
        )
    assert int(exc.value.code) == 2


def test_generate_cli_rejects_negative_num_datasets() -> None:
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "generate",
                "--config",
                "configs/default.yaml",
                "--num-datasets",
                "-1",
                "--no-write",
            ]
        )
    assert int(exc.value.code) == 2


def test_generate_cli_rejects_invalid_curriculum_value() -> None:
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "generate",
                "--config",
                "configs/default.yaml",
                "--num-datasets",
                "1",
                "--curriculum",
                "4",
                "--no-write",
            ]
        )
    assert int(exc.value.code) == 2


def test_generate_cli_applies_curriculum_override_no_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _stub_generate_batch_iter(
        config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
    ):
        captured["curriculum_stage"] = config.curriculum_stage
        captured["num_datasets"] = num_datasets
        captured["seed"] = seed
        captured["device"] = device
        for _ in range(num_datasets):
            yield object()

    monkeypatch.setattr("cauchy_generator.cli.generate_batch_iter", _stub_generate_batch_iter)

    code = main(
        [
            "generate",
            "--config",
            "configs/default.yaml",
            "--num-datasets",
            "2",
            "--device",
            "cpu",
            "--curriculum",
            "2",
            "--no-hardware-aware",
            "--no-write",
        ]
    )
    assert code == 0
    assert captured["curriculum_stage"] == 2
    assert captured["num_datasets"] == 2
    assert captured["device"] == "cpu"


def test_generate_cli_defaults_to_non_staged_when_curriculum_not_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _stub_generate_batch_iter(
        config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
    ):
        captured["curriculum_stage"] = config.curriculum_stage
        yield object()

    monkeypatch.setattr("cauchy_generator.cli.generate_batch_iter", _stub_generate_batch_iter)

    code = main(
        [
            "generate",
            "--config",
            "configs/default.yaml",
            "--num-datasets",
            "1",
            "--device",
            "cpu",
            "--no-hardware-aware",
            "--no-write",
        ]
    )
    assert code == 0
    assert captured["curriculum_stage"] == "off"


def test_benchmark_cli_rejects_negative_warmup() -> None:
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "benchmark",
                "--config",
                "configs/default.yaml",
                "--profile",
                "custom",
                "--suite",
                "smoke",
                "--warmup",
                "-1",
            ]
        )
    assert int(exc.value.code) == 2


def test_generate_cli_coverage_tolerates_null_quantiles_and_targets(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.runtime.device = "cpu"
    cfg.output.out_dir = str(tmp_path / "run")
    cfg.diagnostics.enabled = True
    cfg.diagnostics.quantiles = None  # type: ignore[assignment]
    cfg.diagnostics.meta_feature_targets = None  # type: ignore[assignment]
    cfg.diagnostics.max_values_per_metric = None
    config_path = tmp_path / "null_diagnostics.yaml"
    config_path.write_text(yaml.safe_dump(cfg.to_dict()), encoding="utf-8")

    def _stub_generate_batch_iter(
        _config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
    ):
        _ = seed
        _ = device
        for _ in range(num_datasets):
            yield object()

    monkeypatch.setattr("cauchy_generator.cli.generate_batch_iter", _stub_generate_batch_iter)
    monkeypatch.setattr(
        "cauchy_generator.cli.CoverageAggregator.update_bundle",
        lambda self, _bundle: None,
    )

    code = main(
        [
            "generate",
            "--config",
            str(config_path),
            "--num-datasets",
            "1",
            "--device",
            "cpu",
            "--no-hardware-aware",
            "--no-write",
        ]
    )
    assert code == 0
    assert (tmp_path / "run" / "coverage_summary.json").exists()
    assert (tmp_path / "run" / "coverage_summary.md").exists()


def test_generate_cli_no_write_allows_null_output_dir_when_coverage_disabled(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.runtime.device = "cpu"
    cfg.output.out_dir = None  # type: ignore[assignment]
    cfg.diagnostics.enabled = False
    config_path = tmp_path / "null_output.yaml"
    config_path.write_text(yaml.safe_dump(cfg.to_dict()), encoding="utf-8")

    def _stub_generate_batch_iter(
        _config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
    ):
        _ = seed
        _ = device
        for _ in range(num_datasets):
            yield object()

    monkeypatch.setattr("cauchy_generator.cli.generate_batch_iter", _stub_generate_batch_iter)

    code = main(
        [
            "generate",
            "--config",
            str(config_path),
            "--num-datasets",
            "1",
            "--device",
            "cpu",
            "--no-hardware-aware",
            "--no-write",
        ]
    )
    assert code == 0


def test_generate_cli_enables_diagnostics_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _stub_generate_batch_iter(
        config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
    ):
        captured["diagnostics_enabled"] = config.diagnostics.enabled
        _ = seed
        _ = device
        for _ in range(num_datasets):
            yield object()

    monkeypatch.setattr("cauchy_generator.cli.generate_batch_iter", _stub_generate_batch_iter)
    monkeypatch.setattr(
        "cauchy_generator.cli.CoverageAggregator.update_bundle",
        lambda self, _bundle: None,
    )
    code = main(
        [
            "generate",
            "--config",
            "configs/default.yaml",
            "--num-datasets",
            "1",
            "--device",
            "cpu",
            "--diagnostics",
            "--no-hardware-aware",
            "--no-write",
        ]
    )
    assert code == 0
    assert captured["diagnostics_enabled"] is True


def test_generate_cli_applies_meta_target_override_and_enables_steering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _stub_generate_batch_iter(
        config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
    ):
        captured["steering_enabled"] = config.steering.enabled
        captured["meta_feature_targets"] = config.meta_feature_targets
        _ = seed
        _ = device
        for _ in range(num_datasets):
            yield object()

    monkeypatch.setattr("cauchy_generator.cli.generate_batch_iter", _stub_generate_batch_iter)

    code = main(
        [
            "generate",
            "--config",
            "configs/default.yaml",
            "--num-datasets",
            "1",
            "--device",
            "cpu",
            "--meta-target",
            "linearity_proxy=0.2:0.8:2.5",
            "--no-hardware-aware",
            "--no-write",
        ]
    )
    assert code == 0
    assert captured["steering_enabled"] is True
    assert captured["meta_feature_targets"]["linearity_proxy"] == [0.2, 0.8, 2.5]


@pytest.mark.parametrize(
    "meta_target",
    [
        "linearity_proxy",
        "linearity_proxy=0.2",
        "linearity_proxy=low:0.8",
        "linearity_proxy=0.2:0.8:0",
        "linearity_proxy=0.2:0.8:-1",
        "linearity_proxy=0.2:0.8:inf",
        "unknown_metric=0.1:0.9",
    ],
)
def test_generate_cli_rejects_invalid_meta_target_override(meta_target: str) -> None:
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "generate",
                "--config",
                "configs/default.yaml",
                "--num-datasets",
                "1",
                "--meta-target",
                meta_target,
                "--no-write",
            ]
        )
    assert int(exc.value.code) == 2


def test_generate_cli_rejects_task_incompatible_meta_target(
    tmp_path,
) -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.dataset.task = "regression"
    config_path = tmp_path / "regression.yaml"
    config_path.write_text(yaml.safe_dump(cfg.to_dict()), encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "generate",
                "--config",
                str(config_path),
                "--num-datasets",
                "1",
                "--meta-target",
                "class_entropy=0.2:1.0",
                "--no-write",
            ]
        )
    assert int(exc.value.code) == 2


def test_generate_cli_rejects_unknown_config_target_key_when_steering_enabled(
    tmp_path,
) -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.dataset.task = "regression"
    cfg.runtime.device = "cpu"
    cfg.steering.enabled = True
    cfg.meta_feature_targets = {
        "linearity_proxy": [0.2, 0.8, 1.0],
        "linarity_proxy": [0.2, 0.8, 1.0],
    }
    config_path = tmp_path / "unknown_target.yaml"
    config_path.write_text(yaml.safe_dump(cfg.to_dict()), encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "generate",
                "--config",
                str(config_path),
                "--num-datasets",
                "1",
                "--device",
                "cpu",
                "--no-hardware-aware",
                "--no-write",
            ]
        )
    assert int(exc.value.code) == 2
