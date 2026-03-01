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


def test_generate_cli_curriculum_auto_preset_end_to_end_no_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cauchy_generator.core import dataset as dataset_mod

    captured_curriculum: list[dict[str, object]] = []
    original_generate_batch_iter = dataset_mod.generate_batch_iter

    def _capture_generate_batch_iter(
        config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
    ):
        for bundle in original_generate_batch_iter(
            config,
            num_datasets=num_datasets,
            seed=seed,
            device=device,
        ):
            payload = bundle.metadata["curriculum"]
            assert isinstance(payload, dict)
            captured_curriculum.append(payload)
            yield bundle

    monkeypatch.setattr(
        "cauchy_generator.cli.generate_batch_iter",
        _capture_generate_batch_iter,
    )

    code = main(
        [
            "generate",
            "--config",
            "configs/preset_curriculum_auto_staged.yaml",
            "--num-datasets",
            "3",
            "--device",
            "cpu",
            "--no-hardware-aware",
            "--no-write",
        ]
    )

    assert code == 0
    assert len(captured_curriculum) == 3
    for payload in captured_curriculum:
        assert payload["mode"] == "auto"
        assert int(payload["stage"]) in {1, 2, 3}


def test_generate_cli_curriculum_fixed_stage_preset_end_to_end_no_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cauchy_generator.core import dataset as dataset_mod

    captured_curriculum: list[dict[str, object]] = []
    original_generate_batch_iter = dataset_mod.generate_batch_iter

    def _capture_generate_batch_iter(
        config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
    ):
        for bundle in original_generate_batch_iter(
            config,
            num_datasets=num_datasets,
            seed=seed,
            device=device,
        ):
            payload = bundle.metadata["curriculum"]
            assert isinstance(payload, dict)
            captured_curriculum.append(payload)
            yield bundle

    monkeypatch.setattr(
        "cauchy_generator.cli.generate_batch_iter",
        _capture_generate_batch_iter,
    )

    code = main(
        [
            "generate",
            "--config",
            "configs/preset_curriculum_stage2.yaml",
            "--num-datasets",
            "2",
            "--device",
            "cpu",
            "--no-hardware-aware",
            "--no-write",
        ]
    )

    assert code == 0
    assert len(captured_curriculum) == 2
    for payload in captured_curriculum:
        assert payload["mode"] == "fixed"
        assert int(payload["stage"]) == 2


def test_generate_cli_many_class_preset_end_to_end_no_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cauchy_generator.core import dataset as dataset_mod

    captured_metadata: list[dict[str, object]] = []
    original_generate_batch_iter = dataset_mod.generate_batch_iter

    def _capture_generate_batch_iter(
        config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
    ):
        for bundle in original_generate_batch_iter(
            config,
            num_datasets=num_datasets,
            seed=seed,
            device=device,
        ):
            captured_metadata.append(bundle.metadata)
            yield bundle

    monkeypatch.setattr(
        "cauchy_generator.cli.generate_batch_iter",
        _capture_generate_batch_iter,
    )

    code = main(
        [
            "generate",
            "--config",
            "configs/preset_many_class_generate_smoke.yaml",
            "--num-datasets",
            "2",
            "--device",
            "cpu",
            "--no-hardware-aware",
            "--no-write",
        ]
    )

    assert code == 0
    assert len(captured_metadata) == 2
    for metadata in captured_metadata:
        class_structure = metadata["class_structure"]
        assert isinstance(class_structure, dict)
        assert 2 <= int(class_structure["n_classes_realized"]) <= 32
        filter_metadata = metadata["filter"]
        assert isinstance(filter_metadata, dict)
        assert filter_metadata["enabled"] is False
        assert "accepted" not in filter_metadata


@pytest.mark.parametrize(
    ("config_path", "expected_profile"),
    [
        ("configs/preset_shift_graph_drift_generate_smoke.yaml", "graph_drift"),
        ("configs/preset_shift_mechanism_drift_generate_smoke.yaml", "mechanism_drift"),
        ("configs/preset_shift_noise_drift_generate_smoke.yaml", "noise_drift"),
        ("configs/preset_shift_mixed_generate_smoke.yaml", "mixed"),
    ],
)
def test_generate_cli_shift_presets_emit_shift_metadata_no_write(
    monkeypatch: pytest.MonkeyPatch,
    config_path: str,
    expected_profile: str,
) -> None:
    from cauchy_generator.core import dataset as dataset_mod

    captured_shift: list[dict[str, object]] = []
    original_generate_batch_iter = dataset_mod.generate_batch_iter

    def _capture_generate_batch_iter(
        config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
    ):
        for bundle in original_generate_batch_iter(
            config,
            num_datasets=num_datasets,
            seed=seed,
            device=device,
        ):
            payload = bundle.metadata["shift"]
            assert isinstance(payload, dict)
            captured_shift.append(payload)
            yield bundle

    monkeypatch.setattr(
        "cauchy_generator.cli.generate_batch_iter",
        _capture_generate_batch_iter,
    )

    code = main(
        [
            "generate",
            "--config",
            config_path,
            "--num-datasets",
            "2",
            "--device",
            "cpu",
            "--no-hardware-aware",
            "--no-write",
        ]
    )

    assert code == 0
    assert len(captured_shift) == 2
    for payload in captured_shift:
        assert payload["enabled"] is True
        assert payload["profile"] == expected_profile


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


def test_generate_cli_applies_missingness_overrides_no_write(
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
        captured["missing_rate"] = config.dataset.missing_rate
        captured["missing_mechanism"] = config.dataset.missing_mechanism
        captured["missing_mar_observed_fraction"] = config.dataset.missing_mar_observed_fraction
        captured["missing_mar_logit_scale"] = config.dataset.missing_mar_logit_scale
        captured["missing_mnar_logit_scale"] = config.dataset.missing_mnar_logit_scale
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
            "--missing-rate",
            "0.3",
            "--missing-mechanism",
            "mar",
            "--missing-mar-observed-fraction",
            "0.7",
            "--missing-mar-logit-scale",
            "1.8",
            "--missing-mnar-logit-scale",
            "2.2",
            "--no-hardware-aware",
            "--no-write",
        ]
    )
    assert code == 0
    assert captured["missing_rate"] == pytest.approx(0.3)
    assert captured["missing_mechanism"] == "mar"
    assert captured["missing_mar_observed_fraction"] == pytest.approx(0.7)
    assert captured["missing_mar_logit_scale"] == pytest.approx(1.8)
    assert captured["missing_mnar_logit_scale"] == pytest.approx(2.2)


def test_generate_cli_rejects_invalid_missingness_combination() -> None:
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "generate",
                "--config",
                "configs/default.yaml",
                "--num-datasets",
                "1",
                "--device",
                "cpu",
                "--missing-rate",
                "0.2",
                "--missing-mechanism",
                "none",
                "--no-hardware-aware",
                "--no-write",
            ]
        )
    assert int(exc.value.code) == 2


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--missing-rate", "1.1"),
        ("--missing-rate", "-0.1"),
        ("--missing-mar-observed-fraction", "0"),
        ("--missing-mar-observed-fraction", "1.1"),
        ("--missing-mar-logit-scale", "0"),
        ("--missing-mnar-logit-scale", "-1"),
    ],
)
def test_generate_cli_rejects_invalid_missingness_scalar(flag: str, value: str) -> None:
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "generate",
                "--config",
                "configs/default.yaml",
                "--num-datasets",
                "1",
                flag,
                value,
                "--no-write",
            ]
        )
    assert int(exc.value.code) == 2


@pytest.mark.parametrize(
    "flag_args", [["--steer-meta"], ["--meta-target", "linearity_proxy=0.2:0.8"]]
)
def test_generate_cli_rejects_removed_steering_flags(flag_args: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "generate",
                "--config",
                "configs/default.yaml",
                "--num-datasets",
                "1",
                *flag_args,
                "--no-write",
            ]
        )
    assert int(exc.value.code) == 2


def test_generate_cli_missingness_no_write_end_to_end(tmp_path) -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.runtime.device = "cpu"
    cfg.dataset.task = "classification"
    cfg.dataset.n_train = 32
    cfg.dataset.n_test = 8
    cfg.dataset.n_features_min = 8
    cfg.dataset.n_features_max = 8
    cfg.graph.n_nodes_min = 2
    cfg.graph.n_nodes_max = 4
    cfg.output.out_dir = str(tmp_path / "run")
    cfg.diagnostics.enabled = False
    config_path = tmp_path / "missingness_e2e.yaml"
    config_path.write_text(yaml.safe_dump(cfg.to_dict()), encoding="utf-8")

    code = main(
        [
            "generate",
            "--config",
            str(config_path),
            "--num-datasets",
            "1",
            "--device",
            "cpu",
            "--missing-rate",
            "0.2",
            "--missing-mechanism",
            "mnar",
            "--missing-mnar-logit-scale",
            "1.5",
            "--no-hardware-aware",
            "--no-write",
        ]
    )
    assert code == 0
