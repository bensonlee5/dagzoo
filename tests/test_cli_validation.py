import pytest
import yaml

from dagsynth.cli import main
from dagsynth.config import GeneratorConfig


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


def test_generate_cli_rejects_negative_seed() -> None:
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "generate",
                "--config",
                "configs/default.yaml",
                "--num-datasets",
                "1",
                "--seed",
                "-1",
                "--no-write",
            ]
        )
    assert int(exc.value.code) == 2


def test_generate_cli_rejects_oversized_seed() -> None:
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "generate",
                "--config",
                "configs/default.yaml",
                "--num-datasets",
                "1",
                "--seed",
                "4294967296",
                "--no-write",
            ]
        )
    assert int(exc.value.code) == 2


def test_generate_cli_uses_default_config_without_noise_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, bool] = {"called": False}

    def _stub_generate_batch_iter(
        config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
    ):
        _ = config
        _ = seed
        _ = device
        _ = num_datasets
        captured["called"] = True
        yield object()

    monkeypatch.setattr("dagsynth.cli.generate_batch_iter", _stub_generate_batch_iter)

    code = main(
        [
            "generate",
            "--config",
            "configs/default.yaml",
            "--num-datasets",
            "1",
            "--device",
            "cpu",
            "--hardware-policy",
            "none",
            "--no-write",
        ]
    )
    assert code == 0
    assert captured["called"] is True


def test_generate_cli_writes_resolution_trace_artifact_no_write(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "run"

    def _stub_generate_batch_iter(
        config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
    ):
        _ = config
        _ = seed
        _ = device
        for _ in range(num_datasets):
            yield object()

    monkeypatch.setattr("dagsynth.cli.generate_batch_iter", _stub_generate_batch_iter)

    code = main(
        [
            "generate",
            "--config",
            "configs/default.yaml",
            "--num-datasets",
            "1",
            "--device",
            "cpu",
            "--hardware-policy",
            "none",
            "--out",
            str(out_dir),
            "--no-write",
        ]
    )
    assert code == 0
    trace_path = out_dir / "effective_config_trace.yaml"
    assert trace_path.exists()
    trace_payload = yaml.safe_load(trace_path.read_text(encoding="utf-8"))
    assert isinstance(trace_payload, list)
    assert any(
        isinstance(item, dict) and item.get("path") == "runtime.device" for item in trace_payload
    )


def test_generate_cli_many_class_preset_end_to_end_no_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dagsynth.core import dataset as dataset_mod

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
        "dagsynth.cli.generate_batch_iter",
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
            "--hardware-policy",
            "none",
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
    from dagsynth.core import dataset as dataset_mod

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
        "dagsynth.cli.generate_batch_iter",
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
            "--hardware-policy",
            "none",
            "--no-write",
        ]
    )

    assert code == 0
    assert len(captured_shift) == 2
    for payload in captured_shift:
        assert payload["enabled"] is True
        assert payload["profile"] == expected_profile


@pytest.mark.parametrize(
    ("config_path", "expected_family"),
    [
        ("configs/preset_noise_gaussian_generate_smoke.yaml", "gaussian"),
        ("configs/preset_noise_laplace_generate_smoke.yaml", "laplace"),
        ("configs/preset_noise_student_t_generate_smoke.yaml", "student_t"),
        ("configs/preset_noise_mixture_generate_smoke.yaml", "mixture"),
    ],
)
def test_generate_cli_noise_presets_emit_noise_metadata_no_write(
    monkeypatch: pytest.MonkeyPatch,
    config_path: str,
    expected_family: str,
) -> None:
    from dagsynth.core import dataset as dataset_mod

    captured_noise: list[dict[str, object]] = []
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
            payload = bundle.metadata["noise"]
            assert isinstance(payload, dict)
            captured_noise.append(payload)
            yield bundle

    monkeypatch.setattr(
        "dagsynth.cli.generate_batch_iter",
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
            "--hardware-policy",
            "none",
            "--no-write",
        ]
    )

    assert code == 0
    assert len(captured_noise) == 2
    for payload in captured_noise:
        assert payload["family_requested"] == expected_family
        assert payload["sampling_strategy"] == "dataset_level"
        if expected_family == "mixture":
            assert payload["family_sampled"] in {"gaussian", "laplace", "student_t"}
            weights = payload["mixture_weights"]
            assert isinstance(weights, dict)
            assert sum(float(v) for v in weights.values()) == pytest.approx(1.0)
        else:
            assert payload["family_sampled"] == expected_family
            assert payload["mixture_weights"] is None


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

    monkeypatch.setattr("dagsynth.cli.generate_batch_iter", _stub_generate_batch_iter)
    monkeypatch.setattr(
        "dagsynth.cli.CoverageAggregator.update_bundle",
        lambda _self, _bundle: None,
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
            "--hardware-policy",
            "none",
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

    monkeypatch.setattr("dagsynth.cli.generate_batch_iter", _stub_generate_batch_iter)

    code = main(
        [
            "generate",
            "--config",
            str(config_path),
            "--num-datasets",
            "1",
            "--device",
            "cpu",
            "--hardware-policy",
            "none",
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

    monkeypatch.setattr("dagsynth.cli.generate_batch_iter", _stub_generate_batch_iter)
    monkeypatch.setattr(
        "dagsynth.cli.CoverageAggregator.update_bundle",
        lambda _self, _bundle: None,
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
            "--hardware-policy",
            "none",
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

    monkeypatch.setattr("dagsynth.cli.generate_batch_iter", _stub_generate_batch_iter)

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
            "--hardware-policy",
            "none",
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
                "--hardware-policy",
                "none",
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
            "--hardware-policy",
            "none",
            "--no-write",
        ]
    )
    assert code == 0
