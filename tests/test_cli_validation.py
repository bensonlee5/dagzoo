import pytest

from cauchy_generator.cli import main


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
