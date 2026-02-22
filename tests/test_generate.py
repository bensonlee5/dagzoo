import pytest
import torch
import numpy as np

from cauchy_generator.config import GeneratorConfig
from cauchy_generator.core.dataset import generate_batch, generate_batch_iter, generate_one
from cauchy_generator.types import DatasetBundle


def _tiny_config() -> GeneratorConfig:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.dataset.n_features_min = 8
    cfg.dataset.n_features_max = 8
    cfg.graph.n_nodes_min = 2
    cfg.graph.n_nodes_max = 6
    return cfg


def test_generate_one_shapes() -> None:
    cfg = _tiny_config()
    bundle = generate_one(cfg, seed=7, device="cpu")
    curriculum = bundle.metadata["curriculum"]
    assert isinstance(bundle.X_train, torch.Tensor)
    assert bundle.X_train.shape[0] == cfg.dataset.n_train
    assert bundle.X_test.shape[0] == cfg.dataset.n_test
    assert int(curriculum["n_rows_total"]) == bundle.X_train.shape[0] + bundle.X_test.shape[0]
    assert curriculum["mode"] == "off"
    assert curriculum["stage"] is None
    assert bundle.X_train.shape[1] == bundle.X_test.shape[1]
    assert len(bundle.feature_types) == bundle.X_train.shape[1]


def test_generate_batch_reproducible_metadata() -> None:
    cfg = _tiny_config()
    batch_a = generate_batch(cfg, num_datasets=2, seed=123, device="cpu")
    batch_b = generate_batch(cfg, num_datasets=2, seed=123, device="cpu")
    assert batch_a[0].metadata["seed"] == batch_b[0].metadata["seed"]
    assert batch_a[0].metadata["curriculum"] == batch_b[0].metadata["curriculum"]
    np.testing.assert_allclose(
        np.asarray(batch_a[0].X_train),
        np.asarray(batch_b[0].X_train),
        atol=1e-6,
        rtol=1e-6,
    )


def test_generate_batch_iter_matches_batch_ordering() -> None:
    cfg = _tiny_config()
    batch = generate_batch(cfg, num_datasets=2, seed=321, device="cpu")
    streamed = list(generate_batch_iter(cfg, num_datasets=2, seed=321, device="cpu"))

    assert len(streamed) == len(batch)
    for a, b in zip(batch, streamed, strict=True):
        np.testing.assert_allclose(np.asarray(a.X_train), np.asarray(b.X_train), atol=1e-6)
        assert a.metadata["seed"] == b.metadata["seed"]
        assert a.metadata["curriculum"] == b.metadata["curriculum"]


def test_generate_one_returns_torch_tensors_on_cpu() -> None:
    cfg = _tiny_config()
    bundle = generate_one(cfg, seed=1234, device="cpu")
    assert isinstance(bundle.X_train, torch.Tensor)
    assert isinstance(bundle.y_train, torch.Tensor)
    assert isinstance(bundle.X_test, torch.Tensor)
    assert isinstance(bundle.y_test, torch.Tensor)
    assert bundle.metadata["backend"] == "torch"


def test_torch_path_applies_filter_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, int] = {"count": 0}

    def _stub_filter(*_args, **_kwargs):
        called["count"] += 1
        return True, {"wins_ratio": 1.0, "n_valid_oob": 128}

    monkeypatch.setattr("cauchy_generator.core.dataset.apply_torch_rf_filter", _stub_filter)
    cfg = _tiny_config()
    cfg.filter.enabled = True

    bundle = generate_one(cfg, seed=77, device="cpu")
    assert called["count"] >= 1
    assert bundle.metadata["backend"] == "torch"
    assert bundle.metadata["filter"]["enabled"] is True
    assert bundle.metadata["filter"]["accepted"] is True
    assert "reason" not in bundle.metadata["filter"]


def test_auto_retries_on_cpu_when_mps_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def _stub_generate_torch(_config, _layout, _seed, device, **_kwargs):
        calls.append(device)
        if device == "mps":
            raise RuntimeError("simulated mps failure")
        return DatasetBundle(
            X_train=torch.zeros((2, 2), dtype=torch.float32),
            y_train=torch.zeros(2, dtype=torch.int64),
            X_test=torch.zeros((1, 2), dtype=torch.float32),
            y_test=torch.zeros(1, dtype=torch.int64),
            feature_types=["num", "num"],
            metadata={"backend": "torch", "device": "cpu"},
        )

    monkeypatch.setattr("cauchy_generator.core.dataset._resolve_device", lambda *_args: "mps")
    monkeypatch.setattr("cauchy_generator.core.dataset._generate_torch", _stub_generate_torch)
    cfg = _tiny_config()

    bundle = generate_one(cfg, seed=123, device="auto")
    assert calls == ["mps", "cpu"]
    assert bundle.metadata["backend"] == "torch"


def test_auto_does_not_fallback_to_numpy_if_torch_runtime_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_runtime(*_args, **_kwargs):
        raise RuntimeError("simulated torch runtime failure")

    monkeypatch.setattr("cauchy_generator.core.dataset._generate_torch", _raise_runtime)
    cfg = _tiny_config()

    with pytest.raises(RuntimeError, match="simulated torch runtime failure"):
        generate_one(cfg, seed=123, device="auto")


def test_explicit_cuda_request_raises_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "cauchy_generator.core.dataset.torch.cuda.is_available",
        lambda: False,
    )
    cfg = _tiny_config()
    with pytest.raises(RuntimeError, match="Requested device 'cuda'"):
        generate_one(cfg, seed=123, device="cuda")


def test_invalid_device_raises() -> None:
    cfg = _tiny_config()
    with pytest.raises(ValueError, match="Unsupported device"):
        generate_one(cfg, seed=123, device="cud")


def test_negative_num_datasets_raises() -> None:
    cfg = _tiny_config()
    with pytest.raises(ValueError, match="num_datasets must be >= 0"):
        list(generate_batch_iter(cfg, num_datasets=-1, seed=123, device="cpu"))


def test_zero_num_datasets_does_not_resolve_device(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _tiny_config()

    def _raise_if_called(*_args, **_kwargs):
        raise RuntimeError("device resolution should not run for empty batches")

    monkeypatch.setattr("cauchy_generator.core.dataset._resolve_device", _raise_if_called)
    assert list(generate_batch_iter(cfg, num_datasets=0, seed=5, device="cuda")) == []


def test_auto_curriculum_generate_one_defaults_to_stage_one() -> None:
    cfg = _tiny_config()
    cfg.curriculum_stage = "auto"

    bundle = generate_one(cfg, seed=17, device="cpu")
    curriculum = bundle.metadata["curriculum"]
    assert curriculum["mode"] == "auto"
    assert curriculum["stage"] == 1
    assert curriculum["n_rows_total"] == 1024
    assert 0.30 <= float(curriculum["train_fraction"]) <= 0.90


@pytest.mark.parametrize("stage,low,high", [(2, 400, 10_240), (3, 400, 60_000)])
def test_fixed_curriculum_stage_ranges(stage: int, low: int, high: int) -> None:
    cfg = _tiny_config()
    cfg.curriculum_stage = stage

    bundle = generate_one(cfg, seed=31 + stage, device="cpu")
    curriculum = bundle.metadata["curriculum"]
    assert curriculum["mode"] == "fixed"
    assert curriculum["stage"] == stage
    assert low <= int(curriculum["n_rows_total"]) <= high
    assert float(curriculum["train_fraction"]) == pytest.approx(0.8)
    assert int(curriculum["n_train"]) + int(curriculum["n_test"]) == int(curriculum["n_rows_total"])


def test_auto_curriculum_batch_stage_sequence_reproducible() -> None:
    cfg = _tiny_config()
    cfg.curriculum_stage = "auto"
    batch_a = generate_batch(cfg, num_datasets=5, seed=701, device="cpu")
    batch_b = generate_batch(cfg, num_datasets=5, seed=701, device="cpu")
    stages_a = [int(bundle.metadata["curriculum"]["stage"]) for bundle in batch_a]
    stages_b = [int(bundle.metadata["curriculum"]["stage"]) for bundle in batch_b]
    assert stages_a == stages_b
    assert set(stages_a).issubset({1, 2, 3})


def test_curriculum_respects_configured_split_size_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_config()
    cfg.curriculum_stage = "auto"
    cfg.dataset.task = "regression"
    cfg.dataset.n_train = 16
    cfg.dataset.n_test = 4

    monkeypatch.setattr(
        "cauchy_generator.core.dataset._sample_auto_stage",
        lambda _rng: 3,
    )

    bundle = generate_batch(cfg, num_datasets=1, seed=41, device="cpu")[0]
    curriculum = bundle.metadata["curriculum"]
    assert curriculum["stage"] == 3
    assert int(curriculum["n_rows_total"]) <= 20
    assert int(curriculum["n_train"]) + int(curriculum["n_test"]) == int(curriculum["n_rows_total"])
    assert bundle.X_train.shape[0] == int(curriculum["n_train"])
    assert bundle.X_test.shape[0] == int(curriculum["n_test"])


def test_curriculum_ceiling_applies_when_total_matches_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_config()
    cfg.curriculum_stage = "auto"
    cfg.dataset.task = "regression"
    cfg.dataset.n_train = 900
    cfg.dataset.n_test = 124

    monkeypatch.setattr(
        "cauchy_generator.core.dataset._sample_auto_stage",
        lambda _rng: 3,
    )
    monkeypatch.setattr(
        "cauchy_generator.core.dataset._sample_stage_rows",
        lambda _stage, _rng: (60_000, 0.8),
    )

    bundle = generate_batch(cfg, num_datasets=1, seed=42, device="cpu")[0]
    curriculum = bundle.metadata["curriculum"]
    assert curriculum["stage"] == 3
    assert int(curriculum["n_rows_total"]) == 1024
    assert int(curriculum["n_train"]) + int(curriculum["n_test"]) == 1024
    assert bundle.X_train.shape[0] == int(curriculum["n_train"])
    assert bundle.X_test.shape[0] == int(curriculum["n_test"])


def test_invalid_curriculum_stage_raises() -> None:
    cfg = _tiny_config()
    cfg.curriculum_stage = "stage4"
    with pytest.raises(ValueError, match="Expected off, auto, 1, 2, or 3"):
        generate_one(cfg, seed=123, device="cpu")


def test_invalid_class_split_raises_after_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _tiny_config()
    cfg.dataset.task = "classification"
    cfg.dataset.n_classes_min = 2
    cfg.dataset.n_classes_max = 2
    cfg.filter.max_attempts = 2

    def _tiny_split(*_args, **_kwargs):
        return {
            "mode": "fixed",
            "stage": 1,
            "n_rows_total": 2,
            "n_train": 1,
            "n_test": 1,
            "train_fraction": 0.5,
        }

    monkeypatch.setattr("cauchy_generator.core.dataset._sample_curriculum", _tiny_split)

    with pytest.raises(ValueError, match="Failed to generate a valid dataset"):
        generate_one(cfg, seed=99, device="cpu")
