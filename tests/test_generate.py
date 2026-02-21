import pytest
import torch
import numpy as np

from cauchy_generator.config import GeneratorConfig
from cauchy_generator.core.dataset import generate_batch, generate_batch_iter, generate_one
from cauchy_generator.types import DatasetBundle


def test_generate_one_shapes() -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    bundle = generate_one(cfg, seed=7, device="cpu")
    assert isinstance(bundle.X_train, torch.Tensor)
    assert bundle.X_train.shape[0] == cfg.dataset.n_train
    assert bundle.X_test.shape[0] == cfg.dataset.n_test
    assert bundle.X_train.shape[1] == bundle.X_test.shape[1]
    assert len(bundle.feature_types) == bundle.X_train.shape[1]


def test_generate_batch_reproducible_metadata() -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    batch_a = generate_batch(cfg, num_datasets=2, seed=123, device="cpu")
    batch_b = generate_batch(cfg, num_datasets=2, seed=123, device="cpu")
    assert batch_a[0].metadata["seed"] == batch_b[0].metadata["seed"]
    np.testing.assert_allclose(
        np.asarray(batch_a[0].X_train),
        np.asarray(batch_b[0].X_train),
        atol=1e-6,
        rtol=1e-6,
    )


def test_generate_batch_iter_matches_batch_ordering() -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    batch = generate_batch(cfg, num_datasets=2, seed=321, device="cpu")
    streamed = list(generate_batch_iter(cfg, num_datasets=2, seed=321, device="cpu"))

    assert len(streamed) == len(batch)
    for a, b in zip(batch, streamed, strict=True):
        np.testing.assert_allclose(np.asarray(a.X_train), np.asarray(b.X_train), atol=1e-6)
        assert a.metadata["seed"] == b.metadata["seed"]


def test_generate_one_returns_torch_tensors_on_cpu() -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
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
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.filter.enabled = True

    bundle = generate_one(cfg, seed=77, device="cpu")
    assert called["count"] >= 1
    assert bundle.metadata["backend"] == "torch"
    assert bundle.metadata["filter"]["enabled"] is True
    assert bundle.metadata["filter"]["accepted"] is True
    assert "reason" not in bundle.metadata["filter"]


def test_auto_retries_on_cpu_when_mps_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def _stub_generate_torch(_config, _layout, _seed, device):
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
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")

    bundle = generate_one(cfg, seed=123, device="auto")
    assert calls == ["mps", "cpu"]
    assert bundle.metadata["backend"] == "torch"


def test_auto_does_not_fallback_to_numpy_if_torch_runtime_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_runtime(*_args, **_kwargs):
        raise RuntimeError("simulated torch runtime failure")

    monkeypatch.setattr("cauchy_generator.core.dataset._generate_torch", _raise_runtime)
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")

    with pytest.raises(RuntimeError, match="simulated torch runtime failure"):
        generate_one(cfg, seed=123, device="auto")


def test_explicit_cuda_request_raises_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "cauchy_generator.core.dataset.torch.cuda.is_available",
        lambda: False,
    )
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    with pytest.raises(RuntimeError, match="Requested device 'cuda'"):
        generate_one(cfg, seed=123, device="cuda")


def test_invalid_device_raises() -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    with pytest.raises(ValueError, match="Unsupported device"):
        generate_one(cfg, seed=123, device="cud")


def test_negative_num_datasets_raises() -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    with pytest.raises(ValueError, match="num_datasets must be >= 0"):
        list(generate_batch_iter(cfg, num_datasets=-1, seed=123, device="cpu"))


def test_invalid_class_split_raises_after_attempts() -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.dataset.task = "classification"
    cfg.dataset.n_train = 1
    cfg.dataset.n_test = 1
    cfg.dataset.n_classes_min = 2
    cfg.dataset.n_classes_max = 2
    cfg.filter.max_attempts = 2

    with pytest.raises(ValueError, match="Failed to generate a valid dataset"):
        generate_one(cfg, seed=99, device="cpu")
