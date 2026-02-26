import numpy as np
import pytest
import torch

from cauchy_generator.config import GeneratorConfig
from cauchy_generator.core.dataset import generate_batch, generate_one
from cauchy_generator.diagnostics import extract_dataset_metrics, extract_metrics_batch
from cauchy_generator.types import DatasetBundle


def _tiny_config(task: str) -> GeneratorConfig:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.dataset.task = task
    cfg.dataset.n_train = 64
    cfg.dataset.n_test = 32
    cfg.dataset.n_features_min = 8
    cfg.dataset.n_features_max = 8
    cfg.dataset.n_classes_min = 2
    cfg.dataset.n_classes_max = 4
    cfg.graph.n_nodes_min = 2
    cfg.graph.n_nodes_max = 6
    return cfg


def test_extract_dataset_metrics_classification_invariants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _stub_filter(*_args, **_kwargs):
        return True, {"wins_ratio": 0.75, "n_valid_oob": 64, "backend": "torch_rf"}

    monkeypatch.setattr(
        "cauchy_generator.core.steering_metrics.apply_torch_rf_filter", _stub_filter
    )
    bundle = generate_one(_tiny_config("classification"), seed=7, device="cpu")

    metrics = extract_dataset_metrics(bundle, include_spearman=True)
    assert metrics.task == "classification"
    assert metrics.n_rows == bundle.X_train.shape[0] + bundle.X_test.shape[0]
    assert metrics.n_features == bundle.X_train.shape[1]
    assert metrics.n_classes is not None and metrics.n_classes >= 2
    assert metrics.class_entropy is not None and metrics.class_entropy >= 0.0
    assert metrics.majority_minority_ratio is not None and metrics.majority_minority_ratio >= 1.0
    assert metrics.linearity_proxy is not None and 0.0 <= metrics.linearity_proxy <= 1.0
    assert metrics.wins_ratio_proxy == pytest.approx(0.75)
    assert metrics.nonlinearity_proxy is not None and metrics.nonlinearity_proxy >= 0.0
    assert 0.0 <= metrics.categorical_ratio <= 1.0
    if metrics.pearson_abs_mean is not None:
        assert 0.0 <= metrics.pearson_abs_mean <= 1.0
    if metrics.pearson_abs_max is not None:
        assert 0.0 <= metrics.pearson_abs_max <= 1.0
    if metrics.spearman_abs_mean is not None:
        assert 0.0 <= metrics.spearman_abs_mean <= 1.0
    if metrics.spearman_abs_max is not None:
        assert 0.0 <= metrics.spearman_abs_max <= 1.0


def test_extract_dataset_metrics_regression_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _stub_filter(*_args, **_kwargs):
        return True, {"wins_ratio": 0.55, "n_valid_oob": 64, "backend": "torch_rf"}

    monkeypatch.setattr(
        "cauchy_generator.core.steering_metrics.apply_torch_rf_filter", _stub_filter
    )
    bundle = generate_one(_tiny_config("regression"), seed=17, device="cpu")

    metrics = extract_dataset_metrics(bundle)
    assert metrics.task == "regression"
    assert metrics.n_classes is None
    assert metrics.class_entropy is None
    assert metrics.majority_minority_ratio is None
    assert metrics.linearity_proxy is not None and 0.0 <= metrics.linearity_proxy <= 1.0
    assert metrics.wins_ratio_proxy == pytest.approx(0.55)
    assert metrics.snr_proxy_db is None or np.isfinite(metrics.snr_proxy_db)


def test_extract_dataset_metrics_spearman_toggle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _stub_filter(*_args, **_kwargs):
        return True, {"wins_ratio": 0.61, "n_valid_oob": 6, "backend": "torch_rf"}

    monkeypatch.setattr(
        "cauchy_generator.core.steering_metrics.apply_torch_rf_filter", _stub_filter
    )
    bundle = DatasetBundle(
        X_train=np.asarray([[0.0, 10.0], [1.0, 20.0], [2.0, 30.0], [3.0, 40.0]], dtype=np.float32),
        y_train=np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
        X_test=np.asarray([[4.0, 50.0], [5.0, 60.0]], dtype=np.float32),
        y_test=np.asarray([4.0, 5.0], dtype=np.float32),
        feature_types=["num", "num"],
        metadata={"config": {"dataset": {"task": "regression"}}},
    )

    disabled = extract_dataset_metrics(bundle, include_spearman=False)
    enabled = extract_dataset_metrics(bundle, include_spearman=True)
    assert disabled.spearman_abs_mean is None
    assert disabled.spearman_abs_max is None
    assert enabled.spearman_abs_mean is not None
    assert enabled.spearman_abs_max is not None
    assert enabled.spearman_abs_mean == pytest.approx(1.0)
    assert enabled.spearman_abs_max == pytest.approx(1.0)


def test_extract_dataset_metrics_reproducible_for_fixed_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _stub_filter(*_args, **_kwargs):
        return True, {"wins_ratio": 0.42, "n_valid_oob": 64, "backend": "torch_rf"}

    monkeypatch.setattr(
        "cauchy_generator.core.steering_metrics.apply_torch_rf_filter", _stub_filter
    )
    cfg = _tiny_config("classification")
    bundle_a = generate_one(cfg, seed=123, device="cpu")
    bundle_b = generate_one(cfg, seed=123, device="cpu")

    metrics_1 = extract_dataset_metrics(bundle_a)
    metrics_2 = extract_dataset_metrics(bundle_a)
    metrics_3 = extract_dataset_metrics(bundle_b)
    assert metrics_1 == metrics_2
    assert metrics_1 == metrics_3


def test_extract_dataset_metrics_handles_degenerate_constant_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _stub_filter(*_args, **_kwargs):
        return False, {"reason": "insufficient_oob_predictions", "n_valid_oob": 0}

    monkeypatch.setattr(
        "cauchy_generator.core.steering_metrics.apply_torch_rf_filter", _stub_filter
    )
    bundle = DatasetBundle(
        X_train=np.ones((8, 3), dtype=np.float32),
        y_train=np.linspace(0.0, 1.0, num=8, dtype=np.float32),
        X_test=np.ones((4, 3), dtype=np.float32),
        y_test=np.linspace(1.0, 2.0, num=4, dtype=np.float32),
        feature_types=["num", "cat", "cat"],
        metadata={"config": {"dataset": {"task": "regression"}}},
    )

    metrics = extract_dataset_metrics(bundle, include_spearman=True)
    assert metrics.pearson_abs_mean is None
    assert metrics.pearson_abs_max is None
    assert metrics.spearman_abs_mean is None
    assert metrics.spearman_abs_max is None
    assert metrics.wins_ratio_proxy is None
    assert metrics.nonlinearity_proxy is None
    assert metrics.n_categorical_features == 2
    assert metrics.cat_cardinality_min == 1
    assert metrics.cat_cardinality_mean == pytest.approx(1.0)
    assert metrics.cat_cardinality_max == 1


def test_extract_metrics_batch_preserves_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _stub_filter(*_args, **_kwargs):
        return True, {"wins_ratio": 0.8, "n_valid_oob": 64, "backend": "torch_rf"}

    monkeypatch.setattr(
        "cauchy_generator.core.steering_metrics.apply_torch_rf_filter", _stub_filter
    )
    cfg = _tiny_config("classification")
    bundles = generate_batch(cfg, num_datasets=2, seed=900, device="cpu")

    metrics_batch = extract_metrics_batch(bundles)
    expected = [extract_dataset_metrics(bundle) for bundle in bundles]
    assert metrics_batch == expected


def test_extract_dataset_metrics_normalizes_bundle_to_cpu_before_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = DatasetBundle(
        X_train=torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=torch.float32),
        y_train=torch.tensor([0.0, 1.0], dtype=torch.float32),
        X_test=torch.tensor([[4.0, 5.0]], dtype=torch.float32),
        y_test=torch.tensor([2.0], dtype=torch.float32),
        feature_types=["num", "num"],
        metadata={"config": {"dataset": {"task": "regression"}}},
    )

    captured: dict[str, object] = {}

    def _stub_extract(
        bundle_arg: DatasetBundle,
        *,
        target_metric_names: set[str],
        include_spearman: bool = False,
    ) -> dict[str, float | None | str]:
        captured["bundle"] = bundle_arg
        captured["target_metric_names"] = set(target_metric_names)
        captured["include_spearman"] = include_spearman
        return {
            "task": "regression",
            "n_rows": 3.0,
            "n_features": 2.0,
            "n_classes": None,
            "n_categorical_features": 0.0,
            "categorical_ratio": 0.0,
            "linearity_proxy": 0.25,
            "nonlinearity_proxy": 0.0,
            "wins_ratio_proxy": 0.25,
            "pearson_abs_mean": 0.1,
            "pearson_abs_max": 0.2,
            "spearman_abs_mean": None,
            "spearman_abs_max": None,
            "class_entropy": None,
            "majority_minority_ratio": None,
            "snr_proxy_db": 1.0,
            "cat_cardinality_min": None,
            "cat_cardinality_mean": None,
            "cat_cardinality_max": None,
        }

    monkeypatch.setattr(
        "cauchy_generator.diagnostics.metrics.extract_steering_metrics", _stub_extract
    )

    metrics = extract_dataset_metrics(bundle, include_spearman=False)
    extracted_bundle = captured["bundle"]
    assert isinstance(extracted_bundle, DatasetBundle)
    assert extracted_bundle is not bundle
    assert isinstance(extracted_bundle.X_train, torch.Tensor)
    assert isinstance(extracted_bundle.y_train, torch.Tensor)
    assert isinstance(extracted_bundle.X_test, torch.Tensor)
    assert isinstance(extracted_bundle.y_test, torch.Tensor)
    assert extracted_bundle.X_train.device.type == "cpu"
    assert extracted_bundle.y_train.device.type == "cpu"
    assert extracted_bundle.X_test.device.type == "cpu"
    assert extracted_bundle.y_test.device.type == "cpu"
    assert isinstance(captured["target_metric_names"], set)
    assert "spearman_abs_mean" not in captured["target_metric_names"]
    assert captured["include_spearman"] is False
    assert metrics.n_rows == 3
    assert metrics.n_features == 2
