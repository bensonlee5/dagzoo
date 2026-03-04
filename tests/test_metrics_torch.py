from __future__ import annotations

from dataclasses import fields

import pytest

from dagzoo.config import GeneratorConfig
from dagzoo.core.dataset import generate_one
from dagzoo.core.metrics_torch import extract_torch_metrics
from dagzoo.diagnostics import extract_dataset_metrics
from dagzoo.diagnostics.types import DatasetMetrics


def _tiny_config(task: str) -> GeneratorConfig:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.runtime.device = "cpu"
    cfg.dataset.task = task
    cfg.dataset.n_train = 32
    cfg.dataset.n_test = 16
    cfg.dataset.n_features_min = 8
    cfg.dataset.n_features_max = 12
    cfg.graph.n_nodes_min = 2
    cfg.graph.n_nodes_max = 6
    cfg.filter.enabled = False
    cfg.filter.max_attempts = 8
    return cfg


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_extract_torch_metrics_near_parity_with_diagnostics(task: str) -> None:
    cfg = _tiny_config(task)
    bundle = generate_one(cfg, seed=17, device="cpu")

    metric_names = {
        field_info.name for field_info in fields(DatasetMetrics) if field_info.name != "task"
    }
    torch_metrics = extract_torch_metrics(
        bundle,
        target_metric_names=metric_names,
        include_spearman=True,
    )
    diagnostics = extract_dataset_metrics(bundle, include_spearman=True)

    for metric_name in sorted(metric_names):
        expected = getattr(diagnostics, metric_name)
        actual = torch_metrics[metric_name]
        if expected is None:
            assert actual is None
            continue
        assert actual is not None
        if metric_name in {"linearity_proxy", "nonlinearity_proxy"}:
            assert float(actual) == pytest.approx(float(expected), rel=0.25, abs=0.05)
        else:
            assert float(actual) == pytest.approx(float(expected), rel=1e-3, abs=1e-3)


def test_generation_does_not_call_numpy_diagnostics_extractor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail_extract(*_args, **_kwargs):
        raise AssertionError("generation should not call diagnostics.extract_dataset_metrics")

    monkeypatch.setattr("dagzoo.diagnostics.metrics.extract_dataset_metrics", _fail_extract)
    monkeypatch.setattr("dagzoo.diagnostics.extract_dataset_metrics", _fail_extract)

    cfg = _tiny_config("regression")

    bundle = generate_one(cfg, seed=2027, device="cpu")
    assert "steering" not in bundle.metadata
