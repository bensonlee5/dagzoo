import numpy as np
import pytest

from cauchy_generator.config import GeneratorConfig
from cauchy_generator.core.dataset import generate_batch, generate_one


def _tiny_steering_config() -> GeneratorConfig:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.runtime.device = "cpu"
    cfg.dataset.task = "regression"
    cfg.dataset.n_train = 32
    cfg.dataset.n_test = 16
    cfg.dataset.n_features_min = 8
    cfg.dataset.n_features_max = 16
    cfg.graph.n_nodes_min = 2
    cfg.graph.n_nodes_max = 6
    cfg.filter.enabled = False
    return cfg


def _in_band_fraction(batch: list, *, lo: int, hi: int) -> float:
    count = sum(1 for bundle in batch if lo <= int(bundle.metadata["n_features"]) <= hi)
    return float(count / len(batch)) if batch else 0.0


def test_steering_is_deterministic_and_bounded() -> None:
    cfg = _tiny_steering_config()
    cfg.meta_feature_targets = {"n_features": [12, 12, 1.0]}
    cfg.steering.enabled = True
    cfg.steering.max_attempts = 4
    cfg.steering.temperature = 0.25

    batch_a = generate_batch(cfg, num_datasets=4, seed=2026, device="cpu")
    batch_b = generate_batch(cfg, num_datasets=4, seed=2026, device="cpu")

    for bundle_a, bundle_b in zip(batch_a, batch_b, strict=True):
        np.testing.assert_allclose(
            np.asarray(bundle_a.X_train), np.asarray(bundle_b.X_train), atol=1e-6
        )
        assert bundle_a.metadata["seed"] == bundle_b.metadata["seed"]
        steering_a = bundle_a.metadata["steering"]
        steering_b = bundle_b.metadata["steering"]
        assert steering_a == steering_b
        assert steering_a["enabled"] is True
        assert 1 <= int(steering_a["candidate_count"]) <= cfg.steering.max_attempts
        assert int(steering_a["selected_candidate_index"]) < int(steering_a["candidate_count"])
        assert "targets" in steering_a


def test_steering_improves_target_band_coverage_vs_baseline() -> None:
    baseline_cfg = _tiny_steering_config()
    baseline_cfg.meta_feature_targets = {"n_features": [12, 12, 1.0]}
    baseline_cfg.steering.enabled = False

    steered_cfg = _tiny_steering_config()
    steered_cfg.meta_feature_targets = {"n_features": [12, 12, 1.0]}
    steered_cfg.steering.enabled = True
    steered_cfg.steering.max_attempts = 5
    steered_cfg.steering.temperature = 0.2

    baseline = generate_batch(baseline_cfg, num_datasets=20, seed=77, device="cpu")
    steered = generate_batch(steered_cfg, num_datasets=20, seed=77, device="cpu")

    baseline_fraction = _in_band_fraction(baseline, lo=12, hi=12)
    steered_fraction = _in_band_fraction(steered, lo=12, hi=12)
    assert steered_fraction >= baseline_fraction + 0.10


def test_non_steered_generation_ignores_meta_targets() -> None:
    cfg_plain = _tiny_steering_config()
    cfg_plain.steering.enabled = False

    cfg_targets = _tiny_steering_config()
    cfg_targets.meta_feature_targets = {"linearity_proxy": [0.1, 0.9, 1.0]}
    cfg_targets.steering.enabled = False

    batch_plain = generate_batch(cfg_plain, num_datasets=3, seed=15, device="cpu")
    batch_targets = generate_batch(cfg_targets, num_datasets=3, seed=15, device="cpu")

    for plain, with_targets in zip(batch_plain, batch_targets, strict=True):
        np.testing.assert_allclose(
            np.asarray(plain.X_train),
            np.asarray(with_targets.X_train),
            atol=1e-6,
            rtol=1e-6,
        )
        assert plain.metadata["seed"] == with_targets.metadata["seed"]
        assert "steering" not in plain.metadata
        assert "steering" not in with_targets.metadata


def test_generate_one_emits_steering_metadata_when_enabled() -> None:
    cfg = _tiny_steering_config()
    cfg.meta_feature_targets = {"linearity_proxy": [0.2, 0.8, 1.0]}
    cfg.steering.enabled = True

    bundle = generate_one(cfg, seed=123, device="cpu")
    steering = bundle.metadata["steering"]
    assert steering["enabled"] is True
    assert int(steering["candidate_count"]) >= 1
    assert "targets" in steering


def test_generate_batch_rejects_unknown_steering_metric() -> None:
    cfg = _tiny_steering_config()
    cfg.meta_feature_targets = {"typo_metric": [0.2, 0.8, 1.0]}
    cfg.steering.enabled = True

    with pytest.raises(ValueError, match="Unsupported steering target metric"):
        generate_batch(cfg, num_datasets=2, seed=11, device="cpu")


def test_generate_batch_rejects_unknown_legacy_diagnostics_target() -> None:
    cfg = _tiny_steering_config()
    cfg.diagnostics.meta_feature_targets = {"bad_metric": [0.2, 0.8]}
    cfg.steering.enabled = True

    with pytest.raises(ValueError, match="bad_metric"):
        generate_batch(cfg, num_datasets=2, seed=12, device="cpu")


def test_generate_one_rejects_task_incompatible_steering_metric() -> None:
    cfg = _tiny_steering_config()
    cfg.meta_feature_targets = {"class_entropy": [0.2, 1.0, 1.0]}
    cfg.steering.enabled = True

    with pytest.raises(ValueError, match="class_entropy"):
        generate_one(cfg, seed=13, device="cpu")
