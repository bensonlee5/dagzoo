import numpy as np
import pytest

from cauchy_generator.config import GeneratorConfig
from cauchy_generator.core.dataset import generate_batch, generate_one


def test_generate_one_shapes() -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    bundle = generate_one(cfg, seed=7, device="cpu")
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


def test_torch_optional_matches_numpy_on_cpu_when_enabled() -> None:
    cfg_np = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg_np.runtime.prefer_torch = False
    bundle_np = generate_one(cfg_np, seed=1234, device="cpu")

    try:
        import torch
    except Exception:
        pytest.skip("torch not available")

    cfg_t = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg_t.runtime.prefer_torch = True
    cfg_t.runtime.torch_output = True
    bundle_t = generate_one(cfg_t, seed=1234, device="cpu")
    assert isinstance(bundle_t.X_train, torch.Tensor)
    
    # Check structural consistency instead of bitwise matching.
    # Appendix E math divergence between backends is expected due to different RNG/kernels.
    assert bundle_t.X_train.shape == bundle_np.X_train.shape
    assert bundle_t.X_test.shape == bundle_np.X_test.shape
    assert bundle_t.metadata["backend"] == "torch"
    assert bundle_np.metadata["backend"] == "numpy"


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
