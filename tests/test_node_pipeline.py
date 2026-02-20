import numpy as np

from cauchy_generator.core.node_pipeline import ConverterSpec, apply_node_pipeline


def test_node_pipeline_extracts_requested_columns() -> None:
    rng = np.random.default_rng(123)
    parents = [rng.normal(size=(128, 8)).astype(np.float32)]
    specs = [
        ConverterSpec(key="feature_0", kind="num", dim=1),
        ConverterSpec(key="feature_1", kind="cat", dim=4, cardinality=5),
        ConverterSpec(key="target", kind="target_cls", dim=3, cardinality=3),
    ]

    x_node, extracted = apply_node_pipeline(parents, 128, specs, rng)
    assert x_node.shape[0] == 128
    assert "feature_0" in extracted
    assert "feature_1" in extracted
    assert "target" in extracted
    assert extracted["feature_0"].shape == (128,)
    assert extracted["feature_1"].dtype == np.int64
