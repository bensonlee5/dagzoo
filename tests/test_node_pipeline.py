import torch

from cauchy_generator.core.node_pipeline import (
    ConverterSpec,
    apply_node_pipeline_torch,
)
from conftest import make_generator as _make_generator


def test_node_pipeline_extracts_requested_columns() -> None:
    g = _make_generator(123)
    parents = [torch.randn(128, 8, generator=g)]
    specs = [
        ConverterSpec(key="feature_0", kind="num", dim=1),
        ConverterSpec(key="feature_1", kind="cat", dim=4, cardinality=5),
        ConverterSpec(key="target", kind="target_cls", dim=3, cardinality=3),
    ]

    x_node, extracted = apply_node_pipeline_torch(parents, 128, specs, g, "cpu")
    assert x_node.shape[0] == 128
    assert "feature_0" in extracted
    assert "feature_1" in extracted
    assert "target" in extracted
    assert extracted["feature_0"].shape == (128,)
    assert extracted["feature_1"].dtype == torch.int64


def test_torch_output_shapes() -> None:
    g = _make_generator()
    parents = [torch.randn(64, 6, generator=g)]
    specs = [
        ConverterSpec(key="f0", kind="num", dim=1),
        ConverterSpec(key="f1", kind="cat", dim=3, cardinality=4),
    ]
    x, ext = apply_node_pipeline_torch(parents, 64, specs, g, "cpu")
    assert x.shape[0] == 64
    assert "f0" in ext
    assert ext["f0"].shape == (64,)
    assert "f1" in ext
    assert ext["f1"].shape == (64,)


def test_torch_no_parents() -> None:
    g = _make_generator(7)
    specs = [ConverterSpec(key="v", kind="num", dim=1)]
    x, ext = apply_node_pipeline_torch([], 64, specs, g, "cpu")
    assert x.shape[0] == 64
    assert "v" in ext


def test_torch_deterministic() -> None:
    specs = [ConverterSpec(key="v", kind="num", dim=1)]
    parents = [torch.randn(32, 4)]

    g1 = _make_generator(0)
    x1, e1 = apply_node_pipeline_torch(parents, 32, specs, g1, "cpu")

    g2 = _make_generator(0)
    x2, e2 = apply_node_pipeline_torch(parents, 32, specs, g2, "cpu")

    torch.testing.assert_close(x1, x2)
    torch.testing.assert_close(e1["v"], e2["v"])
