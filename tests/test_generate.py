import numpy as np
import pytest
import torch
import math

from cauchy_generator.config import CurriculumStageConfig, GeneratorConfig
from cauchy_generator.core.dataset import (
    _sample_layout,
    _stratified_split_indices,
    generate_batch,
    generate_batch_iter,
    generate_one,
)
from cauchy_generator.core.shift import mechanism_nonlinear_mass, resolve_shift_runtime_params
from cauchy_generator.io.lineage_schema import (
    LINEAGE_SCHEMA_NAME,
    LINEAGE_SCHEMA_VERSION,
    validate_metadata_lineage,
    validate_lineage_payload,
)
from cauchy_generator.rng import SeedManager
from cauchy_generator.types import DatasetBundle


def _tiny_config() -> GeneratorConfig:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.dataset.n_features_min = 8
    cfg.dataset.n_features_max = 8
    cfg.graph.n_nodes_min = 2
    cfg.graph.n_nodes_max = 6
    return cfg


def _tiny_regression_config() -> GeneratorConfig:
    cfg = _tiny_config()
    cfg.dataset.task = "regression"
    cfg.filter.enabled = False
    cfg.dataset.n_features_min = 8
    cfg.dataset.n_features_max = 8
    return cfg


def _layout_stub(
    *,
    feature_types: list[str],
    graph_nodes: int,
    adjacency: torch.Tensor,
    feature_node_assignment: list[int],
    target_node_assignment: int,
) -> dict[str, object]:
    graph_edges = int(adjacency.to(dtype=torch.int64).sum().item())
    return {
        "feature_types": list(feature_types),
        "graph_nodes": int(graph_nodes),
        "graph_edges": graph_edges,
        "graph_depth_nodes": int(graph_nodes),
        "graph_edge_density": 0.0,
        "adjacency": adjacency,
        "feature_node_assignment": list(feature_node_assignment),
        "target_node_assignment": int(target_node_assignment),
    }


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


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_generate_one_curriculum_emits_realized_complexity_metadata(task: str) -> None:
    cfg = _tiny_config()
    cfg.dataset.task = task
    if task == "classification":
        cfg.dataset.n_classes_min = 2
        cfg.dataset.n_classes_max = 2

    bundle = generate_one(cfg, seed=18 if task == "classification" else 19, device="cpu")
    curriculum = bundle.metadata["curriculum"]
    realized = curriculum["realized_complexity"]

    assert curriculum["monotonicity_axes"] == [
        "n_rows_total",
        "n_features",
        "graph_nodes",
        "graph_depth_nodes",
    ]
    assert int(realized["n_rows_total"]) == int(curriculum["n_rows_total"])
    assert int(realized["n_train"]) == int(curriculum["n_train"]) == int(bundle.X_train.shape[0])
    assert int(realized["n_test"]) == int(curriculum["n_test"]) == int(bundle.X_test.shape[0])
    assert int(realized["n_features"]) == int(bundle.metadata["n_features"])
    assert int(realized["graph_nodes"]) == int(bundle.metadata["graph_nodes"])
    assert int(realized["graph_depth_nodes"]) == int(bundle.metadata["graph_depth_nodes"])
    assert float(realized["graph_edge_density"]) == pytest.approx(
        float(bundle.metadata["graph_edge_density"])
    )


def test_generate_one_emits_lineage_metadata_schema_fields() -> None:
    cfg = _tiny_config()
    bundle = generate_one(cfg, seed=11, device="cpu")
    lineage = bundle.metadata["lineage"]
    assert lineage["schema_name"] == LINEAGE_SCHEMA_NAME
    assert lineage["schema_version"] == LINEAGE_SCHEMA_VERSION
    assert "graph" in lineage
    assert "assignments" in lineage
    validate_metadata_lineage(bundle.metadata, required=True)
    validate_lineage_payload(lineage)


def test_generate_one_lineage_shapes_match_graph_stats() -> None:
    cfg = _tiny_config()
    bundle = generate_one(cfg, seed=12, device="cpu")
    lineage = bundle.metadata["lineage"]
    graph = lineage["graph"]
    adjacency = graph["adjacency"]
    n_nodes = int(bundle.metadata["graph_nodes"])

    assert int(graph["n_nodes"]) == n_nodes
    assert len(adjacency) == n_nodes
    for row in adjacency:
        assert len(row) == n_nodes

    edge_count = sum(sum(int(value) for value in row) for row in adjacency)
    assert edge_count == int(bundle.metadata["graph_edges"])


def test_generate_one_lineage_assignment_lengths_and_bounds() -> None:
    cfg = _tiny_config()
    bundle = generate_one(cfg, seed=13, device="cpu")
    lineage = bundle.metadata["lineage"]
    assignments = lineage["assignments"]
    n_nodes = int(bundle.metadata["graph_nodes"])

    feature_to_node = assignments["feature_to_node"]
    target_to_node = int(assignments["target_to_node"])
    assert len(feature_to_node) == int(bundle.metadata["n_features"])
    assert 0 <= target_to_node < n_nodes
    for node_index in feature_to_node:
        assert 0 <= int(node_index) < n_nodes


def test_generate_one_emits_graph_complexity_metadata() -> None:
    cfg = _tiny_config()
    bundle = generate_one(cfg, seed=14, device="cpu")

    graph_nodes = int(bundle.metadata["graph_nodes"])
    graph_depth_nodes = int(bundle.metadata["graph_depth_nodes"])
    graph_edge_density = float(bundle.metadata["graph_edge_density"])

    assert 1 <= graph_depth_nodes <= graph_nodes
    assert 0.0 <= graph_edge_density <= 1.0


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


def test_generate_batch_reproducible_lineage_for_fixed_seed() -> None:
    cfg = _tiny_config()
    batch_a = generate_batch(cfg, num_datasets=2, seed=678, device="cpu")
    batch_b = generate_batch(cfg, num_datasets=2, seed=678, device="cpu")

    assert len(batch_a) == len(batch_b)
    for bundle_a, bundle_b in zip(batch_a, batch_b, strict=True):
        assert bundle_a.metadata["lineage"] == bundle_b.metadata["lineage"]


def test_generate_one_shift_disabled_preserves_baseline_outputs() -> None:
    baseline = _tiny_regression_config()
    disabled = _tiny_regression_config()
    disabled.shift.enabled = False
    disabled.shift.profile = "off"

    bundle_base = generate_one(baseline, seed=1881, device="cpu")
    bundle_disabled = generate_one(disabled, seed=1881, device="cpu")

    torch.testing.assert_close(bundle_base.X_train, bundle_disabled.X_train)
    torch.testing.assert_close(bundle_base.X_test, bundle_disabled.X_test)
    torch.testing.assert_close(bundle_base.y_train, bundle_disabled.y_train)
    torch.testing.assert_close(bundle_base.y_test, bundle_disabled.y_test)
    assert bundle_base.metadata["graph_edges"] == bundle_disabled.metadata["graph_edges"]
    assert bundle_base.metadata["graph_edge_density"] == pytest.approx(
        bundle_disabled.metadata["graph_edge_density"]
    )


def test_generate_one_shift_metadata_emits_disabled_defaults() -> None:
    cfg = _tiny_regression_config()
    cfg.shift.enabled = False
    cfg.shift.profile = "off"

    bundle = generate_one(cfg, seed=1882, device="cpu")
    shift_metadata = bundle.metadata["shift"]
    assert shift_metadata["enabled"] is False
    assert shift_metadata["profile"] == "off"
    assert shift_metadata["graph_scale"] == pytest.approx(0.0)
    assert shift_metadata["mechanism_scale"] == pytest.approx(0.0)
    assert shift_metadata["noise_scale"] == pytest.approx(0.0)
    assert shift_metadata["edge_logit_bias_shift"] == pytest.approx(0.0)
    assert shift_metadata["mechanism_logit_tilt"] == pytest.approx(0.0)
    assert shift_metadata["noise_sigma_multiplier"] == pytest.approx(1.0)
    assert shift_metadata["edge_odds_multiplier"] == pytest.approx(1.0)
    assert shift_metadata["noise_variance_multiplier"] == pytest.approx(1.0)
    assert shift_metadata["mechanism_nonlinear_mass"] == pytest.approx(
        mechanism_nonlinear_mass(mechanism_logit_tilt=0.0)
    )


def test_generate_one_shift_metadata_matches_resolved_runtime_params() -> None:
    cfg = _tiny_regression_config()
    cfg.shift.enabled = True
    cfg.shift.profile = "custom"
    cfg.shift.graph_scale = 0.6
    cfg.shift.mechanism_scale = 0.3
    cfg.shift.noise_scale = 0.4
    runtime = resolve_shift_runtime_params(cfg)

    bundle = generate_one(cfg, seed=1883, device="cpu")
    shift_metadata = bundle.metadata["shift"]
    assert shift_metadata["enabled"] is True
    assert shift_metadata["profile"] == "custom"
    assert shift_metadata["graph_scale"] == pytest.approx(runtime.graph_scale)
    assert shift_metadata["mechanism_scale"] == pytest.approx(runtime.mechanism_scale)
    assert shift_metadata["noise_scale"] == pytest.approx(runtime.noise_scale)
    assert shift_metadata["edge_logit_bias_shift"] == pytest.approx(runtime.edge_logit_bias_shift)
    assert shift_metadata["mechanism_logit_tilt"] == pytest.approx(runtime.mechanism_logit_tilt)
    assert shift_metadata["noise_sigma_multiplier"] == pytest.approx(runtime.noise_sigma_multiplier)
    assert shift_metadata["edge_odds_multiplier"] == pytest.approx(
        math.exp(runtime.edge_logit_bias_shift)
    )
    assert shift_metadata["noise_variance_multiplier"] == pytest.approx(
        runtime.noise_sigma_multiplier**2
    )
    assert shift_metadata["mechanism_nonlinear_mass"] == pytest.approx(
        mechanism_nonlinear_mass(mechanism_logit_tilt=runtime.mechanism_logit_tilt)
    )


def test_generate_one_graph_drift_increases_edge_density_for_same_seed() -> None:
    baseline = _tiny_regression_config()
    baseline.graph.n_nodes_min = 20
    baseline.graph.n_nodes_max = 20

    shifted = _tiny_regression_config()
    shifted.graph.n_nodes_min = 20
    shifted.graph.n_nodes_max = 20
    shifted.shift.enabled = True
    shifted.shift.profile = "graph_drift"

    bundle_base = generate_one(baseline, seed=2203, device="cpu")
    bundle_shifted = generate_one(shifted, seed=2203, device="cpu")

    assert int(bundle_shifted.metadata["graph_edges"]) >= int(bundle_base.metadata["graph_edges"])
    assert float(bundle_shifted.metadata["graph_edge_density"]) >= float(
        bundle_base.metadata["graph_edge_density"]
    )


@pytest.mark.parametrize(
    ("profile", "override_field", "override_value"),
    [
        ("mechanism_drift", "mechanism_scale", 1.0),
        ("noise_drift", "noise_scale", 1.0),
    ],
)
def test_generate_one_shift_profiles_change_outputs_for_same_seed(
    profile: str, override_field: str, override_value: float
) -> None:
    baseline = _tiny_regression_config()
    baseline.graph.n_nodes_min = 10
    baseline.graph.n_nodes_max = 10

    shifted = _tiny_regression_config()
    shifted.graph.n_nodes_min = 10
    shifted.graph.n_nodes_max = 10
    shifted.shift.enabled = True
    shifted.shift.profile = profile
    setattr(shifted.shift, override_field, override_value)

    bundle_base = generate_one(baseline, seed=2204, device="cpu")
    bundle_shifted = generate_one(shifted, seed=2204, device="cpu")
    assert not torch.allclose(bundle_base.X_train, bundle_shifted.X_train)
    assert not torch.allclose(bundle_base.X_test, bundle_shifted.X_test)


@pytest.mark.parametrize(
    ("profile", "overrides"),
    [
        ("graph_drift", {"graph_scale": 1.0}),
        ("mechanism_drift", {"mechanism_scale": 1.0}),
        ("noise_drift", {"noise_scale": 1.0}),
        ("mixed", {}),
    ],
)
def test_generate_one_shift_profiles_are_seed_reproducible(
    profile: str, overrides: dict[str, float]
) -> None:
    cfg = _tiny_regression_config()
    cfg.graph.n_nodes_min = 10
    cfg.graph.n_nodes_max = 10
    cfg.shift.enabled = True
    cfg.shift.profile = profile
    for key, value in overrides.items():
        setattr(cfg.shift, key, value)

    bundle_a = generate_one(cfg, seed=2205, device="cpu")
    bundle_b = generate_one(cfg, seed=2205, device="cpu")

    torch.testing.assert_close(bundle_a.X_train, bundle_b.X_train)
    torch.testing.assert_close(bundle_a.X_test, bundle_b.X_test)
    torch.testing.assert_close(bundle_a.y_train, bundle_b.y_train)
    torch.testing.assert_close(bundle_a.y_test, bundle_b.y_test)
    assert bundle_a.metadata["graph_edges"] == bundle_b.metadata["graph_edges"]
    assert bundle_a.metadata["graph_edge_density"] == pytest.approx(
        bundle_b.metadata["graph_edge_density"]
    )


def test_generate_one_lineage_assignments_follow_postprocess_feature_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_config()
    cfg.dataset.task = "regression"
    cfg.filter.enabled = False

    layout = _layout_stub(
        feature_types=["num", "cat", "num", "cat"],
        graph_nodes=3,
        adjacency=torch.tensor(
            [
                [0, 1, 1],
                [0, 0, 0],
                [0, 0, 0],
            ],
            dtype=torch.bool,
        ),
        feature_node_assignment=[0, 1, 2, 1],
        target_node_assignment=2,
    )

    monkeypatch.setattr(
        "cauchy_generator.core.dataset._sample_layout",
        lambda *_args, **_kwargs: layout,
    )

    def _stub_generate_graph_dataset_torch(_config, _layout, _seed, _device, *, n_rows, **_kwargs):
        x = torch.arange(n_rows * 4, dtype=torch.float32).reshape(n_rows, 4)
        y = torch.linspace(0.0, 1.0, n_rows, dtype=torch.float32)
        return x, y, {"filter": {"enabled": False}}

    monkeypatch.setattr(
        "cauchy_generator.core.dataset._generate_graph_dataset_torch",
        _stub_generate_graph_dataset_torch,
    )

    def _stub_postprocess_dataset(
        x_train,
        y_train,
        x_test,
        y_test,
        feature_types,
        _task,
        _generator,
        _device,
        *,
        return_feature_index_map=False,
    ):
        assert return_feature_index_map is True
        index_map = [2, 0, 3]
        reordered_types = [feature_types[i] for i in index_map]
        return (
            x_train[:, index_map],
            y_train,
            x_test[:, index_map],
            y_test,
            reordered_types,
            index_map,
        )

    monkeypatch.setattr(
        "cauchy_generator.core.dataset.postprocess_dataset",
        _stub_postprocess_dataset,
    )

    bundle = generate_one(cfg, seed=777, device="cpu")
    assert int(bundle.metadata["n_features"]) == 3
    assert bundle.metadata["lineage"]["assignments"]["feature_to_node"] == [2, 0, 1]


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
        return True, {
            "wins_ratio": 1.0,
            "n_valid_oob": 128,
            "threshold_requested": 0.95,
            "threshold_effective": 0.80,
            "threshold_policy": "class_aware_piecewise_v1",
            "class_count": 32,
            "class_bucket": "25-32",
            "threshold_delta": 0.15,
        }

    monkeypatch.setattr("cauchy_generator.core.dataset.apply_torch_rf_filter", _stub_filter)
    cfg = _tiny_config()
    cfg.filter.enabled = True

    bundle = generate_one(cfg, seed=77, device="cpu")
    assert called["count"] >= 1
    assert bundle.metadata["backend"] == "torch"
    assert bundle.metadata["filter"]["enabled"] is True
    assert bundle.metadata["filter"]["accepted"] is True
    assert bundle.metadata["filter"]["threshold_policy"] == "class_aware_piecewise_v1"
    assert bundle.metadata["filter"]["class_bucket"] == "25-32"
    assert float(bundle.metadata["filter"]["threshold_effective"]) == pytest.approx(0.80)
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


def test_stagewise_structure_bias_increases_density_for_same_rng_stream() -> None:
    cfg = _tiny_config()
    cfg.dataset.task = "regression"
    cfg.dataset.n_features_min = 8
    cfg.dataset.n_features_max = 8
    cfg.graph.n_nodes_min = 20
    cfg.graph.n_nodes_max = 20

    layout_stage1 = _sample_layout(
        cfg,
        SeedManager(190).torch_rng("layout"),
        "cpu",
        curriculum={"stage": 1},
    )
    layout_stage3 = _sample_layout(
        cfg,
        SeedManager(190).torch_rng("layout"),
        "cpu",
        curriculum={"stage": 3},
    )

    assert int(layout_stage3["graph_edges"]) >= int(layout_stage1["graph_edges"])
    assert float(layout_stage3["graph_edge_density"]) >= float(layout_stage1["graph_edge_density"])


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_fixed_curriculum_stage_enforces_graph_depth_bounds(task: str) -> None:
    cfg = _tiny_config()
    cfg.dataset.task = task
    cfg.curriculum_stage = 2
    cfg.filter.enabled = False
    cfg.graph.n_nodes_min = 3
    cfg.graph.n_nodes_max = 3
    cfg.curriculum.stages = {
        2: CurriculumStageConfig(
            n_nodes_min=3,
            n_nodes_max=3,
            depth_min=3,
            depth_max=3,
        )
    }

    if task == "classification":
        cfg.dataset.n_classes_min = 2
        cfg.dataset.n_classes_max = 2

    bundle = generate_one(cfg, seed=912 if task == "classification" else 913, device="cpu")
    assert int(bundle.metadata["curriculum"]["stage"]) == 2
    assert int(bundle.metadata["graph_nodes"]) == 3
    assert int(bundle.metadata["graph_depth_nodes"]) == 3


def test_stage_depth_min_above_nodes_min_still_generates_valid_graph() -> None:
    cfg = _tiny_config()
    cfg.dataset.task = "regression"
    cfg.curriculum_stage = 2
    cfg.filter.enabled = False
    cfg.graph.n_nodes_min = 3
    cfg.graph.n_nodes_max = 6
    cfg.curriculum.stages = {
        2: CurriculumStageConfig(
            n_nodes_min=3,
            n_nodes_max=6,
            depth_min=5,
        )
    }

    bundle = generate_one(cfg, seed=931, device="cpu")
    assert int(bundle.metadata["curriculum"]["stage"]) == 2
    assert int(bundle.metadata["graph_nodes"]) >= 5
    assert int(bundle.metadata["graph_depth_nodes"]) >= 5
    assert bundle.metadata["curriculum"]["stage_bounds"]["n_nodes_min"] == 5
    assert bundle.metadata["curriculum"]["stage_bounds"]["n_nodes_max"] == 6


def test_curriculum_off_ignores_stagewise_depth_bounds() -> None:
    cfg = _tiny_config()
    cfg.dataset.task = "regression"
    cfg.curriculum_stage = "off"
    cfg.filter.enabled = False
    cfg.graph.n_nodes_min = 3
    cfg.graph.n_nodes_max = 3
    cfg.curriculum.stages = {
        1: CurriculumStageConfig(
            n_nodes_min=3,
            n_nodes_max=3,
            depth_min=4,
            depth_max=4,
        )
    }

    bundle = generate_one(cfg, seed=920, device="cpu")
    assert bundle.metadata["curriculum"]["stage"] is None
    assert int(bundle.metadata["graph_nodes"]) == 3
    assert int(bundle.metadata["graph_depth_nodes"]) <= 3


def test_sample_layout_rejects_infeasible_depth_vs_nodes_bounds() -> None:
    cfg = _tiny_config()
    cfg.graph.n_nodes_min = 3
    cfg.graph.n_nodes_max = 5
    cfg.curriculum.stages = {
        2: CurriculumStageConfig(
            n_nodes_min=3,
            n_nodes_max=5,
            depth_min=6,
        )
    }

    with pytest.raises(ValueError, match="Invalid effective node/depth bounds"):
        _sample_layout(
            cfg,
            SeedManager(932).torch_rng("layout"),
            "cpu",
            curriculum={"stage": 2},
        )


def test_sample_layout_stage_bounds_uses_effective_node_min_with_depth_constraints() -> None:
    cfg = _tiny_config()
    cfg.graph.n_nodes_min = 3
    cfg.graph.n_nodes_max = 6
    cfg.curriculum.stages = {
        2: CurriculumStageConfig(
            n_nodes_min=3,
            n_nodes_max=6,
            depth_min=5,
        )
    }

    layout = _sample_layout(
        cfg,
        SeedManager(934).torch_rng("layout"),
        "cpu",
        curriculum={"stage": 2},
    )

    stage_bounds = layout["stage_bounds"]
    assert stage_bounds["n_nodes_min"] == 5
    assert stage_bounds["n_nodes_max"] == 6


def test_generate_one_accepts_depth_bounds_with_global_node_floor() -> None:
    cfg = _tiny_config()
    cfg.dataset.task = "regression"
    cfg.curriculum_stage = 2
    cfg.filter.enabled = False
    cfg.graph.n_nodes_min = 1
    cfg.graph.n_nodes_max = 1
    cfg.curriculum.stages = {
        2: CurriculumStageConfig(
            depth_min=2,
            depth_max=2,
        )
    }

    bundle = generate_one(cfg, seed=935, device="cpu")
    assert int(bundle.metadata["curriculum"]["stage"]) == 2
    assert int(bundle.metadata["graph_nodes"]) == 2
    assert int(bundle.metadata["graph_depth_nodes"]) == 2
    assert bundle.metadata["curriculum"]["stage_bounds"]["n_nodes_min"] == 2
    assert bundle.metadata["curriculum"]["stage_bounds"]["n_nodes_max"] == 2


def test_fixed_curriculum_stage_enforces_feature_and_node_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_config()
    cfg.dataset.task = "regression"
    cfg.curriculum_stage = 2
    cfg.filter.enabled = False
    cfg.dataset.n_features_min = 8
    cfg.dataset.n_features_max = 64
    cfg.graph.n_nodes_min = 2
    cfg.graph.n_nodes_max = 32
    cfg.curriculum.stages = {
        2: CurriculumStageConfig(
            n_features_min=11,
            n_features_max=11,
            n_nodes_min=7,
            n_nodes_max=7,
        )
    }

    def _identity_postprocess(
        x_train,
        y_train,
        x_test,
        y_test,
        feature_types,
        _task,
        _generator,
        _device,
        *,
        return_feature_index_map=False,
    ):
        assert return_feature_index_map is True
        index_map = list(range(int(x_train.shape[1])))
        return x_train, y_train, x_test, y_test, list(feature_types), index_map

    monkeypatch.setattr(
        "cauchy_generator.core.dataset.postprocess_dataset",
        _identity_postprocess,
    )

    bundle = generate_one(cfg, seed=2026, device="cpu")
    assert int(bundle.metadata["curriculum"]["stage"]) == 2
    assert int(bundle.metadata["n_features"]) == 11
    assert int(bundle.metadata["graph_nodes"]) == 7
    assert int(bundle.X_train.shape[1]) == 11
    assert int(bundle.X_test.shape[1]) == 11
    assert bundle.metadata["curriculum"]["stage_bounds"] == {
        "n_features_min": 11,
        "n_features_max": 11,
        "n_nodes_min": 7,
        "n_nodes_max": 7,
        "depth_min": None,
        "depth_max": None,
    }


def test_curriculum_off_ignores_stagewise_feature_and_node_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_config()
    cfg.dataset.task = "regression"
    cfg.curriculum_stage = "off"
    cfg.filter.enabled = False
    cfg.dataset.n_features_min = 6
    cfg.dataset.n_features_max = 6
    cfg.graph.n_nodes_min = 4
    cfg.graph.n_nodes_max = 4
    cfg.curriculum.stages = {
        1: CurriculumStageConfig(
            n_features_min=20,
            n_features_max=20,
            n_nodes_min=12,
            n_nodes_max=12,
        )
    }

    def _identity_postprocess(
        x_train,
        y_train,
        x_test,
        y_test,
        feature_types,
        _task,
        _generator,
        _device,
        *,
        return_feature_index_map=False,
    ):
        assert return_feature_index_map is True
        index_map = list(range(int(x_train.shape[1])))
        return x_train, y_train, x_test, y_test, list(feature_types), index_map

    monkeypatch.setattr(
        "cauchy_generator.core.dataset.postprocess_dataset",
        _identity_postprocess,
    )

    bundle = generate_one(cfg, seed=2027, device="cpu")
    assert bundle.metadata["curriculum"]["stage"] is None
    assert int(bundle.metadata["n_features"]) == 6
    assert int(bundle.metadata["graph_nodes"]) == 4
    assert bundle.metadata["curriculum"]["stage_bounds"] == {
        "n_features_min": 6,
        "n_features_max": 6,
        "n_nodes_min": 4,
        "n_nodes_max": 4,
        "depth_min": None,
        "depth_max": None,
    }


def test_curriculum_stage_bounds_clamp_node_limits_to_sampler_floor() -> None:
    cfg = _tiny_config()
    cfg.dataset.task = "regression"
    cfg.curriculum_stage = "off"
    cfg.filter.enabled = False
    cfg.graph.n_nodes_min = 1
    cfg.graph.n_nodes_max = 1

    bundle = generate_one(cfg, seed=2028, device="cpu")
    assert int(bundle.metadata["graph_nodes"]) == 2
    assert bundle.metadata["curriculum"]["stage_bounds"]["n_nodes_min"] == 2
    assert bundle.metadata["curriculum"]["stage_bounds"]["n_nodes_max"] == 2


def test_stagewise_layout_sampling_is_seed_reproducible_for_feature_and_node_bounds() -> None:
    cfg = _tiny_config()
    cfg.dataset.n_features_min = 8
    cfg.dataset.n_features_max = 64
    cfg.graph.n_nodes_min = 2
    cfg.graph.n_nodes_max = 32
    cfg.curriculum.stages = {
        3: CurriculumStageConfig(
            n_features_min=13,
            n_features_max=19,
            n_nodes_min=5,
            n_nodes_max=9,
        )
    }
    curriculum = {"stage": 3}

    layout_a = _sample_layout(
        cfg,
        SeedManager(909).torch_rng("layout"),
        "cpu",
        curriculum=curriculum,
    )
    layout_b = _sample_layout(
        cfg,
        SeedManager(909).torch_rng("layout"),
        "cpu",
        curriculum=curriculum,
    )

    assert layout_a["n_features"] == layout_b["n_features"]
    assert layout_a["graph_nodes"] == layout_b["graph_nodes"]
    assert 13 <= int(layout_a["n_features"]) <= 19
    assert 5 <= int(layout_a["graph_nodes"]) <= 9


def test_stagewise_layout_sampling_is_seed_reproducible_with_depth_constraints() -> None:
    cfg = _tiny_config()
    cfg.dataset.n_features_min = 8
    cfg.dataset.n_features_max = 8
    cfg.graph.n_nodes_min = 3
    cfg.graph.n_nodes_max = 6
    cfg.curriculum.stages = {
        2: CurriculumStageConfig(
            n_nodes_min=3,
            n_nodes_max=6,
            depth_min=5,
            depth_max=5,
        )
    }
    curriculum = {"stage": 2}

    layout_a = _sample_layout(
        cfg,
        SeedManager(933).torch_rng("layout"),
        "cpu",
        curriculum=curriculum,
    )
    layout_b = _sample_layout(
        cfg,
        SeedManager(933).torch_rng("layout"),
        "cpu",
        curriculum=curriculum,
    )

    assert layout_a["graph_nodes"] == layout_b["graph_nodes"]
    assert layout_a["graph_depth_nodes"] == layout_b["graph_depth_nodes"]
    assert layout_a["graph_edges"] == layout_b["graph_edges"]
    torch.testing.assert_close(layout_a["adjacency"], layout_b["adjacency"])
    assert int(layout_a["graph_depth_nodes"]) == 5


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_curriculum_complexity_metadata_is_monotonic_across_stages(
    monkeypatch: pytest.MonkeyPatch,
    task: str,
) -> None:
    cfg = _tiny_config()
    cfg.dataset.task = task
    cfg.filter.enabled = False
    cfg.dataset.n_features_min = 8
    cfg.dataset.n_features_max = 32
    cfg.graph.n_nodes_min = 4
    cfg.graph.n_nodes_max = 8
    if task == "classification":
        cfg.dataset.n_classes_min = 3
        cfg.dataset.n_classes_max = 3

    cfg.curriculum.stages = {
        1: CurriculumStageConfig(
            n_features_min=8,
            n_features_max=8,
            n_nodes_min=4,
            n_nodes_max=4,
            depth_max=2,
        ),
        2: CurriculumStageConfig(
            n_features_min=12,
            n_features_max=12,
            n_nodes_min=6,
            n_nodes_max=6,
            depth_min=2,
            depth_max=3,
        ),
        3: CurriculumStageConfig(
            n_features_min=16,
            n_features_max=16,
            n_nodes_min=8,
            n_nodes_max=8,
            depth_min=3,
        ),
    }

    def _fixed_stage_rows(
        stage: int, _generator: torch.Generator, _device: str
    ) -> tuple[int, float]:
        return {
            1: (512, 0.75),
            2: (1024, 0.80),
            3: (2048, 0.80),
        }[int(stage)]

    monkeypatch.setattr("cauchy_generator.core.curriculum._sample_stage_rows", _fixed_stage_rows)

    realized_by_stage: list[dict[str, int]] = []
    for stage in (1, 2, 3):
        cfg.curriculum_stage = stage
        bundle = generate_one(cfg, seed=2700 + stage, device="cpu")
        curriculum = bundle.metadata["curriculum"]
        realized = curriculum["realized_complexity"]
        assert curriculum["monotonicity_axes"] == [
            "n_rows_total",
            "n_features",
            "graph_nodes",
            "graph_depth_nodes",
        ]
        realized = curriculum["realized_complexity"]
        realized_by_stage.append(
            {
                "n_rows_total": int(realized["n_rows_total"]),
                "n_features": int(realized["n_features"]),
                "graph_nodes": int(realized["graph_nodes"]),
                "graph_depth_nodes": int(realized["graph_depth_nodes"]),
            }
        )

    for axis in ("n_rows_total", "n_features", "graph_nodes", "graph_depth_nodes"):
        axis_values = [values[axis] for values in realized_by_stage]
        assert axis_values == sorted(axis_values)


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
        "cauchy_generator.core.curriculum._sample_auto_stage",
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
        "cauchy_generator.core.curriculum._sample_auto_stage",
        lambda _rng: 3,
    )
    monkeypatch.setattr(
        "cauchy_generator.core.curriculum._sample_stage_rows",
        lambda _stage, _gen, _dev: (60_000, 0.8),
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


def test_infeasible_curriculum_split_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
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

    monkeypatch.setattr("cauchy_generator.core.curriculum._sample_curriculum", _tiny_split)

    with pytest.raises(ValueError, match=r"infeasible class/split combination"):
        generate_one(cfg, seed=99, device="cpu")


def test_sampled_curriculum_split_fails_fast_when_classes_exceed_train_or_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_config()
    cfg.dataset.task = "classification"
    cfg.dataset.n_classes_min = 32
    cfg.dataset.n_classes_max = 32

    def _infeasible_curriculum(*_args, **_kwargs):
        return {
            "mode": "fixed",
            "stage": 1,
            "n_rows_total": 128,
            "n_train": 31,
            "n_test": 97,
            "train_fraction": 31 / 128,
        }

    monkeypatch.setattr(
        "cauchy_generator.core.curriculum._sample_curriculum", _infeasible_curriculum
    )

    with pytest.raises(ValueError, match=r"sampled classification split constraints"):
        generate_one(cfg, seed=99, device="cpu")


def test_stratified_split_ensures_valid_class_split_with_many_classes() -> None:
    """High n_classes with low n_test should not fail with stratified splitting."""
    cfg = _tiny_config()
    cfg.dataset.task = "classification"
    cfg.dataset.n_classes_min = 10
    cfg.dataset.n_classes_max = 10
    cfg.dataset.n_train = 100
    cfg.dataset.n_test = 28
    cfg.filter.max_attempts = 3

    bundle = generate_one(cfg, seed=42, device="cpu")
    train_classes = set(torch.unique(bundle.y_train).tolist())
    test_classes = set(torch.unique(bundle.y_test).tolist())
    all_classes = torch.unique(torch.cat([bundle.y_train, bundle.y_test], dim=0), sorted=True)
    expected = torch.arange(all_classes.numel(), dtype=all_classes.dtype)
    assert len(train_classes) >= 2
    assert train_classes == test_classes
    assert torch.equal(all_classes, expected)

    class_structure = bundle.metadata["class_structure"]
    assert bundle.metadata["n_classes"] == int(class_structure["n_classes_realized"])
    assert int(class_structure["n_classes_sampled"]) == 10
    assert bool(class_structure["labels_contiguous"]) is True
    assert bool(class_structure["train_test_class_match"]) is True
    assert int(class_structure["min_label"]) == 0
    assert int(class_structure["max_label"]) == int(all_classes.numel() - 1)


def test_metadata_n_classes_uses_realized_class_count_for_classification() -> None:
    cfg = _tiny_config()
    cfg.dataset.task = "classification"
    cfg.dataset.n_classes_min = 32
    cfg.dataset.n_classes_max = 32
    cfg.dataset.n_train = 256
    cfg.dataset.n_test = 256
    cfg.filter.max_attempts = 3

    bundle = generate_one(cfg, seed=52, device="cpu")
    all_classes = torch.unique(torch.cat([bundle.y_train, bundle.y_test], dim=0), sorted=True)
    assert bundle.metadata["n_classes"] == int(all_classes.numel())
    assert int(bundle.metadata["class_structure"]["n_classes_sampled"]) == 32
    assert int(bundle.metadata["class_structure"]["n_classes_realized"]) == int(all_classes.numel())


def test_stratified_split_indices_returns_exact_requested_sizes() -> None:
    y = torch.tensor([0] * 8 + [1] * 5 + [2] * 3 + [3], dtype=torch.int64)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(123)
    train_idx, test_idx = _stratified_split_indices(y, 10, generator, "cpu")

    assert int(train_idx.shape[0]) == 10
    assert int(test_idx.shape[0]) == 7

    train_set = set(train_idx.tolist())
    test_set = set(test_idx.tolist())
    assert train_set.isdisjoint(test_set)
    assert train_set | test_set == set(range(int(y.shape[0])))


def test_stratified_split_indices_raises_for_infeasible_constraints() -> None:
    y = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int64)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(123)
    with pytest.raises(ValueError, match="infeasible_stratified_split"):
        _stratified_split_indices(y, 2, generator, "cpu")


def test_generate_retries_when_stratified_split_is_infeasible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_config()
    cfg.dataset.task = "classification"
    cfg.filter.max_attempts = 2

    def _raise_infeasible_split(
        *_args: object, **_kwargs: object
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise ValueError("infeasible_stratified_split: forced for test")

    monkeypatch.setattr(
        "cauchy_generator.core.dataset._stratified_split_indices",
        _raise_infeasible_split,
    )

    with pytest.raises(ValueError, match=r"Last reason: invalid_class_split"):
        generate_one(cfg, seed=99, device="cpu")


def _tiny_missingness_config(
    *,
    task: str,
    mechanism: str,
    missing_rate: float = 0.25,
) -> GeneratorConfig:
    cfg = _tiny_config()
    cfg.dataset.task = task
    cfg.dataset.n_train = 320
    cfg.dataset.n_test = 160
    cfg.dataset.missing_rate = missing_rate
    cfg.dataset.missing_mechanism = mechanism  # type: ignore[assignment]
    return cfg


@pytest.mark.parametrize("mechanism", ["mcar", "mar", "mnar"])
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_generate_one_applies_missingness_and_emits_summary(task: str, mechanism: str) -> None:
    cfg = _tiny_missingness_config(task=task, mechanism=mechanism, missing_rate=0.25)
    bundle = generate_one(cfg, seed=2718, device="cpu")

    assert bundle.X_train.shape[0] == cfg.dataset.n_train
    assert bundle.X_test.shape[0] == cfg.dataset.n_test
    assert bundle.X_train.shape[1] == bundle.X_test.shape[1]
    assert len(bundle.feature_types) == bundle.X_train.shape[1]

    assert torch.isnan(bundle.X_train).any()
    assert torch.isnan(bundle.X_test).any()

    payload = bundle.metadata["missingness"]
    assert payload["enabled"] is True
    assert payload["mechanism"] == mechanism
    assert payload["target_rate"] == pytest.approx(0.25)
    assert payload["missing_count_overall"] == (
        payload["missing_count_train"] + payload["missing_count_test"]
    )
    assert 0.0 <= float(payload["realized_rate_train"]) <= 1.0
    assert 0.0 <= float(payload["realized_rate_test"]) <= 1.0
    assert 0.0 <= float(payload["realized_rate_overall"]) <= 1.0
    assert abs(float(payload["realized_rate_overall"]) - 0.25) <= 0.05

    if task == "classification":
        assert bundle.y_train.dtype == torch.int64
        assert bundle.y_test.dtype == torch.int64
    else:
        assert torch.isfinite(bundle.y_train).all()
        assert torch.isfinite(bundle.y_test).all()


def test_generate_one_missingness_disabled_preserves_default_behavior() -> None:
    cfg = _tiny_missingness_config(task="classification", mechanism="none", missing_rate=0.0)
    bundle = generate_one(cfg, seed=31415, device="cpu")
    assert "missingness" not in bundle.metadata
    assert not torch.isnan(bundle.X_train).any()
    assert not torch.isnan(bundle.X_test).any()


def test_generate_one_missingness_mask_is_reproducible_for_fixed_seed() -> None:
    cfg = _tiny_missingness_config(task="classification", mechanism="mar", missing_rate=0.3)
    a = generate_one(cfg, seed=12345, device="cpu")
    b = generate_one(cfg, seed=12345, device="cpu")

    assert torch.equal(torch.isnan(a.X_train), torch.isnan(b.X_train))
    assert torch.equal(torch.isnan(a.X_test), torch.isnan(b.X_test))
    assert a.metadata["missingness"] == b.metadata["missingness"]
