import pytest
import torch
import numpy as np

from cauchy_generator.config import CurriculumStageConfig, GeneratorConfig
from cauchy_generator.core.dataset import (
    _sample_layout,
    _stratified_split_indices,
    generate_batch,
    generate_batch_iter,
    generate_one,
)
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


def test_generate_one_lineage_assignments_follow_postprocess_feature_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_config()
    cfg.dataset.task = "regression"
    cfg.filter.enabled = False

    layout = {
        "feature_types": ["num", "cat", "num", "cat"],
        "graph_nodes": 3,
        "graph_edges": 2,
        "adjacency": torch.tensor(
            [
                [0, 1, 1],
                [0, 0, 0],
                [0, 0, 0],
            ],
            dtype=torch.bool,
        ),
        "feature_node_assignment": [0, 1, 2, 1],
        "target_node_assignment": 2,
    }

    monkeypatch.setattr(
        "cauchy_generator.core.dataset._sample_layout",
        lambda *_args, **_kwargs: layout,
    )

    def _stub_generate_graph_dataset_torch(_config, _layout, _seed, _device, *, n_rows):
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
    assert len(train_classes) >= 2
    assert train_classes == test_classes


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
