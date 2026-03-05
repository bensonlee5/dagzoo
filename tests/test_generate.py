import numpy as np
import pytest
import torch
import math

from dagzoo.config import (
    GeneratorConfig,
    NOISE_FAMILY_GAUSSIAN,
    NOISE_FAMILY_LAPLACE,
    NOISE_FAMILY_STUDENT_T,
)
from dagzoo.core.constants import NODE_SPEC_SEED_OFFSET, SPLIT_PERMUTATION_SEED_OFFSET
from dagzoo.core.dataset import (
    FixedLayoutPlan,
    generate_batch,
    generate_batch_fixed_layout,
    generate_batch_fixed_layout_iter,
    generate_batch_iter,
    generate_one,
    sample_fixed_layout,
)
from dagzoo.core.generation_context import _attempt_seed, _node_spec_seed, _split_permutation_seed
from dagzoo.core.generation_engine import _generate_torch, _parent_node_indices
from dagzoo.core.layout_types import LayoutPlan
from dagzoo.core.shift import mechanism_nonlinear_mass, resolve_shift_runtime_params
from dagzoo.core.validation import _stratified_split_indices
from dagzoo.io.lineage_schema import (
    LINEAGE_SCHEMA_NAME,
    LINEAGE_SCHEMA_VERSION,
    validate_metadata_lineage,
    validate_lineage_payload,
)
from dagzoo.rng import offset_seed32
from dagzoo.types import DatasetBundle


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
) -> LayoutPlan:
    graph_edges = int(adjacency.to(dtype=torch.int64).sum().item())
    n_features = len(feature_types)
    cat_idx = [idx for idx, kind in enumerate(feature_types) if kind == "cat"]
    card_by_feature = {idx: 4 for idx in cat_idx}
    density_denominator = graph_nodes * max(graph_nodes - 1, 1)
    graph_edge_density = float(graph_edges) / float(density_denominator) if graph_nodes > 1 else 0.0
    return LayoutPlan(
        n_features=n_features,
        n_cat=len(cat_idx),
        cat_idx=cat_idx,
        cardinalities=[4 for _ in cat_idx],
        card_by_feature=card_by_feature,
        n_classes=3,
        feature_types=list(feature_types),
        graph_nodes=int(graph_nodes),
        graph_edges=graph_edges,
        graph_depth_nodes=int(graph_nodes),
        graph_edge_density=graph_edge_density,
        adjacency=adjacency,
        feature_node_assignment=list(feature_node_assignment),
        target_node_assignment=int(target_node_assignment),
    )


def test_parent_node_indices_reads_parents_from_adjacency_columns() -> None:
    adjacency = torch.tensor(
        [
            [0, 1, 0, 1],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ],
        dtype=torch.bool,
    )
    assert _parent_node_indices(adjacency, 0) == []
    assert _parent_node_indices(adjacency, 1) == [0]
    assert _parent_node_indices(adjacency, 2) == [1]
    assert _parent_node_indices(adjacency, 3) == [0, 1, 2]


def test_dataset_seed_helpers_match_offset_seed32_formulas() -> None:
    run_seed = 1337
    attempt = 5
    node_index = 7
    assert _attempt_seed(run_seed, attempt) == offset_seed32(run_seed, attempt)
    assert _node_spec_seed(run_seed, node_index) == offset_seed32(
        run_seed, NODE_SPEC_SEED_OFFSET + node_index
    )
    assert _split_permutation_seed(run_seed, attempt) == offset_seed32(
        run_seed, SPLIT_PERMUTATION_SEED_OFFSET + attempt
    )


def test_generate_one_shapes() -> None:
    cfg = _tiny_config()
    bundle = generate_one(cfg, seed=7, device="cpu")
    assert isinstance(bundle.X_train, torch.Tensor)
    assert bundle.X_train.shape[0] == cfg.dataset.n_train
    assert bundle.X_test.shape[0] == cfg.dataset.n_test
    assert bundle.X_train.shape[1] == bundle.X_test.shape[1]
    assert len(bundle.feature_types) == bundle.X_train.shape[1]


def test_generate_one_uses_fixed_dataset_rows_and_updates_metadata_config_split() -> None:
    cfg = _tiny_config()
    cfg.dataset.rows = 1024  # type: ignore[assignment]
    cfg.dataset.n_test = 256
    cfg.dataset.n_train = 32

    bundle = generate_one(cfg, seed=8, device="cpu")
    assert int(bundle.X_train.shape[0]) == 768
    assert int(bundle.X_test.shape[0]) == 256
    assert int(bundle.metadata["config"]["dataset"]["n_train"]) == 768
    assert int(bundle.metadata["config"]["dataset"]["n_test"]) == 256


def test_generate_batch_rows_choices_are_seed_reproducible() -> None:
    cfg = _tiny_config()
    cfg.dataset.rows = [1024, 2048, 4096]  # type: ignore[assignment]
    cfg.dataset.n_test = 256

    batch_a = generate_batch(cfg, num_datasets=5, seed=19, device="cpu")
    batch_b = generate_batch(cfg, num_datasets=5, seed=19, device="cpu")

    allowed_train_sizes = {768, 1792, 3840}
    for bundle_a, bundle_b in zip(batch_a, batch_b, strict=True):
        assert int(bundle_a.X_train.shape[0]) in allowed_train_sizes
        assert int(bundle_b.X_train.shape[0]) in allowed_train_sizes
        assert int(bundle_a.X_train.shape[0]) == int(bundle_b.X_train.shape[0])
        assert int(bundle_a.X_test.shape[0]) == 256
        assert int(bundle_b.X_test.shape[0]) == 256


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
    disabled.shift.mode = "off"

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
    cfg.shift.mode = "off"

    bundle = generate_one(cfg, seed=1882, device="cpu")
    shift_metadata = bundle.metadata["shift"]
    assert shift_metadata["enabled"] is False
    assert shift_metadata["mode"] == "off"
    assert shift_metadata["graph_scale"] == pytest.approx(0.0)
    assert shift_metadata["mechanism_scale"] == pytest.approx(0.0)
    assert shift_metadata["variance_scale"] == pytest.approx(0.0)
    assert shift_metadata["edge_logit_bias_shift"] == pytest.approx(0.0)
    assert shift_metadata["mechanism_logit_tilt"] == pytest.approx(0.0)
    assert shift_metadata["variance_sigma_multiplier"] == pytest.approx(1.0)
    assert shift_metadata["edge_odds_multiplier"] == pytest.approx(1.0)
    assert shift_metadata["noise_variance_multiplier"] == pytest.approx(1.0)
    assert shift_metadata["mechanism_nonlinear_mass"] == pytest.approx(
        mechanism_nonlinear_mass(mechanism_logit_tilt=0.0)
    )


def test_generate_one_shift_metadata_matches_resolved_runtime_params() -> None:
    cfg = _tiny_regression_config()
    cfg.shift.enabled = True
    cfg.shift.mode = "custom"
    cfg.shift.graph_scale = 0.6
    cfg.shift.mechanism_scale = 0.3
    cfg.shift.variance_scale = 0.4
    runtime = resolve_shift_runtime_params(cfg)

    bundle = generate_one(cfg, seed=1883, device="cpu")
    shift_metadata = bundle.metadata["shift"]
    assert shift_metadata["enabled"] is True
    assert shift_metadata["mode"] == "custom"
    assert shift_metadata["graph_scale"] == pytest.approx(runtime.graph_scale)
    assert shift_metadata["mechanism_scale"] == pytest.approx(runtime.mechanism_scale)
    assert shift_metadata["variance_scale"] == pytest.approx(runtime.variance_scale)
    assert shift_metadata["edge_logit_bias_shift"] == pytest.approx(runtime.edge_logit_bias_shift)
    assert shift_metadata["mechanism_logit_tilt"] == pytest.approx(runtime.mechanism_logit_tilt)
    assert shift_metadata["variance_sigma_multiplier"] == pytest.approx(
        runtime.variance_sigma_multiplier
    )
    assert shift_metadata["edge_odds_multiplier"] == pytest.approx(
        math.exp(runtime.edge_logit_bias_shift)
    )
    assert shift_metadata["noise_variance_multiplier"] == pytest.approx(
        runtime.variance_sigma_multiplier**2
    )
    assert shift_metadata["mechanism_nonlinear_mass"] == pytest.approx(
        mechanism_nonlinear_mass(mechanism_logit_tilt=runtime.mechanism_logit_tilt)
    )


def test_generate_one_shift_metadata_respects_mechanism_family_mix() -> None:
    cfg = _tiny_regression_config()
    cfg.shift.enabled = True
    cfg.shift.mode = "mechanism_drift"
    cfg.shift.mechanism_scale = 1.0
    cfg.mechanism.function_family_mix = {"linear": 1.0}

    bundle = generate_one(cfg, seed=18835, device="cpu")
    shift_metadata = bundle.metadata["shift"]
    assert shift_metadata["mechanism_nonlinear_mass"] == pytest.approx(0.0)
    assert bundle.metadata["config"]["mechanism"]["function_family_mix"] == {"linear": 1.0}


def test_generate_one_noise_metadata_emits_gaussian_defaults() -> None:
    cfg = _tiny_regression_config()
    cfg.noise.family = "gaussian"
    cfg.noise.base_scale = 1.0
    cfg.noise.student_t_df = 5.0

    bundle = generate_one(cfg, seed=1884, device="cpu")
    noise_metadata = bundle.metadata["noise_distribution"]
    assert noise_metadata["family_requested"] == "gaussian"
    assert noise_metadata["family_sampled"] == "gaussian"
    assert noise_metadata["sampling_strategy"] == "dataset_level"
    assert noise_metadata["base_scale"] == pytest.approx(1.0)
    assert noise_metadata["student_t_df"] == pytest.approx(5.0)
    assert noise_metadata["mixture_weights"] is None


@pytest.mark.parametrize("family", [NOISE_FAMILY_LAPLACE, NOISE_FAMILY_STUDENT_T])
def test_generate_one_nongaussian_noise_family_changes_outputs_for_same_seed(family: str) -> None:
    baseline = _tiny_regression_config()
    baseline.noise.family = NOISE_FAMILY_GAUSSIAN

    drifted = _tiny_regression_config()
    drifted.noise.family = family
    drifted.noise.base_scale = 1.0
    if family == NOISE_FAMILY_STUDENT_T:
        drifted.noise.student_t_df = 6.0

    bundle_base = generate_one(baseline, seed=1885, device="cpu")
    bundle_drifted = generate_one(drifted, seed=1885, device="cpu")
    assert not torch.allclose(bundle_base.X_train, bundle_drifted.X_train)
    assert bundle_drifted.metadata["noise_distribution"]["family_requested"] == family
    assert bundle_drifted.metadata["noise_distribution"]["family_sampled"] == family


def test_generate_one_mixture_noise_is_dataset_level_and_reproducible() -> None:
    cfg = _tiny_regression_config()
    cfg.noise.family = "mixture"
    cfg.noise.mixture_weights = {"gaussian": 0.7, "laplace": 0.2, "student_t": 0.1}

    bundle_a = generate_one(cfg, seed=1886, device="cpu")
    bundle_b = generate_one(cfg, seed=1886, device="cpu")
    noise_a = bundle_a.metadata["noise_distribution"]
    noise_b = bundle_b.metadata["noise_distribution"]

    assert noise_a == noise_b
    assert noise_a["family_requested"] == "mixture"
    assert noise_a["family_sampled"] in {"gaussian", "laplace", "student_t"}
    assert noise_a["sampling_strategy"] == "dataset_level"
    assert noise_a["mixture_weights"] is not None
    assert sum(noise_a["mixture_weights"].values()) == pytest.approx(1.0)


def test_generate_one_graph_drift_increases_edge_density_for_same_seed() -> None:
    baseline = _tiny_regression_config()
    baseline.graph.n_nodes_min = 20
    baseline.graph.n_nodes_max = 20

    shifted = _tiny_regression_config()
    shifted.graph.n_nodes_min = 20
    shifted.graph.n_nodes_max = 20
    shifted.shift.enabled = True
    shifted.shift.mode = "graph_drift"

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
        ("noise_drift", "variance_scale", 1.0),
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
    shifted.shift.mode = profile
    setattr(shifted.shift, override_field, override_value)

    bundle_base = generate_one(baseline, seed=2204, device="cpu")
    bundle_shifted = generate_one(shifted, seed=2204, device="cpu")
    if bundle_base.X_train.shape == bundle_shifted.X_train.shape:
        assert not torch.allclose(bundle_base.X_train, bundle_shifted.X_train)
    else:
        assert bundle_base.X_train.shape != bundle_shifted.X_train.shape
    if bundle_base.X_test.shape == bundle_shifted.X_test.shape:
        assert not torch.allclose(bundle_base.X_test, bundle_shifted.X_test)
    else:
        assert bundle_base.X_test.shape != bundle_shifted.X_test.shape


@pytest.mark.parametrize(
    ("profile", "overrides"),
    [
        ("graph_drift", {"graph_scale": 1.0}),
        ("mechanism_drift", {"mechanism_scale": 1.0}),
        ("noise_drift", {"variance_scale": 1.0}),
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
    cfg.shift.mode = profile
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
        "dagzoo.core.generation_engine._sample_layout",
        lambda *_args, **_kwargs: layout,
    )

    def _stub_generate_graph_dataset_torch(_config, _layout, _seed, _device, *, n_rows, **_kwargs):
        x = torch.arange(n_rows * 4, dtype=torch.float32).reshape(n_rows, 4)
        y = torch.linspace(0.0, 1.0, n_rows, dtype=torch.float32)
        return x, y, {"filter": {"enabled": False}}

    monkeypatch.setattr(
        "dagzoo.core.generation_engine._generate_graph_dataset_torch",
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
        preserve_feature_schema=False,
    ):
        assert return_feature_index_map is True
        assert preserve_feature_schema is False
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
        "dagzoo.core.generation_engine.postprocess_dataset",
        _stub_postprocess_dataset,
    )

    bundle = generate_one(cfg, seed=777, device="cpu")
    assert int(bundle.metadata["n_features"]) == 3
    assert bundle.metadata["lineage"]["assignments"]["feature_to_node"] == [2, 0, 1]


def test_generate_torch_forces_cpu_for_stratified_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_config()
    cfg.dataset.task = "classification"
    cfg.filter.enabled = False
    captured: dict[str, str] = {}

    class _SplitSentinel(Exception):
        pass

    def _stub_generate_graph_dataset_torch(
        _config: GeneratorConfig,
        _layout: dict[str, object],
        _seed: int,
        _device: str,
        *,
        n_rows: int,
        **_kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
        x = torch.arange(n_rows * 4, dtype=torch.float32).reshape(n_rows, 4)
        y = torch.arange(n_rows, dtype=torch.int64) % 3
        return x, y, {"accepted": True, "filter": {"enabled": False}}

    def _stub_stratified_split_indices(
        y: torch.Tensor,
        n_train: int,
        generator: torch.Generator,
        device: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _ = n_train
        captured["split_device_arg"] = device
        captured["split_y_device"] = y.device.type
        captured["split_rng_device"] = str(generator.device)
        raise _SplitSentinel

    monkeypatch.setattr(
        "dagzoo.core.generation_engine._generate_graph_dataset_torch",
        _stub_generate_graph_dataset_torch,
    )
    monkeypatch.setattr(
        "dagzoo.core.generation_engine._stratified_split_indices",
        _stub_stratified_split_indices,
    )

    with pytest.raises(_SplitSentinel):
        _generate_torch(
            cfg,
            layout=_layout_stub(
                feature_types=["num", "num", "num", "num"],
                graph_nodes=4,
                adjacency=torch.zeros((4, 4), dtype=torch.bool),
                feature_node_assignment=[0, 1, 2, 3],
                target_node_assignment=3,
            ),
            seed=111,
            device="cuda",
            n_train=8,
            n_test=4,
            requested_device="cuda",
            resolved_device="cuda",
        )

    assert captured["split_device_arg"] == "cpu"
    assert captured["split_y_device"] == "cpu"
    assert captured["split_rng_device"] == "cpu"


def test_generate_torch_routes_postprocess_to_runtime_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_regression_config()
    captured: dict[str, str] = {}

    class _PostprocessSentinel(Exception):
        pass

    def _stub_generate_graph_dataset_torch(
        _config: GeneratorConfig,
        _layout: dict[str, object],
        _seed: int,
        _device: str,
        *,
        n_rows: int,
        **_kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
        x = torch.arange(n_rows * 4, dtype=torch.float32).reshape(n_rows, 4)
        y = torch.linspace(0.0, 1.0, n_rows, dtype=torch.float32)
        return x, y, {"accepted": True, "filter": {"enabled": False}}

    def _stub_postprocess_dataset(
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        _feature_types: list[str],
        _task: str,
        generator: torch.Generator,
        device: str,
        *,
        return_feature_index_map: bool = False,
        preserve_feature_schema: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[int]]:
        _ = return_feature_index_map
        _ = preserve_feature_schema
        _ = y_train
        _ = y_test
        captured["postprocess_device_arg"] = device
        captured["postprocess_x_train_device"] = x_train.device.type
        captured["postprocess_x_test_device"] = x_test.device.type
        captured["postprocess_rng_device"] = str(generator.device)
        raise _PostprocessSentinel

    monkeypatch.setattr(
        "dagzoo.core.generation_engine._generate_graph_dataset_torch",
        _stub_generate_graph_dataset_torch,
    )
    monkeypatch.setattr(
        "dagzoo.core.generation_engine.postprocess_dataset",
        _stub_postprocess_dataset,
    )

    with pytest.raises(_PostprocessSentinel):
        _generate_torch(
            cfg,
            layout=_layout_stub(
                feature_types=["num", "num", "num", "num"],
                graph_nodes=4,
                adjacency=torch.zeros((4, 4), dtype=torch.bool),
                feature_node_assignment=[0, 1, 2, 3],
                target_node_assignment=3,
            ),
            seed=222,
            device="cuda",
            n_train=8,
            n_test=4,
            requested_device="cuda",
            resolved_device="cuda",
        )

    assert captured["postprocess_device_arg"] == "cuda"
    assert captured["postprocess_x_train_device"] == "cpu"
    assert captured["postprocess_x_test_device"] == "cpu"
    assert captured["postprocess_rng_device"] == "cpu"


def test_generate_torch_routes_missingness_to_runtime_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_regression_config()
    captured: dict[str, str] = {}

    class _MissingnessSentinel(Exception):
        pass

    def _stub_generate_graph_dataset_torch(
        _config: GeneratorConfig,
        _layout: dict[str, object],
        _seed: int,
        _device: str,
        *,
        n_rows: int,
        **_kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
        x = torch.arange(n_rows * 4, dtype=torch.float32).reshape(n_rows, 4)
        y = torch.linspace(0.0, 1.0, n_rows, dtype=torch.float32)
        return x, y, {"accepted": True, "filter": {"enabled": False}}

    def _stub_postprocess_dataset(
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        feature_types: list[str],
        _task: str,
        _generator: torch.Generator,
        _device: str,
        *,
        return_feature_index_map: bool = False,
        preserve_feature_schema: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[int]]:
        _ = return_feature_index_map
        _ = preserve_feature_schema
        feature_index_map = list(range(x_train.shape[1]))
        return x_train, y_train, x_test, y_test, feature_types, feature_index_map

    def _stub_inject_missingness(
        x_train: torch.Tensor,
        x_test: torch.Tensor,
        *,
        dataset_cfg,
        seed: int,
        attempt: int,
        device: str,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, object] | None]:
        _ = dataset_cfg
        _ = seed
        _ = attempt
        captured["missingness_device_arg"] = device
        captured["missingness_x_train_device"] = x_train.device.type
        captured["missingness_x_test_device"] = x_test.device.type
        raise _MissingnessSentinel

    monkeypatch.setattr(
        "dagzoo.core.generation_engine._generate_graph_dataset_torch",
        _stub_generate_graph_dataset_torch,
    )
    monkeypatch.setattr(
        "dagzoo.core.generation_engine.postprocess_dataset",
        _stub_postprocess_dataset,
    )
    monkeypatch.setattr(
        "dagzoo.core.generation_engine.inject_missingness",
        _stub_inject_missingness,
    )

    with pytest.raises(_MissingnessSentinel):
        _generate_torch(
            cfg,
            layout=_layout_stub(
                feature_types=["num", "num", "num", "num"],
                graph_nodes=4,
                adjacency=torch.zeros((4, 4), dtype=torch.bool),
                feature_node_assignment=[0, 1, 2, 3],
                target_node_assignment=3,
            ),
            seed=333,
            device="cuda",
            n_train=8,
            n_test=4,
            requested_device="cuda",
            resolved_device="cuda",
        )

    assert captured["missingness_device_arg"] == "cuda"
    assert captured["missingness_x_train_device"] == "cpu"
    assert captured["missingness_x_test_device"] == "cpu"


def test_generate_batch_iter_matches_batch_ordering() -> None:
    cfg = _tiny_config()
    batch = generate_batch(cfg, num_datasets=2, seed=321, device="cpu")
    streamed = list(generate_batch_iter(cfg, num_datasets=2, seed=321, device="cpu"))

    assert len(streamed) == len(batch)
    for a, b in zip(batch, streamed, strict=True):
        np.testing.assert_allclose(np.asarray(a.X_train), np.asarray(b.X_train), atol=1e-6)
        assert a.metadata["seed"] == b.metadata["seed"]


def test_sample_fixed_layout_rejects_variable_rows_spec() -> None:
    cfg = _tiny_regression_config()
    cfg.dataset.rows = "400..60000"  # type: ignore[assignment]

    with pytest.raises(ValueError, match=r"variable dataset\.rows"):
        sample_fixed_layout(cfg, seed=90209, device="cpu")


def test_sample_fixed_layout_accepts_fixed_rows_spec() -> None:
    cfg = _tiny_regression_config()
    cfg.dataset.rows = 1024  # type: ignore[assignment]
    cfg.dataset.n_test = 256
    cfg.dataset.n_train = 64

    plan = sample_fixed_layout(cfg, seed=90208, device="cpu")
    assert plan.n_train == 768
    assert plan.n_test == 256


def test_sample_fixed_layout_is_deterministic_for_seed() -> None:
    cfg = _tiny_regression_config()
    plan_a = sample_fixed_layout(cfg, seed=90210, device="cpu")
    plan_b = sample_fixed_layout(cfg, seed=90210, device="cpu")

    assert plan_a.layout_signature == plan_b.layout_signature
    assert plan_a.plan_seed == plan_b.plan_seed
    assert int(plan_a.layout.n_features) == int(plan_b.layout.n_features)
    assert list(plan_a.layout.feature_types) == list(plan_b.layout.feature_types)


def test_generate_batch_fixed_layout_iter_matches_materialized_ordering() -> None:
    cfg = _tiny_regression_config()
    plan = sample_fixed_layout(cfg, seed=77, device="cpu")
    batch = generate_batch_fixed_layout(cfg, plan=plan, num_datasets=3, seed=800)
    streamed = list(generate_batch_fixed_layout_iter(cfg, plan=plan, num_datasets=3, seed=800))

    assert len(batch) == len(streamed)
    for a, b in zip(batch, streamed, strict=True):
        np.testing.assert_allclose(np.asarray(a.X_train), np.asarray(b.X_train), atol=1e-6)
        np.testing.assert_allclose(np.asarray(a.X_test), np.asarray(b.X_test), atol=1e-6)
        assert a.metadata["seed"] == b.metadata["seed"]
        assert a.metadata["layout_mode"] == "fixed"
        assert b.metadata["layout_mode"] == "fixed"


def test_generate_batch_fixed_layout_enforces_layout_reuse() -> None:
    cfg = _tiny_regression_config()
    plan = sample_fixed_layout(cfg, seed=101, device="cpu")
    batch = generate_batch_fixed_layout(cfg, plan=plan, num_datasets=4, seed=404)

    assert len(batch) == 4
    n_features = int(batch[0].metadata["n_features"])
    feature_types = list(batch[0].feature_types)
    layout_signature = str(batch[0].metadata["layout_signature"])
    feature_to_node = tuple(batch[0].metadata["lineage"]["assignments"]["feature_to_node"])
    for bundle in batch:
        assert bundle.metadata["layout_mode"] == "fixed"
        assert int(bundle.metadata["layout_plan_seed"]) == plan.plan_seed
        assert str(bundle.metadata["layout_signature"]) == layout_signature
        assert int(bundle.metadata["n_features"]) == n_features
        assert list(bundle.feature_types) == feature_types
        assert (
            tuple(bundle.metadata["lineage"]["assignments"]["feature_to_node"]) == feature_to_node
        )
    assert int(batch[0].metadata["seed"]) != int(batch[1].metadata["seed"])
    assert not torch.equal(batch[0].X_train, batch[1].X_train)


def test_generate_batch_fixed_layout_rejects_plan_config_drift() -> None:
    cfg = _tiny_regression_config()
    plan = sample_fixed_layout(cfg, seed=111, device="cpu")

    drifted = _tiny_regression_config()
    drifted.dataset.n_train = cfg.dataset.n_train + 1

    with pytest.raises(
        ValueError,
        match=r"Fixed-layout plan/config mismatch.*dataset\.n_train",
    ):
        list(generate_batch_fixed_layout_iter(drifted, plan=plan, num_datasets=1, seed=222))


def test_generate_batch_fixed_layout_rejects_plan_without_compatibility_snapshot() -> None:
    cfg = _tiny_regression_config()
    sampled = sample_fixed_layout(cfg, seed=112, device="cpu")
    legacy_plan = FixedLayoutPlan(
        layout=sampled.layout,
        requested_device=sampled.requested_device,
        resolved_device=sampled.resolved_device,
        plan_seed=sampled.plan_seed,
        n_train=sampled.n_train,
        n_test=sampled.n_test,
        layout_signature=sampled.layout_signature,
        compatibility_snapshot=None,
    )

    with pytest.raises(ValueError, match=r"compatibility_snapshot must be a mapping"):
        list(generate_batch_fixed_layout_iter(cfg, plan=legacy_plan, num_datasets=1, seed=223))


def test_generate_batch_fixed_layout_allows_non_contract_config_drift() -> None:
    cfg = _tiny_regression_config()
    plan = sample_fixed_layout(cfg, seed=113, device="cpu")

    drifted = _tiny_regression_config()
    drifted.filter.max_attempts = cfg.filter.max_attempts + 1
    drifted.shift.enabled = True
    drifted.shift.mode = "noise_drift"

    batch = list(generate_batch_fixed_layout_iter(drifted, plan=plan, num_datasets=2, seed=224))
    assert len(batch) == 2
    for bundle in batch:
        assert bundle.metadata["layout_mode"] == "fixed"


def test_generate_batch_fixed_layout_rejects_tampered_plan_layout_signature() -> None:
    cfg = _tiny_regression_config()
    plan = sample_fixed_layout(cfg, seed=114, device="cpu")
    plan.layout.graph_edges = int(plan.layout.graph_edges) + 1

    with pytest.raises(ValueError, match=r"plan integrity mismatch"):
        list(generate_batch_fixed_layout_iter(cfg, plan=plan, num_datasets=1, seed=225))


def test_generate_batch_fixed_layout_rejects_tampered_plan_resolved_device() -> None:
    cfg = _tiny_regression_config()
    plan = sample_fixed_layout(cfg, seed=115, device="cpu")
    plan.resolved_device = "cuda"

    with pytest.raises(ValueError, match=r"plan\.resolved_device"):
        list(generate_batch_fixed_layout_iter(cfg, plan=plan, num_datasets=1, seed=226))


def test_generate_batch_fixed_layout_raises_on_schema_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_regression_config()
    plan = sample_fixed_layout(cfg, seed=11, device="cpu")
    calls: dict[str, int] = {"count": 0}

    def _stub_generate_one_with_resolved_layout(*_args, **_kwargs) -> DatasetBundle:
        calls["count"] += 1
        n_features = 3 if calls["count"] == 1 else 2
        return DatasetBundle(
            X_train=torch.zeros((4, n_features), dtype=torch.float32),
            y_train=torch.zeros(4, dtype=torch.float32),
            X_test=torch.zeros((2, n_features), dtype=torch.float32),
            y_test=torch.zeros(2, dtype=torch.float32),
            feature_types=["num"] * n_features,
            metadata={
                "n_features": n_features,
                "lineage": {
                    "assignments": {
                        "feature_to_node": list(range(n_features)),
                        "target_to_node": 0,
                    }
                },
            },
        )

    monkeypatch.setattr(
        "dagzoo.core.fixed_layout._generate_one_with_resolved_layout",
        _stub_generate_one_with_resolved_layout,
    )

    with pytest.raises(ValueError, match="Fixed-layout schema mismatch"):
        list(generate_batch_fixed_layout_iter(cfg, plan=plan, num_datasets=2, seed=123))


def test_generate_one_returns_torch_tensors_on_cpu() -> None:
    cfg = _tiny_config()
    bundle = generate_one(cfg, seed=1234, device="cpu")
    assert isinstance(bundle.X_train, torch.Tensor)
    assert isinstance(bundle.y_train, torch.Tensor)
    assert isinstance(bundle.X_test, torch.Tensor)
    assert isinstance(bundle.y_test, torch.Tensor)
    assert bundle.metadata["backend"] == "torch"


def test_torch_path_sets_deferred_filter_not_run_metadata() -> None:
    cfg = _tiny_config()

    bundle = generate_one(cfg, seed=77, device="cpu")
    assert bundle.metadata["backend"] == "torch"
    assert bundle.metadata["filter"]["mode"] == "deferred"
    assert bundle.metadata["filter"]["status"] == "not_run"
    attempts = bundle.metadata["generation_attempts"]
    assert attempts["total_attempts"] >= 1
    assert attempts["retry_count"] == int(attempts["total_attempts"]) - 1
    assert attempts["filter_attempts"] == 0
    assert attempts["filter_rejections"] == 0
    assert attempts["filter_rejection_rate"] is None


def test_generate_rejects_inline_filter_enabled() -> None:
    cfg = _tiny_config()
    cfg.filter.enabled = True
    with pytest.raises(ValueError, match="Inline filtering has been removed from generate"):
        _ = generate_one(cfg, seed=1122, device="cpu")


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

    monkeypatch.setattr("dagzoo.core.generation_context._resolve_device", lambda *_args: "mps")
    monkeypatch.setattr("dagzoo.core.generation_engine._generate_torch", _stub_generate_torch)
    cfg = _tiny_config()

    bundle = generate_one(cfg, seed=123, device="auto")
    assert calls == ["mps", "cpu"]
    assert bundle.metadata["backend"] == "torch"


def test_auto_does_not_fallback_to_numpy_if_torch_runtime_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_runtime(*_args, **_kwargs):
        raise RuntimeError("simulated torch runtime failure")

    monkeypatch.setattr("dagzoo.core.generation_engine._generate_torch", _raise_runtime)
    cfg = _tiny_config()

    with pytest.raises(RuntimeError, match="simulated torch runtime failure"):
        generate_one(cfg, seed=123, device="auto")


def test_explicit_cuda_request_raises_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dagzoo.core.generation_context.torch.cuda.is_available",
        lambda: False,
    )
    cfg = _tiny_config()
    with pytest.raises(RuntimeError, match="Requested device 'cuda'"):
        generate_one(cfg, seed=123, device="cuda")


def test_invalid_device_raises() -> None:
    cfg = _tiny_config()
    with pytest.raises(ValueError, match="Unsupported device"):
        generate_one(cfg, seed=123, device="cud")


def test_generate_one_rejects_inverted_graph_node_bounds() -> None:
    cfg = _tiny_config()
    cfg.graph.n_nodes_min = 10
    cfg.graph.n_nodes_max = 5

    with pytest.raises(ValueError, match=r"graph\.n_nodes_min must be <= n_nodes_max"):
        generate_one(cfg, seed=123, device="cpu")


def test_generate_one_rejects_inverted_feature_bounds() -> None:
    cfg = _tiny_config()
    cfg.dataset.n_features_min = 10
    cfg.dataset.n_features_max = 5

    with pytest.raises(ValueError, match=r"dataset\.n_features_min must be <= n_features_max"):
        generate_one(cfg, seed=123, device="cpu")


def test_negative_num_datasets_raises() -> None:
    cfg = _tiny_config()
    with pytest.raises(ValueError, match="num_datasets must be >= 0"):
        list(generate_batch_iter(cfg, num_datasets=-1, seed=123, device="cpu"))


@pytest.mark.parametrize("bad_seed", [-1, 4294967296])
def test_generate_one_rejects_out_of_range_seed_override(bad_seed: int) -> None:
    cfg = _tiny_config()
    with pytest.raises(ValueError, match=r"seed must be an integer in \[0, 4294967295\]"):
        generate_one(cfg, seed=bad_seed, device="cpu")


@pytest.mark.parametrize("bad_seed", [-1, 4294967296])
def test_generate_batch_iter_rejects_out_of_range_seed_override(bad_seed: int) -> None:
    cfg = _tiny_config()
    with pytest.raises(ValueError, match=r"seed must be an integer in \[0, 4294967295\]"):
        list(generate_batch_iter(cfg, num_datasets=1, seed=bad_seed, device="cpu"))


@pytest.mark.parametrize("bad_seed", [-1, 4294967296])
def test_sample_fixed_layout_rejects_out_of_range_seed_override(bad_seed: int) -> None:
    cfg = _tiny_config()
    with pytest.raises(ValueError, match=r"seed must be an integer in \[0, 4294967295\]"):
        sample_fixed_layout(cfg, seed=bad_seed, device="cpu")


def test_sample_fixed_layout_accepts_32bit_seed_boundaries() -> None:
    cfg = _tiny_config()
    plan_min = sample_fixed_layout(cfg, seed=0, device="cpu")
    plan_max = sample_fixed_layout(cfg, seed=4294967295, device="cpu")
    assert plan_min.plan_seed == 0
    assert plan_max.plan_seed == 4294967295


def test_zero_num_datasets_does_not_resolve_device(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _tiny_config()

    def _raise_if_called(*_args, **_kwargs):
        raise RuntimeError("device resolution should not run for empty batches")

    monkeypatch.setattr("dagzoo.core.generation_context._resolve_device", _raise_if_called)
    assert list(generate_batch_iter(cfg, num_datasets=0, seed=5, device="cuda")) == []


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
        "dagzoo.core.generation_engine._stratified_split_indices",
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
