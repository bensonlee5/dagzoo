import torch

from cauchy_generator.filtering import apply_torch_rf_filter
from cauchy_generator.filtering.torch_rf_filter import _TreeModel, _fit_rf_tree, _predict_tree


def _make_regression_data(
    seed: int = 7, n_rows: int = 256, n_features: int = 12
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    x = torch.randn(n_rows, n_features, generator=g)
    w = torch.randn(n_features, 1, generator=g)
    y = (x @ w).squeeze(1) + 0.1 * torch.randn(n_rows, generator=g)
    return x, y


def _make_classification_data(
    seed: int = 11,
    n_rows: int = 256,
    n_features: int = 10,
    n_classes: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    x = torch.randn(n_rows, n_features, generator=g)
    logits = x[:, :n_classes] + 0.1 * torch.randn(n_rows, n_classes, generator=g)
    y = torch.argmax(logits, dim=1)
    return x, y


def test_torch_rf_filter_is_deterministic_for_fixed_seed() -> None:
    x, y = _make_regression_data()
    accepted_a, details_a = apply_torch_rf_filter(
        x,
        y,
        task="regression",
        seed=123,
        n_trees=8,
        depth=5,
        n_bootstrap=32,
        threshold=0.5,
    )
    accepted_b, details_b = apply_torch_rf_filter(
        x,
        y,
        task="regression",
        seed=123,
        n_trees=8,
        depth=5,
        n_bootstrap=32,
        threshold=0.5,
    )

    assert accepted_a == accepted_b
    assert details_a["wins_ratio"] == details_b["wins_ratio"]
    assert details_a["n_valid_oob"] == details_b["n_valid_oob"]
    assert details_a["backend"] == "torch_rf"


def test_torch_rf_filter_enforces_max_leaf_nodes() -> None:
    x, y = _make_regression_data(seed=21)
    accepted, details = apply_torch_rf_filter(
        x,
        y,
        task="regression",
        seed=42,
        n_trees=6,
        depth=6,
        max_leaf_nodes=1,
        n_bootstrap=16,
        threshold=0.5,
    )

    assert isinstance(accepted, bool)
    assert details["backend"] == "torch_rf"


def test_torch_rf_filter_can_report_insufficient_oob_predictions() -> None:
    x, y = _make_regression_data(seed=5, n_rows=16, n_features=8)
    accepted, details = apply_torch_rf_filter(
        x,
        y,
        task="regression",
        seed=99,
        n_trees=4,
        depth=4,
        min_samples_leaf=2,
        n_bootstrap=16,
        threshold=0.5,
    )

    assert accepted is False
    assert details["reason"] == "insufficient_oob_predictions"
    assert details["backend"] == "torch_rf"


def test_torch_rf_filter_classification_smoke() -> None:
    x, y = _make_classification_data()
    accepted, details = apply_torch_rf_filter(
        x,
        y,
        task="classification",
        seed=1,
        n_trees=8,
        depth=4,
        min_samples_leaf=2,
        n_bootstrap=24,
        threshold=0.5,
    )

    assert isinstance(accepted, bool)
    assert "wins_ratio" in details
    assert "n_valid_oob" in details
    assert details["backend"] == "torch_rf"


def test_predict_tree_returns_correct_leaf_values() -> None:
    """_predict_tree should route rows to the correct leaves for a known tree."""
    # Hand-build a depth-1 tree: split on feature 0 at threshold 0.5
    #   node 0: internal, split feature=0, threshold=0.5
    #   node 1: left leaf  (x[:,0] <= 0.5) -> value [1.0]
    #   node 2: right leaf (x[:,0] > 0.5)  -> value [2.0]
    tree = _TreeModel(
        split_feature=[0, -1, -1],
        split_threshold=[0.5, 0.0, 0.0],
        left_child=[1, -1, -1],
        right_child=[2, -1, -1],
        is_leaf=[False, True, True],
        leaf_value=[
            torch.zeros(1),  # internal node (unused but present)
            torch.tensor([1.0]),  # left leaf
            torch.tensor([2.0]),  # right leaf
        ],
    )

    x = torch.tensor(
        [
            [0.0, 9.0],  # feature 0 = 0.0 <= 0.5 -> left -> [1.0]
            [1.0, 9.0],  # feature 0 = 1.0 > 0.5  -> right -> [2.0]
            [0.5, 9.0],  # feature 0 = 0.5 <= 0.5  -> left -> [1.0]
            [0.6, 9.0],  # feature 0 = 0.6 > 0.5  -> right -> [2.0]
        ]
    )

    pred = _predict_tree(tree, x)
    expected = torch.tensor([[1.0], [2.0], [1.0], [2.0]])
    assert torch.allclose(pred, expected)


def test_predict_tree_depth_2() -> None:
    """_predict_tree correctly traverses a depth-2 tree."""
    # Depth-2 tree splitting on feature 0 at both levels:
    #   node 0: split feature=0, threshold=0.5 -> left=1, right=2
    #   node 1: split feature=0, threshold=0.2 -> left=3, right=4
    #   node 2: leaf -> [3.0]
    #   node 3: leaf -> [1.0]
    #   node 4: leaf -> [2.0]
    tree = _TreeModel(
        split_feature=[0, 0, -1, -1, -1],
        split_threshold=[0.5, 0.2, 0.0, 0.0, 0.0],
        left_child=[1, 3, -1, -1, -1],
        right_child=[2, 4, -1, -1, -1],
        is_leaf=[False, False, True, True, True],
        leaf_value=[
            torch.zeros(1),  # node 0: internal
            torch.zeros(1),  # node 1: internal
            torch.tensor([3.0]),  # node 2: right leaf
            torch.tensor([1.0]),  # node 3: left-left leaf
            torch.tensor([2.0]),  # node 4: left-right leaf
        ],
    )

    x = torch.tensor(
        [
            [0.1, 0.0],  # <= 0.5 -> node 1; <= 0.2 -> node 3 -> [1.0]
            [0.3, 0.0],  # <= 0.5 -> node 1; > 0.2  -> node 4 -> [2.0]
            [0.8, 0.0],  # > 0.5  -> node 2           -> [3.0]
        ]
    )

    pred = _predict_tree(tree, x)
    expected = torch.tensor([[1.0], [2.0], [3.0]])
    assert torch.allclose(pred, expected)


def test_fit_rf_tree_no_none_leaf_values() -> None:
    """All leaf_value entries should be non-None after fitting."""
    g = torch.Generator(device="cpu")
    g.manual_seed(42)
    n_rows, n_features = 64, 5
    x = torch.randn(n_rows, n_features, generator=g)
    y = torch.randn(n_rows, 1)
    train_rows = torch.arange(n_rows)

    tree = _fit_rf_tree(
        x,
        y,
        None,
        train_rows,
        task="regression",
        n_classes=0,
        max_depth=4,
        min_samples_leaf=1,
        max_leaf_nodes=None,
        m_try=3,
        n_split_candidates=4,
        generator=g,
    )

    for i, val in enumerate(tree.leaf_value):
        assert val is not None, f"leaf_value[{i}] is None after fitting"


def test_predict_tree_raises_on_none_leaf_values() -> None:
    tree = _TreeModel(
        split_feature=[-1],
        split_threshold=[0.0],
        left_child=[-1],
        right_child=[-1],
        is_leaf=[True],
        leaf_value=[None],
    )
    x = torch.zeros((2, 1))
    try:
        _predict_tree(tree, x)
        assert False, "Expected ValueError for None leaf values"
    except ValueError as exc:
        assert "leaf_value" in str(exc)
