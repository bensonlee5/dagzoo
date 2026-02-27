"""Torch-native random-forest filtering for learnability checks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


_CLASS_AWARE_THRESHOLD_POLICY = "class_aware_piecewise_v1"
_CLASS_AWARE_THRESHOLD_FLOOR = 0.80


@dataclass(slots=True)
class _TreeModel:
    split_feature: list[int]
    split_threshold: list[float]
    left_child: list[int]
    right_child: list[int]
    is_leaf: list[bool]
    leaf_value: list[torch.Tensor | None]


def _new_tree_node(tree: _TreeModel) -> int:
    tree.split_feature.append(-1)
    tree.split_threshold.append(0.0)
    tree.left_child.append(-1)
    tree.right_child.append(-1)
    tree.is_leaf.append(True)
    tree.leaf_value.append(None)
    return len(tree.is_leaf) - 1


def _resolve_max_features(max_features: str | int | float, n_features: int, task: str) -> int:
    if isinstance(max_features, str):
        key = max_features.lower()
        if key == "auto":
            value = (
                int(math.sqrt(n_features)) if task == "classification" else max(1, n_features // 3)
            )
        elif key == "sqrt":
            value = int(math.sqrt(n_features))
        elif key == "log2":
            value = int(math.log2(max(2, n_features)))
        elif key in {"all", "none"}:
            value = n_features
        else:
            raise ValueError(
                f"Unsupported max_features='{max_features}'. "
                "Expected one of: auto, sqrt, log2, all, none."
            )
    elif isinstance(max_features, int):
        value = int(max_features)
    elif isinstance(max_features, float):
        if not (0.0 < max_features <= 1.0):
            raise ValueError(f"Float max_features must be in (0, 1], got {max_features}")
        value = int(round(max_features * n_features))
    else:
        raise TypeError(
            f"max_features must be str, int, or float, got {type(max_features).__name__}"
        )
    return max(1, min(n_features, value))


def _regression_impurity(y: torch.Tensor) -> float:
    if y.shape[0] <= 1:
        return 0.0
    mu = torch.mean(y, dim=0, keepdim=True)
    return float(torch.mean((y - mu) ** 2).item())


def _gini_impurity(y_cls: torch.Tensor, n_classes: int) -> float:
    if y_cls.numel() <= 1:
        return 0.0
    counts = torch.bincount(y_cls, minlength=n_classes).to(torch.float32)
    probs = counts / torch.clamp(torch.sum(counts), min=1.0)
    return float((1.0 - torch.sum(probs * probs)).item())


def _find_best_random_split(
    x: torch.Tensor,
    y_target: torch.Tensor,
    y_cls: torch.Tensor | None,
    row_idx: torch.Tensor,
    *,
    task: str,
    n_classes: int,
    m_try: int,
    n_split_candidates: int,
    min_samples_leaf: int,
    generator: torch.Generator,
) -> tuple[int, float, torch.Tensor, torch.Tensor] | None:
    n_node = int(row_idx.numel())
    if n_node < (2 * min_samples_leaf):
        return None

    n_features = x.shape[1]
    device = x.device
    node_x = x[row_idx]

    if task == "classification":
        assert y_cls is not None
        parent_impurity = _gini_impurity(y_cls[row_idx], n_classes=n_classes)
    else:
        parent_impurity = _regression_impurity(y_target[row_idx])
    if parent_impurity <= 1e-12:
        return None

    feature_order = torch.randperm(n_features, generator=generator, device=device)[:m_try]
    best_gain = 0.0
    best_split: tuple[int, float, torch.Tensor, torch.Tensor] | None = None

    for feature in feature_order.tolist():
        col = node_x[:, feature]
        col_min = float(torch.min(col).item())
        col_max = float(torch.max(col).item())
        if (not math.isfinite(col_min)) or (not math.isfinite(col_max)) or col_min == col_max:
            continue

        num_candidates = min(max(1, int(n_split_candidates)), n_node)
        cand_idx = torch.randint(
            0,
            n_node,
            (num_candidates,),
            generator=generator,
            device=device,
        )
        thresholds = torch.unique(col[cand_idx])
        for threshold_tensor in thresholds:
            threshold = float(threshold_tensor.item())
            left_mask = col <= threshold
            left_n = int(left_mask.sum().item())
            right_n = n_node - left_n
            if left_n < min_samples_leaf or right_n < min_samples_leaf:
                continue

            left_rows = row_idx[left_mask]
            right_rows = row_idx[~left_mask]

            if task == "classification":
                assert y_cls is not None
                impurity_left = _gini_impurity(y_cls[left_rows], n_classes=n_classes)
                impurity_right = _gini_impurity(y_cls[right_rows], n_classes=n_classes)
            else:
                impurity_left = _regression_impurity(y_target[left_rows])
                impurity_right = _regression_impurity(y_target[right_rows])

            weighted_child_impurity = (left_n / n_node) * impurity_left + (
                right_n / n_node
            ) * impurity_right
            gain = parent_impurity - weighted_child_impurity
            if gain > best_gain:
                best_gain = gain
                best_split = (feature, threshold, left_rows, right_rows)

    return best_split


def _fit_rf_tree(
    x: torch.Tensor,
    y_target: torch.Tensor,
    y_cls: torch.Tensor | None,
    train_rows: torch.Tensor,
    *,
    task: str,
    n_classes: int,
    max_depth: int,
    min_samples_leaf: int,
    max_leaf_nodes: int | None,
    m_try: int,
    n_split_candidates: int,
    generator: torch.Generator,
) -> _TreeModel:
    tree = _TreeModel(
        split_feature=[],
        split_threshold=[],
        left_child=[],
        right_child=[],
        is_leaf=[],
        leaf_value=[],
    )
    root_id = _new_tree_node(tree)
    work_stack: list[tuple[int, torch.Tensor, int]] = [(root_id, train_rows, 0)]
    splits_done = 0
    target_dim = y_target.shape[1]

    while work_stack:
        node_id, rows, node_depth = work_stack.pop()
        node_target = y_target[rows]
        node_mean = torch.mean(node_target, dim=0)
        can_split = (
            node_depth < max_depth
            and int(rows.numel()) >= (2 * min_samples_leaf)
            and (max_leaf_nodes is None or (splits_done + 1) < max_leaf_nodes)
        )
        if not can_split:
            tree.is_leaf[node_id] = True
            tree.leaf_value[node_id] = node_mean
            continue

        best_split = _find_best_random_split(
            x,
            y_target,
            y_cls,
            rows,
            task=task,
            n_classes=n_classes,
            m_try=m_try,
            n_split_candidates=n_split_candidates,
            min_samples_leaf=min_samples_leaf,
            generator=generator,
        )
        if best_split is None:
            tree.is_leaf[node_id] = True
            tree.leaf_value[node_id] = node_mean
            continue

        feature, threshold, left_rows, right_rows = best_split
        left_id = _new_tree_node(tree)
        right_id = _new_tree_node(tree)
        tree.is_leaf[node_id] = False
        tree.split_feature[node_id] = int(feature)
        tree.split_threshold[node_id] = float(threshold)
        tree.left_child[node_id] = left_id
        tree.right_child[node_id] = right_id
        tree.leaf_value[node_id] = torch.zeros(target_dim, device=x.device, dtype=torch.float32)
        splits_done += 1

        work_stack.append((right_id, right_rows, node_depth + 1))
        work_stack.append((left_id, left_rows, node_depth + 1))

    for i, leaf_value in enumerate(tree.leaf_value):
        if leaf_value is None:
            tree.leaf_value[i] = torch.zeros(target_dim, device=x.device, dtype=torch.float32)
    return tree


def _predict_tree(tree: _TreeModel, x: torch.Tensor) -> torch.Tensor:
    """
    Predict tree outputs for all rows.

    This traversal uses a Python loop over active node ids; it is appropriate for
    shallow trees used by default filter settings.
    """

    n_rows = x.shape[0]
    device = x.device
    node_ids = torch.zeros(n_rows, dtype=torch.long, device=device)

    while True:
        unique_ids = torch.unique(node_ids)
        progressed = False
        for node_tensor in unique_ids:
            node_id = int(node_tensor.item())
            if tree.is_leaf[node_id]:
                continue
            progressed = True
            row_mask = node_ids == node_id
            row_idx = torch.where(row_mask)[0]
            if row_idx.numel() == 0:
                continue
            feature = tree.split_feature[node_id]
            threshold = tree.split_threshold[node_id]
            left_node = tree.left_child[node_id]
            right_node = tree.right_child[node_id]
            left_mask = x[row_idx, feature] <= threshold
            node_ids[row_idx[left_mask]] = left_node
            node_ids[row_idx[~left_mask]] = right_node
        if not progressed:
            break

    if not all(v is not None for v in tree.leaf_value):
        raise ValueError("All leaf_value entries must be non-None before prediction")
    leaf_values = torch.stack(tree.leaf_value, dim=0)  # type: ignore[arg-type]
    return leaf_values[node_ids]


def _resolve_threshold_diagnostics(
    *,
    task: str,
    requested_threshold: float,
    class_count: int | None,
) -> dict[str, Any]:
    """Return class-aware threshold diagnostics and effective threshold."""

    if task != "classification" or class_count is None:
        effective_threshold = float(requested_threshold)
        return {
            "threshold_requested": float(requested_threshold),
            "threshold_effective": effective_threshold,
            "threshold_policy": _CLASS_AWARE_THRESHOLD_POLICY,
            "class_count": None,
            "class_bucket": "not_applicable",
            "threshold_delta": float(requested_threshold) - effective_threshold,
        }

    if class_count <= 8:
        class_bucket = "<=8"
        threshold_delta = 0.00
    elif class_count <= 16:
        class_bucket = "9-16"
        threshold_delta = 0.05
    elif class_count <= 24:
        class_bucket = "17-24"
        threshold_delta = 0.10
    else:
        class_bucket = "25-32"
        threshold_delta = 0.15

    relaxed_threshold = max(
        _CLASS_AWARE_THRESHOLD_FLOOR, float(requested_threshold) - threshold_delta
    )
    effective_threshold = min(float(requested_threshold), relaxed_threshold)
    return {
        "threshold_requested": float(requested_threshold),
        "threshold_effective": float(effective_threshold),
        "threshold_policy": _CLASS_AWARE_THRESHOLD_POLICY,
        "class_count": int(class_count),
        "class_bucket": class_bucket,
        "threshold_delta": float(requested_threshold) - float(effective_threshold),
    }


def apply_torch_rf_filter(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    task: str,
    seed: int,
    n_trees: int = 25,
    depth: int = 6,
    min_samples_leaf: int = 1,
    max_leaf_nodes: int | None = None,
    max_features: str | int | float = "auto",
    n_split_candidates: int = 8,
    n_bootstrap: int = 200,
    threshold: float = 0.95,
) -> tuple[bool, dict[str, Any]]:
    """Apply Torch RF-based filtering with OOB bootstrap win-ratio scoring."""

    if n_trees < 1:
        raise ValueError(f"n_trees must be >= 1, got {n_trees}")
    if depth < 0:
        raise ValueError(f"depth must be >= 0, got {depth}")
    if min_samples_leaf < 1:
        raise ValueError(f"min_samples_leaf must be >= 1, got {min_samples_leaf}")
    if max_leaf_nodes is not None and max_leaf_nodes < 1:
        raise ValueError(f"max_leaf_nodes must be >= 1 when set, got {max_leaf_nodes}")
    if n_split_candidates < 1:
        raise ValueError(f"n_split_candidates must be >= 1, got {n_split_candidates}")
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}")

    x_arr = x.to(torch.float32)
    device = x_arr.device
    n_rows = x_arr.shape[0]
    n_features = x_arr.shape[1]
    m_try = _resolve_max_features(max_features, n_features, task)

    if task == "classification":
        y_raw = y.to(torch.int64).view(-1)
        unique_labels, y_cls = torch.unique(y_raw, sorted=True, return_inverse=True)
        n_classes = int(unique_labels.numel())
        class_count: int | None = n_classes
        y_target = torch.nn.functional.one_hot(y_cls, num_classes=n_classes).to(torch.float32)
    else:
        y_cls = None
        n_classes = 0
        class_count = None
        y_target = y.to(torch.float32)
        if y_target.dim() == 1:
            y_target = y_target.unsqueeze(1)

    threshold_details = _resolve_threshold_diagnostics(
        task=task,
        requested_threshold=float(threshold),
        class_count=class_count,
    )
    effective_threshold = float(threshold_details["threshold_effective"])

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    oob_pred_sum = torch.zeros_like(y_target)
    oob_count = torch.zeros(n_rows, 1, device=device, dtype=torch.float32)

    for _ in range(n_trees):
        bootstrap_rows = torch.randint(0, n_rows, (n_rows,), generator=gen, device=device)
        in_bag_mask = torch.zeros(n_rows, dtype=torch.bool, device=device)
        in_bag_mask.scatter_(0, bootstrap_rows, True)
        oob_mask = ~in_bag_mask
        if int(oob_mask.sum().item()) == 0:
            continue

        tree = _fit_rf_tree(
            x_arr,
            y_target,
            y_cls,
            bootstrap_rows,
            task=task,
            n_classes=n_classes,
            max_depth=depth,
            min_samples_leaf=min_samples_leaf,
            max_leaf_nodes=max_leaf_nodes,
            m_try=m_try,
            n_split_candidates=n_split_candidates,
            generator=gen,
        )

        oob_rows = torch.where(oob_mask)[0]
        oob_pred = _predict_tree(tree, x_arr[oob_rows])
        oob_pred_sum[oob_rows] += oob_pred
        oob_count[oob_rows] += 1.0

    valid_oob = (oob_count[:, 0] > 0) & torch.isfinite(oob_pred_sum).all(dim=1)
    n_valid = int(valid_oob.sum().item())
    if n_valid < max(32, int(0.25 * n_rows)):
        return False, {
            "reason": "insufficient_oob_predictions",
            "n_valid_oob": n_valid,
            "backend": "torch_rf",
            **threshold_details,
        }

    pred = oob_pred_sum[valid_oob] / oob_count[valid_oob]
    target = y_target[valid_oob]
    baseline = torch.mean(target, dim=0, keepdim=True)

    wins = 0
    chunk_size = 16
    for start in range(0, n_bootstrap, chunk_size):
        bs = min(chunk_size, n_bootstrap - start)
        idx = torch.randint(0, n_valid, (bs, n_valid), generator=gen, device=device)
        sampled_target = target[idx]
        sampled_pred = pred[idx]
        mse_pred = ((sampled_pred - sampled_target) ** 2).mean(dim=(1, 2))
        mse_base = ((baseline - sampled_target) ** 2).mean(dim=(1, 2))
        wins += int((mse_pred < mse_base).sum().item())

    wins_ratio = wins / float(n_bootstrap)
    return bool(wins_ratio >= effective_threshold), {
        "wins_ratio": float(wins_ratio),
        "n_valid_oob": n_valid,
        "backend": "torch_rf",
        **threshold_details,
    }
