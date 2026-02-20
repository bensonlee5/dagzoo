"""ExtraTrees-based filtering (Appendix E.14)."""

from __future__ import annotations

import numpy as np

from sklearn.ensemble import ExtraTreesRegressor


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    """Compute mean squared error between aligned arrays."""

    diff = a - b
    return float(np.mean(diff * diff))


def apply_extratrees_filter(
    x: np.ndarray,
    y: np.ndarray,
    *,
    task: str,
    seed: int,
    n_estimators: int = 25,
    max_depth: int = 6,
    n_bootstrap: int = 200,
    threshold: float = 0.95,
) -> tuple[bool, dict[str, float | int | str]]:
    """
    Apply E.14-style OOB ExtraTrees filtering.

    Returns (accepted, details).
    """

    x_arr = np.asarray(x, dtype=np.float32)
    if task == "classification":
        y_int = np.asarray(y, dtype=np.int64)
        n_classes = int(np.max(y_int)) + 1
        y_target = np.eye(n_classes, dtype=np.float32)[y_int]
    else:
        y_target = np.asarray(y, dtype=np.float32)
        if y_target.ndim == 1:
            y_target = y_target[:, None]

    model = ExtraTreesRegressor(
        n_estimators=n_estimators,
        bootstrap=True,
        max_depth=max_depth,
        oob_score=True,
        random_state=seed,
        n_jobs=1,
    )

    model.fit(x_arr, y_target)
    pred = np.asarray(model.oob_prediction_)
    if pred.ndim == 1:
        pred = pred[:, None]
    target = y_target if y_target.ndim == 2 else y_target[:, None]

    valid = np.isfinite(pred).all(axis=1)
    if valid.sum() < max(32, int(0.25 * len(valid))):
        return False, {"reason": "insufficient_oob_predictions"}

    pred = pred[valid]
    target = target[valid]
    baseline = np.mean(target, axis=0, keepdims=True)

    rng = np.random.default_rng(seed + 13)
    n = pred.shape[0]
    wins = 0
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        mse_pred = _mse(pred[idx], target[idx])
        mse_base = _mse(np.repeat(baseline, len(idx), axis=0), target[idx])
        if mse_pred < mse_base:
            wins += 1

    score = wins / float(n_bootstrap)
    return score >= threshold, {"wins_ratio": score, "n_valid_oob": int(n)}
