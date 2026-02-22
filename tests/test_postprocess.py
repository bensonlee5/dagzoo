"""Tests for postprocess/postprocess.py — Appendix E.13."""

import numpy as np

from cauchy_generator.postprocess.postprocess import postprocess_dataset


def _make_data(
    rng: np.random.Generator,
    n_train: int = 100,
    n_test: int = 50,
    n_feat: int = 5,
    n_classes: int = 3,
    task: str = "classification",
    add_constant_col: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], str]:
    x_train = rng.normal(size=(n_train, n_feat)).astype(np.float32)
    x_test = rng.normal(size=(n_test, n_feat)).astype(np.float32)
    if add_constant_col:
        x_train[:, -1] = 1.0
        x_test[:, -1] = 1.0
    ftypes = ["num"] * n_feat

    if task == "classification":
        y_train = rng.integers(0, n_classes, size=n_train).astype(np.int64)
        y_test = rng.integers(0, n_classes, size=n_test).astype(np.int64)
    else:
        y_train = rng.normal(size=n_train).astype(np.float32)
        y_test = rng.normal(size=n_test).astype(np.float32)
    return x_train, y_train, x_test, y_test, ftypes, task


def test_removes_constant_columns() -> None:
    rng = np.random.default_rng(0)
    xt, yt, xte, yte, ft, task = _make_data(rng, n_feat=4, add_constant_col=True)
    xtp, _, xtep, _, ft_out = postprocess_dataset(xt, yt, xte, yte, ft, task, rng)
    assert xtp.shape[1] < 4
    assert len(ft_out) == xtp.shape[1]


def test_standardizes_numeric() -> None:
    rng = np.random.default_rng(1)
    xt, yt, xte, yte, ft, task = _make_data(rng)
    xtp, _, xtep, _, _ = postprocess_dataset(xt, yt, xte, yte, ft, task, rng)
    combined = np.concatenate([xtp, xtep], axis=0)
    for i in range(combined.shape[1]):
        col = combined[:, i]
        assert abs(float(np.mean(col))) < 0.15
        assert float(np.std(col)) < 1.5


def test_preserves_class_counts() -> None:
    rng = np.random.default_rng(2)
    xt, yt, xte, yte, ft, task = _make_data(rng, n_classes=4)
    _, ytp, _, ytep, _ = postprocess_dataset(xt, yt, xte, yte, ft, task, rng)
    y_all_before = np.concatenate([yt, yte])
    y_all_after = np.concatenate([ytp, ytep])
    # Same total count for each class (just remapped)
    before_counts = sorted(np.bincount(y_all_before).tolist())
    after_counts = sorted(np.bincount(y_all_after).tolist())
    assert before_counts == after_counts


def test_regression_clips_targets() -> None:
    rng = np.random.default_rng(3)
    xt, yt, xte, yte, ft, task = _make_data(rng, task="regression")
    # Add extreme outlier
    yt[0] = 1e6
    _, ytp, _, ytep, _ = postprocess_dataset(xt, yt, xte, yte, ft, task, rng)
    y_all = np.concatenate([ytp, ytep])
    assert np.all(np.isfinite(y_all))


def test_deterministic() -> None:
    rng1 = np.random.default_rng(99)
    xt, yt, xte, yte, ft, task = _make_data(rng1)
    out1 = postprocess_dataset(
        xt.copy(), yt.copy(), xte.copy(), yte.copy(), list(ft), task, np.random.default_rng(0)
    )

    out2 = postprocess_dataset(
        xt.copy(), yt.copy(), xte.copy(), yte.copy(), list(ft), task, np.random.default_rng(0)
    )

    np.testing.assert_array_equal(out1[0], out2[0])
    np.testing.assert_array_equal(out1[1], out2[1])
