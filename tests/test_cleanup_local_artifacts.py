from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_cleanup_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "cleanup_local_artifacts.py"
    spec = importlib.util.spec_from_file_location("cleanup_local_artifacts", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cleanup_local_artifacts_dry_run_does_not_delete(
    tmp_path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_cleanup_module()
    (tmp_path / "data").mkdir()
    (tmp_path / "benchmarks" / "results").mkdir(parents=True)
    (tmp_path / "public").mkdir()

    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setitem(
        module.TARGET_GROUPS,
        "runtime",
        (
            tmp_path / "data",
            tmp_path / "benchmarks" / "results",
            tmp_path / "effective_config_artifacts",
        ),
    )
    monkeypatch.setitem(
        module.TARGET_GROUPS,
        "docs",
        (
            tmp_path / "public",
            tmp_path / "site" / "public",
            tmp_path / "site" / ".generated",
        ),
    )

    code = module.main(["--group", "all"])

    assert code == 0
    assert (tmp_path / "data").exists()
    assert (tmp_path / "benchmarks" / "results").exists()
    assert (tmp_path / "public").exists()
    captured = capsys.readouterr()
    assert "Would remove: data" in captured.out
    assert "Dry run only." in captured.out


def test_cleanup_local_artifacts_apply_runtime_removes_only_runtime_paths(
    tmp_path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_cleanup_module()
    (tmp_path / "data").mkdir()
    (tmp_path / "benchmarks" / "results").mkdir(parents=True)
    (tmp_path / "effective_config_artifacts").mkdir()
    (tmp_path / "site" / "public").mkdir(parents=True)
    (tmp_path / "keep_me").mkdir()

    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setitem(
        module.TARGET_GROUPS,
        "runtime",
        (
            tmp_path / "data",
            tmp_path / "benchmarks" / "results",
            tmp_path / "effective_config_artifacts",
        ),
    )
    monkeypatch.setitem(
        module.TARGET_GROUPS,
        "docs",
        (
            tmp_path / "public",
            tmp_path / "site" / "public",
            tmp_path / "site" / ".generated",
        ),
    )

    code = module.main(["--group", "runtime", "--apply"])

    assert code == 0
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "benchmarks" / "results").exists()
    assert not (tmp_path / "effective_config_artifacts").exists()
    assert (tmp_path / "site" / "public").exists()
    assert (tmp_path / "keep_me").exists()
    captured = capsys.readouterr()
    assert "Removed: data" in captured.out


def test_cleanup_local_artifacts_apply_docs_removes_only_docs_paths(
    tmp_path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_cleanup_module()
    (tmp_path / "public").mkdir()
    (tmp_path / "site" / "public").mkdir(parents=True)
    (tmp_path / "site" / ".generated").mkdir(parents=True)
    (tmp_path / "data").mkdir()

    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setitem(
        module.TARGET_GROUPS,
        "runtime",
        (
            tmp_path / "data",
            tmp_path / "benchmarks" / "results",
            tmp_path / "effective_config_artifacts",
        ),
    )
    monkeypatch.setitem(
        module.TARGET_GROUPS,
        "docs",
        (
            tmp_path / "public",
            tmp_path / "site" / "public",
            tmp_path / "site" / ".generated",
        ),
    )

    code = module.main(["--group", "docs", "--apply"])

    assert code == 0
    assert not (tmp_path / "public").exists()
    assert not (tmp_path / "site" / "public").exists()
    assert not (tmp_path / "site" / ".generated").exists()
    assert (tmp_path / "data").exists()
    captured = capsys.readouterr()
    assert "Removed: public" in captured.out
