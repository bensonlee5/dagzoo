from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import yaml

from cauchy_generator.config import GeneratorConfig


def _write_tiny_config(path: Path) -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.runtime.device = "cpu"
    cfg.filter.enabled = False
    cfg.dataset.task = "classification"
    cfg.dataset.n_features_min = 8
    cfg.dataset.n_features_max = 8
    cfg.dataset.n_classes_min = 2
    cfg.dataset.n_classes_max = 2
    cfg.graph.n_nodes_min = 2
    cfg.graph.n_nodes_max = 4
    path.write_text(yaml.safe_dump(cfg.to_dict()), encoding="utf-8")


def test_curriculum_shell_dry_run_range(tmp_path: Path) -> None:
    config_path = tmp_path / "tiny.yaml"
    _write_tiny_config(config_path)
    out_root = tmp_path / "out"

    proc = subprocess.run(
        [
            "bash",
            "scripts/generate-curriculum.sh",
            "--base-config",
            str(config_path),
            "--out-root",
            str(out_root),
            "--datasets-per-stage",
            "2",
            "--n-test",
            "2",
            "--train-start",
            "8",
            "--train-stop",
            "10",
            "--train-step",
            "1",
            "--chunk-size",
            "1",
            "--n-features",
            "8",
            "--no-write",
            "--dry-run",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.count("DRY RUN:") == 6
    assert "cauchy-gen generate" in proc.stdout

    manifest_path = out_root / "curriculum_manifest.json"
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["status"] == "success"
    assert payload["train_rows"] == [8, 9, 10]
    assert payload["fixed_n_features"] == 8
    assert payload["total_generated_datasets"] == 0
    assert [stage["status"] for stage in payload["stages"]] == ["dry_run", "dry_run", "dry_run"]
    assert all(stage["generated_datasets"] == 0 for stage in payload["stages"])


def test_curriculum_shell_rejects_stage_columns_length_mismatch(tmp_path: Path) -> None:
    config_path = tmp_path / "tiny.yaml"
    _write_tiny_config(config_path)

    proc = subprocess.run(
        [
            "bash",
            "scripts/generate-curriculum.sh",
            "--base-config",
            str(config_path),
            "--out-root",
            str(tmp_path / "out"),
            "--datasets-per-stage",
            "1",
            "--n-test",
            "2",
            "--train-values",
            "8,9,10",
            "--stage-columns",
            "8,9",
            "--dry-run",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        text=True,
        capture_output=True,
    )

    assert proc.returncode != 0
    assert "must match stage rows count" in proc.stderr


def test_curriculum_shell_no_write_smoke(tmp_path: Path) -> None:
    config_path = tmp_path / "tiny.yaml"
    _write_tiny_config(config_path)
    out_root = tmp_path / "out"

    proc = subprocess.run(
        [
            "bash",
            "scripts/generate-curriculum.sh",
            "--base-config",
            str(config_path),
            "--out-root",
            str(out_root),
            "--datasets-per-stage",
            "1",
            "--n-test",
            "2",
            "--train-values",
            "8,9",
            "--stage-columns",
            "8,9",
            "--chunk-size",
            "1",
            "--device",
            "cpu",
            "--no-hardware-aware",
            "--no-write",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stderr
    manifest_path = out_root / "curriculum_manifest.json"
    assert manifest_path.exists()

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["status"] == "success"
    assert payload["total_generated_datasets"] == 2
    assert len(payload["stages"]) == 2
    assert payload["stage_columns"] == [8, 9]


def test_curriculum_shell_marks_unexecuted_stages_skipped_after_failure(tmp_path: Path) -> None:
    config_path = tmp_path / "tiny.yaml"
    _write_tiny_config(config_path)
    out_root = tmp_path / "out"
    fake_gen = tmp_path / "fake-cauchy-gen.sh"
    fake_gen.write_text("#!/usr/bin/env bash\nexit 1\n", encoding="utf-8")
    fake_gen.chmod(0o755)

    proc = subprocess.run(
        [
            "bash",
            "scripts/generate-curriculum.sh",
            "--base-config",
            str(config_path),
            "--out-root",
            str(out_root),
            "--datasets-per-stage",
            "1",
            "--n-test",
            "2",
            "--train-values",
            "8,9,10",
            "--chunk-size",
            "1",
            "--no-write",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        text=True,
        capture_output=True,
        env={**os.environ, "CURRICULUM_CAUCHY_GEN_BIN": str(fake_gen)},
    )

    assert proc.returncode != 0
    manifest_path = out_root / "curriculum_manifest.json"
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert payload["status"] == "failed"
    assert payload["total_generated_datasets"] == 0
    assert len(payload["stages"]) == 3

    first_stage = payload["stages"][0]
    assert first_stage["status"] == "failed"
    assert first_stage["generated_datasets"] == 0
    assert len(first_stage["chunks"]) == 1

    for stage in payload["stages"][1:]:
        assert stage["status"] == "skipped"
        assert stage["generated_datasets"] == 0
        assert stage["chunks"] == []
