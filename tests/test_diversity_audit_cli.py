import json

from dagzoo.cli import main


def test_diversity_audit_cli_local_phase_writes_artifacts(tmp_path) -> None:
    out_dir = tmp_path / "diversity_local"
    code = main(
        [
            "diversity-audit",
            "--phase",
            "local",
            "--n-seeds",
            "2",
            "--n-rows",
            "128",
            "--n-cols",
            "8",
            "--out-dim",
            "8",
            "--nn-degenerate-trials",
            "1000",
            "--out-dir",
            str(out_dir),
        ]
    )

    assert code == 0
    run_summary_path = out_dir / "run_summary.json"
    equivalence_path = out_dir / "equivalence_report.json"
    assert run_summary_path.exists()
    assert equivalence_path.exists()

    payload = json.loads(run_summary_path.read_text(encoding="utf-8"))
    assert payload["phase"] == "local"
    assert isinstance(payload["local_report"], dict)
    assert payload["scale_report"] is None
