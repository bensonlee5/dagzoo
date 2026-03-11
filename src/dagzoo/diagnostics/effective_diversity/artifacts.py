"""Artifact formatting and writing for diversity-audit reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dagzoo.math_utils import sanitize_json

from .calibration import format_filter_calibration_threshold


def _fmt(value: object, *, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def format_effective_diversity_markdown(report: dict[str, Any]) -> str:
    """Render a concise markdown summary for the diversity audit."""

    summary = report.get("summary", {})
    baseline = report.get("baseline", {})
    variants = report.get("variants", [])
    comparisons = report.get("comparisons", [])

    lines = [
        "# Diversity Audit",
        "",
        f"- Overall status: `{summary.get('overall_status', 'insufficient_metrics')}`",
        f"- Warn threshold pct: `{_fmt(summary.get('warn_threshold_pct'))}`",
        f"- Fail threshold pct: `{_fmt(summary.get('fail_threshold_pct'))}`",
        f"- Num variants: `{_fmt(summary.get('num_variants'), digits=0)}`",
        f"- Probe num datasets: `{_fmt(summary.get('probe_num_datasets'), digits=0)}`",
        f"- Probe warmup datasets: `{_fmt(summary.get('probe_warmup_datasets'), digits=0)}`",
        "",
        "## Baseline",
        "",
        f"- Label: `{baseline.get('label', '-')}`",
        f"- Config path: `{baseline.get('config_path', '-')}`",
        f"- Datasets/min: `{_fmt(baseline.get('datasets_per_minute'))}`",
        f"- Filter accepted/min: `{_fmt(baseline.get('filter_accepted_datasets_per_minute'))}`",
        f"- Filter accept dataset pct: `{_fmt(_pct(baseline.get('filter_acceptance_rate_dataset_level')))}`",
        "",
        "## Variants",
        "",
        "| Variant | Status | Composite Shift % | Datasets/min Δ% | Filter Accepted/min Δ% |",
        "|---|---|---:|---:|---:|",
    ]

    comparison_map = {
        str(item.get("variant_label")): item
        for item in comparisons
        if isinstance(item, dict) and item.get("variant_label") is not None
    }
    for variant in variants if isinstance(variants, list) else []:
        if not isinstance(variant, dict):
            continue
        label = str(variant.get("label", "-"))
        comparison = comparison_map.get(label, {})
        lines.append(
            "| "
            f"{label} | "
            f"{comparison.get('diversity_status', 'insufficient_metrics')} | "
            f"{_fmt(comparison.get('diversity_composite_shift_pct'))} | "
            f"{_fmt(comparison.get('datasets_per_minute_delta_pct'))} | "
            f"{_fmt(comparison.get('filter_accepted_datasets_per_minute_delta_pct'))} |"
        )

    lines.extend(["", "## Notes", ""])
    lines.append(
        "- This report compares accepted-corpus diagnostics coverage against the baseline config."
    )
    lines.append(
        "- `summary.json` / `summary.md` are the canonical persisted artifacts for the rewritten audit."
    )
    return "\n".join(lines).rstrip() + "\n"


def _pct(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) * 100.0


def write_effective_diversity_artifacts(
    report: dict[str, Any],
    *,
    out_dir: str | Path,
) -> dict[str, Path]:
    """Write diversity-audit report artifacts to disk."""

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    summary_json = out_path / "summary.json"
    summary_md = out_path / "summary.md"
    summary_json.write_text(
        json.dumps(sanitize_json(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary_md.write_text(format_effective_diversity_markdown(report), encoding="utf-8")
    return {
        "summary_json": summary_json,
        "summary_md": summary_md,
    }


def format_filter_calibration_markdown(report: dict[str, Any]) -> str:
    """Render a concise markdown summary for one filter-calibration run."""

    summary = report.get("summary", {})
    baseline = report.get("baseline", {})
    candidates = report.get("candidates", [])

    lines = [
        "# Filter Calibration",
        "",
        f"- Overall status: `{summary.get('overall_status', 'insufficient_metrics')}`",
        f"- Baseline threshold: `{format_filter_calibration_threshold(summary.get('baseline_threshold_requested'))}`",
        f"- Best overall threshold: `{format_filter_calibration_threshold(summary.get('best_overall_threshold_requested'))}`",
        f"- Best overall status: `{summary.get('best_overall_diversity_status', '-')}`",
        f"- Best passing threshold: `{format_filter_calibration_threshold(summary.get('best_passing_threshold_requested'))}`",
        f"- Num candidates: `{_fmt(summary.get('num_candidates'), digits=0)}`",
        f"- Probe num datasets: `{_fmt(summary.get('probe_num_datasets'), digits=0)}`",
        f"- Probe warmup datasets: `{_fmt(summary.get('probe_warmup_datasets'), digits=0)}`",
        "",
        "## Baseline",
        "",
        f"- Threshold requested: `{format_filter_calibration_threshold(baseline.get('threshold_requested'))}`",
        f"- Filter accepted/min: `{_fmt(baseline.get('filter_accepted_datasets_per_minute'))}`",
        f"- Filter accept dataset pct: `{_fmt(_pct(baseline.get('filter_acceptance_rate_dataset_level')))}`",
        "",
        "## Candidates",
        "",
        "| Candidate | Threshold | Status | Filter Accepted/min | Accept % | Composite Shift % |",
        "|---|---:|---|---:|---:|---:|",
    ]

    for candidate in candidates if isinstance(candidates, list) else []:
        if not isinstance(candidate, dict):
            continue
        lines.append(
            "| "
            f"{candidate.get('label', '-')} | "
            f"{format_filter_calibration_threshold(candidate.get('threshold_requested'))} | "
            f"{candidate.get('diversity_status', 'insufficient_metrics')} | "
            f"{_fmt(candidate.get('filter_accepted_datasets_per_minute'))} | "
            f"{_fmt(_pct(candidate.get('filter_acceptance_rate_dataset_level')))} | "
            f"{_fmt(candidate.get('diversity_composite_shift_pct'))} |"
        )

    lines.extend(["", "## Notes", ""])
    lines.append("- Use this report to balance accepted-corpus throughput against diversity shift.")
    lines.append(
        "- `summary.json` / `summary.md` are the canonical persisted artifacts for filter calibration."
    )
    return "\n".join(lines).rstrip() + "\n"


def write_filter_calibration_artifacts(
    report: dict[str, Any],
    *,
    out_dir: str | Path,
) -> dict[str, Path]:
    """Write filter-calibration artifacts to disk."""

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    summary_json = out_path / "summary.json"
    summary_md = out_path / "summary.md"
    summary_json.write_text(
        json.dumps(sanitize_json(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary_md.write_text(format_filter_calibration_markdown(report), encoding="utf-8")
    return {
        "summary_json": summary_json,
        "summary_md": summary_md,
    }


def format_effective_diversity_run_markdown(report: dict[str, Any]) -> str:
    """Backward-compatible alias for the rewritten audit markdown renderer."""

    return format_effective_diversity_markdown(report)


def write_effective_diversity_run_artifacts(
    report: dict[str, Any],
    *,
    out_dir: str | Path,
) -> dict[str, Path]:
    """Backward-compatible alias for the rewritten audit artifact writer."""

    return write_effective_diversity_artifacts(report, out_dir=out_dir)
