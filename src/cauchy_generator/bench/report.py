"""Benchmark suite artifact writers (JSON and Markdown)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from cauchy_generator.math_utils import sanitize_json as _sanitize_json


def write_suite_json(summary: dict[str, Any], out_path: str | Path) -> Path:
    """Write a suite summary as JSON."""

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_sanitize_json(summary), f, indent=2, sort_keys=True)
    return path


def _format_float(value: Any, digits: int = 3) -> str:
    """Render floats consistently for markdown tables."""

    if not isinstance(value, (int, float)):
        return "-"
    if not math.isfinite(float(value)):
        return "-"
    return f"{float(value):.{digits}f}"


def _build_profile_table(profile_results: list[dict[str, Any]]) -> list[str]:
    """Create a markdown table summarizing per-profile performance metrics."""

    lines = [
        "| Profile | Device | Backend | Datasets/min | Elapsed (s) | Latency p95 (ms) | Peak RSS (MB) | Diagnostics | Missingness | Lineage | Shift | Noise |",
        "|---|---|---:|---:|---:|---:|---:|---|---|---|---|---|",
    ]
    for result in profile_results:
        diagnostics_state = "on" if bool(result.get("diagnostics_enabled")) else "off"
        missingness_state = "off"
        guardrails = result.get("missingness_guardrails")
        if isinstance(guardrails, dict) and bool(guardrails.get("enabled")):
            missingness_state = str(guardrails.get("status", "pass"))
        lineage_state = "off"
        lineage_guardrails = result.get("lineage_guardrails")
        if isinstance(lineage_guardrails, dict) and bool(lineage_guardrails.get("enabled")):
            lineage_state = str(lineage_guardrails.get("status", "pass"))
        shift_state = "off"
        shift_guardrails = result.get("shift_guardrails")
        if isinstance(shift_guardrails, dict) and bool(shift_guardrails.get("enabled")):
            shift_state = str(shift_guardrails.get("status", "pass"))
        noise_state = "off"
        noise_guardrails = result.get("noise_guardrails")
        if isinstance(noise_guardrails, dict) and bool(noise_guardrails.get("enabled")):
            noise_state = str(noise_guardrails.get("status", "pass"))
        lines.append(
            "| "
            f"{result.get('profile_key', '-')} | "
            f"{result.get('device', '-')} | "
            f"{result.get('hardware_backend', '-')} | "
            f"{_format_float(result.get('datasets_per_minute'), 2)} | "
            f"{_format_float(result.get('elapsed_seconds'), 3)} | "
            f"{_format_float(result.get('latency_p95_ms'), 2)} | "
            f"{_format_float(result.get('peak_rss_mb'), 2)} | "
            f"{diagnostics_state} |"
            f" {missingness_state} |"
            f" {lineage_state} |"
            f" {shift_state} |"
            f" {noise_state} |"
        )
    return lines


def _build_diagnostics_table(profile_results: list[dict[str, Any]]) -> list[str]:
    """Create a markdown table with per-profile diagnostics artifact pointers."""

    lines = [
        "| Profile | Coverage JSON | Coverage Markdown |",
        "|---|---|---|",
    ]
    for result in profile_results:
        artifacts = result.get("diagnostics_artifacts")
        json_path = "-"
        md_path = "-"
        if isinstance(artifacts, dict):
            json_value = artifacts.get("json")
            md_value = artifacts.get("markdown")
            if isinstance(json_value, str) and json_value:
                json_path = f"`{json_value}`"
            if isinstance(md_value, str) and md_value:
                md_path = f"`{md_value}`"
        lines.append(f"| {result.get('profile_key', '-')} | {json_path} | {md_path} |")
    return lines


def write_suite_markdown(summary: dict[str, Any], out_path: str | Path) -> Path:
    """Write a concise markdown report for one benchmark suite run."""

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# Benchmark Suite Report",
        "",
        f"- Suite: `{summary.get('suite', '-')}`",
        f"- Generated at: `{summary.get('generated_at', '-')}`",
    ]

    regression = summary.get("regression", {})
    if isinstance(regression, dict):
        lines.append(f"- Regression status: `{regression.get('status', 'pass')}`")
    lines.append("")

    profile_results = summary.get("profile_results", [])
    if isinstance(profile_results, list) and profile_results:
        lines.append("## Profiles")
        lines.extend(_build_profile_table(profile_results))
        lines.append("")
        if any(bool(result.get("diagnostics_enabled")) for result in profile_results):
            lines.append("## Diagnostics Artifacts")
            lines.extend(_build_diagnostics_table(profile_results))
            lines.append("")

    if isinstance(regression, dict) and regression.get("issues"):
        lines.append("## Regression Issues")
        lines.append("| Severity | Profile | Metric | Current | Baseline | Degradation % |")
        lines.append("|---|---|---|---:|---:|---:|")
        for issue in regression["issues"]:
            lines.append(
                "| "
                f"{issue.get('severity')} | "
                f"{issue.get('profile')} | "
                f"{issue.get('metric')} | "
                f"{_format_float(issue.get('current'), 3)} | "
                f"{_format_float(issue.get('baseline'), 3)} | "
                f"{_format_float(issue.get('degradation_pct'), 2)} |"
            )
        lines.append("")

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")
    return path
