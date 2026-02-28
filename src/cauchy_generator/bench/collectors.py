"""Per-bundle collector helpers for benchmark diagnostics and guardrails."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from typing import Any

from cauchy_generator.types import DatasetBundle

from .constants import (
    CURRICULUM_STAGE_VALUES,
    MISSINGNESS_RATE_FAIL_ABS_ERROR,
    MISSINGNESS_RATE_WARN_ABS_ERROR,
)
from .guardrails import (
    _severity_from_thresholds,
    _status_from_issues,
)


def _matrix_cell_count(matrix: Any) -> int:
    """Return cell count for a rank-2 matrix-like payload."""

    shape = getattr(matrix, "shape", None)
    if shape is None or len(shape) < 2:
        return 0
    try:
        n_rows = max(0, int(shape[0]))
        n_cols = max(0, int(shape[1]))
    except (TypeError, ValueError):
        return 0
    return n_rows * n_cols


@dataclass(slots=True)
class _CurriculumMetadataCollector:
    """Collect staged curriculum metadata coverage during benchmark throughput runs."""

    expected_mode: str
    bundles_seen: int = 0
    bundles_with_stage_metadata: int = 0
    mode_mismatch_bundles: int = 0

    def update(self, bundle: DatasetBundle) -> None:
        """Collect curriculum metadata counters for one generated bundle."""

        self.bundles_seen += 1
        payload = bundle.metadata.get("curriculum")
        if not isinstance(payload, dict):
            return
        mode_value = payload.get("mode")
        if mode_value != self.expected_mode:
            self.mode_mismatch_bundles += 1
        stage_value = payload.get("stage")
        if isinstance(stage_value, bool):
            return
        if isinstance(stage_value, int) and stage_value in CURRICULUM_STAGE_VALUES:
            self.bundles_with_stage_metadata += 1

    def build_summary(self) -> dict[str, Any]:
        """Build curriculum metadata guardrail metrics and issues."""

        coverage_rate = (
            float(self.bundles_with_stage_metadata) / float(self.bundles_seen)
            if self.bundles_seen > 0
            else 0.0
        )

        issues: list[dict[str, Any]] = []
        if self.bundles_with_stage_metadata != self.bundles_seen:
            issues.append(
                {
                    "metric": "curriculum_stage_metadata_coverage",
                    "severity": "fail",
                    "current": float(coverage_rate),
                    "baseline": 1.0,
                    "degradation_pct": float(max(0.0, (1.0 - coverage_rate) * 100.0)),
                    "detail": "Curriculum stage metadata must be present for all staged bundles.",
                }
            )
        if self.mode_mismatch_bundles > 0:
            mismatch_rate = float(self.mode_mismatch_bundles) / float(self.bundles_seen)
            issues.append(
                {
                    "metric": "curriculum_mode_metadata_mismatch",
                    "severity": "fail",
                    "current": float(mismatch_rate),
                    "baseline": 0.0,
                    "degradation_pct": float(mismatch_rate * 100.0),
                    "detail": "Curriculum metadata mode must match configured staged mode.",
                }
            )

        return {
            "stage_metadata_coverage_rate": float(coverage_rate),
            "mode_mismatch_bundles": int(self.mode_mismatch_bundles),
            "issues": issues,
            "status": _status_from_issues(issues),
        }


@dataclass(slots=True)
class _MissingnessAcceptanceCollector:
    """Collect per-bundle missingness metadata for acceptance guardrails."""

    target_rate: float
    bundles_seen: int = 0
    bundles_with_metadata: int = 0
    missing_cells: int = 0
    total_cells: int = 0

    def update(self, bundle: DatasetBundle) -> None:
        """Collect missingness counters for one generated bundle."""

        self.bundles_seen += 1
        payload = bundle.metadata.get("missingness")
        if not isinstance(payload, dict):
            return

        total_cells = _matrix_cell_count(bundle.X_train) + _matrix_cell_count(bundle.X_test)
        if total_cells <= 0:
            return

        missing_count_raw = payload.get("missing_count_overall")
        if isinstance(missing_count_raw, bool) or not isinstance(missing_count_raw, (int, float)):
            return
        missing_count = int(max(0, min(total_cells, int(missing_count_raw))))

        self.bundles_with_metadata += 1
        self.total_cells += total_cells
        self.missing_cells += missing_count

    def build_summary(self) -> dict[str, Any]:
        """Build acceptance guardrail metrics and issues."""

        coverage_rate = (
            float(self.bundles_with_metadata) / float(self.bundles_seen)
            if self.bundles_seen > 0
            else 0.0
        )
        realized_rate = (
            float(self.missing_cells) / float(self.total_cells) if self.total_cells > 0 else 0.0
        )
        rate_abs_error = abs(realized_rate - float(self.target_rate))

        issues: list[dict[str, Any]] = []
        if self.bundles_with_metadata != self.bundles_seen:
            issues.append(
                {
                    "metric": "missingness_metadata_coverage",
                    "severity": "fail",
                    "current": float(coverage_rate),
                    "baseline": 1.0,
                    "degradation_pct": float(max(0.0, (1.0 - coverage_rate) * 100.0)),
                    "detail": "Missingness metadata must be present for all generated bundles.",
                }
            )

        rate_error_pp = rate_abs_error * 100.0
        rate_severity = _severity_from_thresholds(
            rate_abs_error,
            warn=MISSINGNESS_RATE_WARN_ABS_ERROR,
            fail=MISSINGNESS_RATE_FAIL_ABS_ERROR,
        )
        if rate_severity != "pass":
            threshold_pp = (
                MISSINGNESS_RATE_FAIL_ABS_ERROR * 100.0
                if rate_severity == "fail"
                else MISSINGNESS_RATE_WARN_ABS_ERROR * 100.0
            )
            issues.append(
                {
                    "metric": "missingness_realized_rate_error_pp",
                    "severity": rate_severity,
                    "current": float(rate_error_pp),
                    "baseline": float(threshold_pp),
                    "degradation_pct": float(rate_error_pp),
                    "detail": "Realized missing rate drifted from configured target.",
                }
            )

        return {
            "metadata_coverage_rate": float(coverage_rate),
            "realized_rate_overall": float(realized_rate),
            "rate_abs_error": float(rate_abs_error),
            "issues": issues,
            "status": _status_from_issues(issues),
        }


@dataclass(slots=True)
class _ShiftGuardrailCollector:
    """Collect shift metadata coverage and directional observability signals."""

    bundles_seen: int = 0
    bundles_with_shift_metadata: int = 0
    bundles_with_shift_enabled_true: int = 0

    graph_edge_density_sum: float = 0.0
    graph_edge_density_count: int = 0
    edge_odds_multiplier_sum: float = 0.0
    edge_odds_multiplier_count: int = 0
    mechanism_nonlinear_mass_sum: float = 0.0
    mechanism_nonlinear_mass_count: int = 0
    noise_variance_multiplier_sum: float = 0.0
    noise_variance_multiplier_count: int = 0

    def update(self, bundle: DatasetBundle) -> None:
        """Collect shift metadata counters and directional metrics for one bundle."""

        self.bundles_seen += 1
        shift_payload = bundle.metadata.get("shift")
        if not isinstance(shift_payload, dict):
            return

        self.bundles_with_shift_metadata += 1
        if shift_payload.get("enabled") is True:
            self.bundles_with_shift_enabled_true += 1

        graph_edge_density = _coerce_optional_finite_float(
            bundle.metadata.get("graph_edge_density")
        )
        if graph_edge_density is not None:
            self.graph_edge_density_sum += float(graph_edge_density)
            self.graph_edge_density_count += 1

        edge_odds_multiplier = _coerce_optional_finite_float(
            shift_payload.get("edge_odds_multiplier")
        )
        if edge_odds_multiplier is not None:
            self.edge_odds_multiplier_sum += float(edge_odds_multiplier)
            self.edge_odds_multiplier_count += 1

        mechanism_nonlinear_mass = _coerce_optional_finite_float(
            shift_payload.get("mechanism_nonlinear_mass")
        )
        if mechanism_nonlinear_mass is not None:
            self.mechanism_nonlinear_mass_sum += float(mechanism_nonlinear_mass)
            self.mechanism_nonlinear_mass_count += 1

        noise_variance_multiplier = _coerce_optional_finite_float(
            shift_payload.get("noise_variance_multiplier")
        )
        if noise_variance_multiplier is not None:
            self.noise_variance_multiplier_sum += float(noise_variance_multiplier)
            self.noise_variance_multiplier_count += 1

    def build_summary(self) -> dict[str, Any]:
        """Build shift metadata coverage and directional-mean summary."""

        metadata_coverage_rate = (
            float(self.bundles_with_shift_metadata) / float(self.bundles_seen)
            if self.bundles_seen > 0
            else 0.0
        )
        shift_enabled_coverage_rate = (
            float(self.bundles_with_shift_enabled_true) / float(self.bundles_seen)
            if self.bundles_seen > 0
            else 0.0
        )
        return {
            "metadata_coverage_rate": float(metadata_coverage_rate),
            "shift_enabled_coverage_rate": float(shift_enabled_coverage_rate),
            "mean_graph_edge_density": _mean_or_none(
                total=self.graph_edge_density_sum,
                count=self.graph_edge_density_count,
            ),
            "mean_edge_odds_multiplier": _mean_or_none(
                total=self.edge_odds_multiplier_sum,
                count=self.edge_odds_multiplier_count,
            ),
            "mean_mechanism_nonlinear_mass": _mean_or_none(
                total=self.mechanism_nonlinear_mass_sum,
                count=self.mechanism_nonlinear_mass_count,
            ),
            "mean_noise_variance_multiplier": _mean_or_none(
                total=self.noise_variance_multiplier_sum,
                count=self.noise_variance_multiplier_count,
            ),
        }


def _compose_bundle_callback(
    *,
    diagnostics_aggregator: Any,
    missingness_acceptance: _MissingnessAcceptanceCollector | None,
    curriculum_metadata: _CurriculumMetadataCollector | None,
    shift_guardrails: _ShiftGuardrailCollector | None,
) -> Callable[[DatasetBundle], None] | None:
    """Compose optional per-bundle collectors into one callback."""

    if (
        diagnostics_aggregator is None
        and missingness_acceptance is None
        and curriculum_metadata is None
        and shift_guardrails is None
    ):
        return None

    def _on_bundle(bundle: DatasetBundle) -> None:
        if diagnostics_aggregator is not None:
            diagnostics_aggregator.update_bundle(bundle)
        if missingness_acceptance is not None:
            missingness_acceptance.update(bundle)
        if curriculum_metadata is not None:
            curriculum_metadata.update(bundle)
        if shift_guardrails is not None:
            shift_guardrails.update(bundle)

    return _on_bundle


def _coerce_optional_finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    as_float = float(value)
    if not math.isfinite(as_float):
        return None
    return as_float


def _mean_or_none(*, total: float, count: int) -> float | None:
    if count <= 0:
        return None
    return float(total / float(count))
