"""Per-bundle collector helpers for benchmark diagnostics and guardrails."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import math
from typing import Any

from dagzoo.math_utils import (
    coerce_optional_finite_float as _coerce_optional_finite_float,
)
from dagzoo.types import DatasetBundle

from .constants import MISSINGNESS_RATE_FAIL_ABS_ERROR, MISSINGNESS_RATE_WARN_ABS_ERROR
from .guardrails import (
    _severity_from_thresholds,
    _status_from_issues,
)

_NOISE_FAMILY_MIXTURE = "mixture"
_NOISE_MIXTURE_COMPONENTS = {"gaussian", "laplace", "student_t"}


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


@dataclass(slots=True)
class _NoiseGuardrailCollector:
    """Collect noise metadata coverage and validity checks."""

    expected_family_requested: str
    bundles_seen: int = 0
    bundles_with_metadata: int = 0
    bundles_with_valid_metadata: int = 0
    sampled_family_counts: dict[str, int] = field(default_factory=dict)
    invalid_reason_counts: dict[str, int] = field(default_factory=dict)

    def update(self, bundle: DatasetBundle) -> None:
        """Collect noise metadata validity counters for one generated bundle."""

        self.bundles_seen += 1
        payload = bundle.metadata.get("noise_distribution")
        if not isinstance(payload, dict):
            return

        self.bundles_with_metadata += 1
        valid, sampled_family, reason = self._validate_payload(payload)
        if not valid:
            if reason is not None:
                self.invalid_reason_counts[reason] = (
                    int(self.invalid_reason_counts.get(reason, 0)) + 1
                )
            return

        self.bundles_with_valid_metadata += 1
        self.sampled_family_counts[sampled_family] = (
            int(self.sampled_family_counts.get(sampled_family, 0)) + 1
        )

    def _validate_payload(self, payload: dict[str, Any]) -> tuple[bool, str, str | None]:
        expected = str(self.expected_family_requested).strip().lower()
        family_requested_raw = payload.get("family_requested")
        family_sampled_raw = payload.get("family_sampled")
        sampling_strategy_raw = payload.get("sampling_strategy")
        scale_raw = payload.get("base_scale")
        student_t_df_raw = payload.get("student_t_df")
        mixture_weights_raw = payload.get("mixture_weights")

        if not isinstance(family_requested_raw, str):
            return False, "", "family_requested_type"
        family_requested = family_requested_raw.strip().lower()
        if family_requested != expected:
            return False, "", "family_requested_mismatch"

        if not isinstance(family_sampled_raw, str):
            return False, "", "family_sampled_type"
        family_sampled = family_sampled_raw.strip().lower()

        if not isinstance(sampling_strategy_raw, str):
            return False, "", "sampling_strategy_type"
        if sampling_strategy_raw.strip().lower() != "dataset_level":
            return False, "", "sampling_strategy_value"

        scale = _coerce_optional_finite_float(scale_raw)
        student_t_df = _coerce_optional_finite_float(student_t_df_raw)
        if scale is None or scale <= 0.0:
            return False, "", "base_scale_value"
        if student_t_df is None or student_t_df <= 2.0:
            return False, "", "student_t_df_value"

        if family_requested != _NOISE_FAMILY_MIXTURE:
            if family_sampled != family_requested:
                return False, "", "family_sampled_mismatch"
            if mixture_weights_raw is not None:
                return False, "", "mixture_weights_unexpected"
            return True, family_sampled, None

        if family_sampled not in _NOISE_MIXTURE_COMPONENTS:
            return False, "", "family_sampled_invalid_for_mixture"
        if not isinstance(mixture_weights_raw, dict):
            return False, "", "mixture_weights_type"

        total_weight = 0.0
        for key_raw, value_raw in mixture_weights_raw.items():
            if not isinstance(key_raw, str):
                return False, "", "mixture_weights_key_type"
            key = key_raw.strip().lower()
            if key not in _NOISE_MIXTURE_COMPONENTS:
                return False, "", "mixture_weights_key_value"
            value = _coerce_optional_finite_float(value_raw)
            if value is None or value < 0.0:
                return False, "", "mixture_weights_value"
            total_weight += float(value)
        if total_weight <= 0.0:
            return False, "", "mixture_weights_total_nonpositive"
        if not math.isclose(total_weight, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            return False, "", "mixture_weights_total_not_one"
        return True, family_sampled, None

    def build_summary(self) -> dict[str, Any]:
        """Build noise metadata coverage/validity summary."""

        metadata_coverage_rate = (
            float(self.bundles_with_metadata) / float(self.bundles_seen)
            if self.bundles_seen > 0
            else 0.0
        )
        metadata_valid_rate = (
            float(self.bundles_with_valid_metadata) / float(self.bundles_seen)
            if self.bundles_seen > 0
            else 0.0
        )

        issues: list[dict[str, Any]] = []
        if self.bundles_with_metadata != self.bundles_seen:
            issues.append(
                {
                    "metric": "noise_metadata_coverage",
                    "severity": "fail",
                    "current": float(metadata_coverage_rate),
                    "baseline": 1.0,
                    "degradation_pct": float(max(0.0, (1.0 - metadata_coverage_rate) * 100.0)),
                    "detail": "Noise metadata must be present for all generated bundles.",
                }
            )
        if self.bundles_with_valid_metadata != self.bundles_seen:
            issues.append(
                {
                    "metric": "noise_metadata_validity",
                    "severity": "fail",
                    "current": float(metadata_valid_rate),
                    "baseline": 1.0,
                    "degradation_pct": float(max(0.0, (1.0 - metadata_valid_rate) * 100.0)),
                    "detail": "Noise metadata must be valid and consistent with configured family.",
                }
            )

        return {
            "metadata_coverage_rate": float(metadata_coverage_rate),
            "metadata_valid_rate": float(metadata_valid_rate),
            "valid_metadata_count": int(self.bundles_with_valid_metadata),
            "sampled_family_counts": {
                key: int(self.sampled_family_counts[key])
                for key in sorted(self.sampled_family_counts)
            },
            "invalid_reason_counts": {
                key: int(self.invalid_reason_counts[key])
                for key in sorted(self.invalid_reason_counts)
            },
            "issues": issues,
            "status": _status_from_issues(issues),
        }


@dataclass(slots=True)
class _ThroughputPressureCollector:
    """Collect attempt and filter-pressure counters needed for throughput attribution."""

    datasets_seen: int = 0
    attempts_total: int = 0
    retry_dataset_count: int = 0
    filter_attempts_total: int = 0
    filter_rejections_total: int = 0
    filter_retry_dataset_count: int = 0

    @staticmethod
    def _coerce_non_negative_int(value: Any, *, default: int) -> int:
        if isinstance(value, bool):
            return int(default)
        if isinstance(value, int):
            return int(max(0, value))
        if isinstance(value, float):
            if not math.isfinite(value):
                return int(default)
            return int(max(0, int(value)))
        if isinstance(value, str):
            normalized = value.strip()
            signless = normalized[1:] if normalized.startswith(("+", "-")) else normalized
            if not signless.isdigit():
                return int(default)
            return int(max(0, int(normalized)))
        return int(default)

    def update(self, bundle: DatasetBundle) -> None:
        """Collect generation-attempt and filter-retry counters for one bundle."""

        self.datasets_seen += 1
        metadata = bundle.metadata
        attempts_payload = metadata.get("generation_attempts")

        total_attempts = 1
        filter_attempts = 0
        filter_rejections = 0

        if isinstance(attempts_payload, dict):
            total_attempts = self._coerce_non_negative_int(
                attempts_payload.get("total_attempts"),
                default=1,
            )
            filter_attempts = self._coerce_non_negative_int(
                attempts_payload.get("filter_attempts"),
                default=0,
            )
            filter_rejections = self._coerce_non_negative_int(
                attempts_payload.get("filter_rejections"),
                default=0,
            )
        else:
            attempt_used = self._coerce_non_negative_int(metadata.get("attempt_used"), default=0)
            total_attempts = max(1, attempt_used + 1)
            filter_payload = metadata.get("filter")
            if isinstance(filter_payload, dict) and bool(filter_payload.get("enabled")):
                filter_attempts = 1
                if not bool(filter_payload.get("accepted", False)):
                    filter_rejections = 1

        total_attempts = max(1, int(total_attempts))
        filter_attempts = max(0, int(filter_attempts))
        filter_rejections = max(0, min(int(filter_rejections), filter_attempts))

        self.attempts_total += total_attempts
        if total_attempts > 1:
            self.retry_dataset_count += 1

        self.filter_attempts_total += filter_attempts
        self.filter_rejections_total += filter_rejections
        if filter_rejections > 0:
            self.filter_retry_dataset_count += 1

    def build_summary(self) -> dict[str, Any]:
        """Build aggregate attempt/filter pressure summary metrics."""

        datasets = int(self.datasets_seen)
        attempts = int(self.attempts_total)
        filter_attempts = int(self.filter_attempts_total)
        filter_rejections = int(self.filter_rejections_total)
        retry_datasets = int(self.retry_dataset_count)
        filter_retry_datasets = int(self.filter_retry_dataset_count)

        attempts_per_dataset = float(attempts) / float(datasets) if datasets > 0 else 0.0
        retry_dataset_rate = float(retry_datasets) / float(datasets) if datasets > 0 else None
        filter_retry_dataset_rate = (
            float(filter_retry_datasets) / float(datasets)
            if datasets > 0 and filter_attempts > 0
            else None
        )
        filter_rejection_rate_attempt_level = (
            float(filter_rejections) / float(filter_attempts) if filter_attempts > 0 else None
        )

        return {
            "datasets_seen": datasets,
            "attempts_total": attempts,
            "attempts_per_dataset_mean": float(attempts_per_dataset),
            "retry_dataset_count": retry_datasets,
            "retry_dataset_rate": retry_dataset_rate,
            "filter_attempts_total": filter_attempts,
            "filter_rejections_total": filter_rejections,
            "filter_rejection_rate_attempt_level": filter_rejection_rate_attempt_level,
            "filter_retry_dataset_count": filter_retry_datasets,
            "filter_retry_dataset_rate": filter_retry_dataset_rate,
        }


def _compose_bundle_callback(
    *,
    diagnostics_aggregator: Any,
    missingness_acceptance: _MissingnessAcceptanceCollector | None,
    shift_guardrails: _ShiftGuardrailCollector | None,
    noise_guardrails: _NoiseGuardrailCollector | None,
    throughput_pressure: _ThroughputPressureCollector | None = None,
) -> Callable[[DatasetBundle], None] | None:
    """Compose optional per-bundle collectors into one callback."""

    if (
        diagnostics_aggregator is None
        and missingness_acceptance is None
        and shift_guardrails is None
        and noise_guardrails is None
        and throughput_pressure is None
    ):
        return None

    def _on_bundle(bundle: DatasetBundle) -> None:
        if diagnostics_aggregator is not None:
            diagnostics_aggregator.update_bundle(bundle)
        if missingness_acceptance is not None:
            missingness_acceptance.update(bundle)
        if shift_guardrails is not None:
            shift_guardrails.update(bundle)
        if noise_guardrails is not None:
            noise_guardrails.update(bundle)
        if throughput_pressure is not None:
            throughput_pressure.update(bundle)

    return _on_bundle


def _mean_or_none(*, total: float, count: int) -> float | None:
    if count <= 0:
        return None
    return float(total / float(count))
