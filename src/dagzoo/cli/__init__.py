"""CLI entrypoints and compatibility re-exports."""

from dagzoo.bench.baseline import build_baseline_payload, write_baseline
from dagzoo.bench.report import write_suite_json, write_suite_markdown
from dagzoo.bench.suite import run_benchmark_suite
from dagzoo.core.dataset import generate_batch_iter
from dagzoo.diagnostics import CoverageAggregator
from dagzoo.diagnostics.effective_diversity import (
    run_effective_diversity_audit,
    run_filter_calibration,
    write_effective_diversity_artifacts,
    write_filter_calibration_artifacts,
)
from dagzoo.filtering import run_deferred_filter
from dagzoo.hardware import detect_hardware

from .commands.benchmark import _print_preset_result_line
from .entrypoint import main
from .parser import build_parser
from .parser import build_parser as _build_parser
from .parsing import (
    DEVICE_CHOICES,
    HARDWARE_POLICY_CHOICES,
    MISSINGNESS_MECHANISM_CLI_CHOICES,
)

__all__ = [
    "DEVICE_CHOICES",
    "HARDWARE_POLICY_CHOICES",
    "MISSINGNESS_MECHANISM_CLI_CHOICES",
    "CoverageAggregator",
    "_build_parser",
    "_print_preset_result_line",
    "build_baseline_payload",
    "build_parser",
    "detect_hardware",
    "generate_batch_iter",
    "main",
    "run_benchmark_suite",
    "run_deferred_filter",
    "run_effective_diversity_audit",
    "run_filter_calibration",
    "write_baseline",
    "write_effective_diversity_artifacts",
    "write_filter_calibration_artifacts",
    "write_suite_json",
    "write_suite_markdown",
]
