"""Shared config-resolution helpers for generate and benchmark command paths."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from dagzoo.config import GeneratorConfig
from dagzoo.hardware import HardwareInfo, detect_hardware
from dagzoo.hardware_policy import apply_hardware_policy

_MISSING_VALUE = "<missing>"


@dataclass(slots=True, frozen=True)
class ResolutionEvent:
    """One field-level config override event."""

    path: str
    source: str
    old_value: Any
    new_value: Any


@dataclass(slots=True, frozen=True)
class BenchmarkSmokeCaps:
    """Hard caps applied to benchmark preset configs in smoke mode."""

    n_train: int
    n_test: int
    n_features: int
    n_nodes: int


@dataclass(slots=True)
class ResolvedGenerateConfig:
    """Fully resolved generate config with hardware info and override trace."""

    config: GeneratorConfig
    hardware: HardwareInfo
    requested_device: str
    trace_events: list[ResolutionEvent]


@dataclass(slots=True)
class ResolvedBenchmarkPresetConfig:
    """Fully resolved benchmark preset config with hardware info and override trace."""

    preset_key: str
    config: GeneratorConfig
    hardware: HardwareInfo
    requested_device: str
    trace_events: list[ResolutionEvent]


def _clone_config(config: GeneratorConfig) -> GeneratorConfig:
    """Clone nested config state so resolution does not mutate caller-owned objects."""

    return GeneratorConfig.from_dict(config.to_dict())


def _append_event(
    *,
    events: list[ResolutionEvent],
    path: str,
    source: str,
    old_value: Any,
    new_value: Any,
) -> None:
    """Append one trace event when values differ."""

    if old_value == new_value:
        return
    events.append(
        ResolutionEvent(
            path=path,
            source=source,
            old_value=old_value,
            new_value=new_value,
        )
    )


def _set_config_path(
    config: GeneratorConfig,
    *,
    path: str,
    value: Any,
    source: str,
    events: list[ResolutionEvent],
) -> None:
    """Set a dotted dataclass path and emit one trace event when changed."""

    parts = path.split(".")
    target: Any = config
    for part in parts[:-1]:
        target = getattr(target, part)
    field_name = parts[-1]
    old_value = getattr(target, field_name)
    if old_value == value:
        return
    setattr(target, field_name, value)
    _append_event(
        events=events,
        path=path,
        source=source,
        old_value=old_value,
        new_value=value,
    )


def _append_diff_events(
    before: Any,
    after: Any,
    *,
    source: str,
    events: list[ResolutionEvent],
    path: str = "",
) -> None:
    """Append trace events by diffing serialized config payloads."""

    if isinstance(before, dict) and isinstance(after, dict):
        keys = sorted(set(before) | set(after))
        for key in keys:
            child_path = key if not path else f"{path}.{key}"
            old_value = before.get(key, _MISSING_VALUE)
            new_value = after.get(key, _MISSING_VALUE)
            _append_diff_events(
                old_value,
                new_value,
                source=source,
                events=events,
                path=child_path,
            )
        return
    _append_event(
        events=events,
        path=path,
        source=source,
        old_value=before,
        new_value=after,
    )


def _normalize_requested_device(device: str | None, fallback: str | None) -> str:
    """Normalize device text into the same lower-cased runtime shape used by CLI paths."""

    if device is not None:
        candidate = str(device).strip().lower()
        return candidate or "auto"
    if fallback is None:
        return "auto"
    candidate = str(fallback).strip().lower()
    return candidate or "auto"


def _apply_missingness_overrides(
    config: GeneratorConfig,
    *,
    missing_rate: float | None,
    missing_mechanism: str | None,
    missing_mar_observed_fraction: float | None,
    missing_mar_logit_scale: float | None,
    missing_mnar_logit_scale: float | None,
    events: list[ResolutionEvent],
) -> None:
    """Apply generate missingness overrides."""

    overrides = (
        ("dataset.missing_rate", missing_rate),
        ("dataset.missing_mechanism", missing_mechanism),
        ("dataset.missing_mar_observed_fraction", missing_mar_observed_fraction),
        ("dataset.missing_mar_logit_scale", missing_mar_logit_scale),
        ("dataset.missing_mnar_logit_scale", missing_mnar_logit_scale),
    )
    has_override = any(value is not None for _, value in overrides)
    if not has_override:
        return

    for path, value in overrides:
        if value is None:
            continue
        _set_config_path(
            config,
            path=path,
            value=value,
            source="cli.missingness_override",
            events=events,
        )


def _apply_smoke_caps(
    config: GeneratorConfig,
    *,
    smoke_caps: BenchmarkSmokeCaps,
    events: list[ResolutionEvent],
) -> None:
    """Apply benchmark smoke caps and trace each field-level cap event."""

    cap_specs = (
        ("dataset.n_train", int(smoke_caps.n_train)),
        ("dataset.n_test", int(smoke_caps.n_test)),
        ("dataset.n_features_min", int(smoke_caps.n_features)),
        ("dataset.n_features_max", int(smoke_caps.n_features)),
        ("graph.n_nodes_min", int(smoke_caps.n_nodes)),
        ("graph.n_nodes_max", int(smoke_caps.n_nodes)),
    )
    for path, cap_value in cap_specs:
        parts = path.split(".")
        target: Any = config
        for part in parts[:-1]:
            target = getattr(target, part)
        field_name = parts[-1]
        old_value = int(getattr(target, field_name))
        new_value = min(old_value, cap_value)
        if old_value == new_value:
            continue
        setattr(target, field_name, new_value)
        _append_event(
            events=events,
            path=path,
            source="benchmark.suite_smoke_caps",
            old_value=old_value,
            new_value=new_value,
        )


def resolve_generate_config(
    config: GeneratorConfig,
    *,
    device_override: str | None,
    hardware_policy: str,
    missing_rate: float | None,
    missing_mechanism: str | None,
    missing_mar_observed_fraction: float | None,
    missing_mar_logit_scale: float | None,
    missing_mnar_logit_scale: float | None,
    diagnostics_enabled: bool,
) -> ResolvedGenerateConfig:
    """Resolve effective config for one generate command invocation."""

    resolved = _clone_config(config)
    trace_events: list[ResolutionEvent] = []

    requested_device = _normalize_requested_device(device_override, resolved.runtime.device)
    _set_config_path(
        resolved,
        path="runtime.device",
        value=requested_device,
        source="cli.device",
        events=trace_events,
    )

    hw = detect_hardware(requested_device)
    before_policy = resolved.to_dict()
    resolved = apply_hardware_policy(resolved, hw, policy_name=hardware_policy)
    after_policy = resolved.to_dict()
    _append_diff_events(
        before_policy,
        after_policy,
        source=f"hardware_policy.{str(hardware_policy).strip().lower()}",
        events=trace_events,
    )

    _apply_missingness_overrides(
        resolved,
        missing_rate=missing_rate,
        missing_mechanism=missing_mechanism,
        missing_mar_observed_fraction=missing_mar_observed_fraction,
        missing_mar_logit_scale=missing_mar_logit_scale,
        missing_mnar_logit_scale=missing_mnar_logit_scale,
        events=trace_events,
    )

    if diagnostics_enabled:
        _set_config_path(
            resolved,
            path="diagnostics.enabled",
            value=True,
            source="cli.diagnostics",
            events=trace_events,
        )

    resolved.validate_generation_constraints()
    return ResolvedGenerateConfig(
        config=resolved,
        hardware=hw,
        requested_device=requested_device,
        trace_events=trace_events,
    )


def resolve_benchmark_preset_config(
    *,
    preset_key: str,
    config: GeneratorConfig,
    preset_device: str | None,
    suite: str,
    hardware_policy: str,
    smoke_caps: BenchmarkSmokeCaps | None,
) -> ResolvedBenchmarkPresetConfig:
    """Resolve effective config for one benchmark preset run."""

    resolved = _clone_config(config)
    trace_events: list[ResolutionEvent] = []

    requested_device = _normalize_requested_device(preset_device, resolved.runtime.device)
    _set_config_path(
        resolved,
        path="runtime.device",
        value=requested_device,
        source="benchmark.preset_device",
        events=trace_events,
    )

    hw = detect_hardware(requested_device)
    before_policy = resolved.to_dict()
    resolved = apply_hardware_policy(resolved, hw, policy_name=hardware_policy)
    after_policy = resolved.to_dict()
    _append_diff_events(
        before_policy,
        after_policy,
        source=f"hardware_policy.{str(hardware_policy).strip().lower()}",
        events=trace_events,
    )

    normalized_suite = str(suite).strip().lower()
    if normalized_suite == "smoke":
        if smoke_caps is None:
            raise ValueError("Benchmark smoke suite config resolution requires smoke cap values.")
        _apply_smoke_caps(resolved, smoke_caps=smoke_caps, events=trace_events)

    resolved.validate_generation_constraints()
    return ResolvedBenchmarkPresetConfig(
        preset_key=preset_key,
        config=resolved,
        hardware=hw,
        requested_device=requested_device,
        trace_events=trace_events,
    )


def serialize_resolution_events(events: list[ResolutionEvent]) -> list[dict[str, Any]]:
    """Convert trace dataclasses into JSON/YAML-safe dictionaries."""

    return [asdict(event) for event in events]


__all__ = [
    "BenchmarkSmokeCaps",
    "ResolutionEvent",
    "ResolvedBenchmarkPresetConfig",
    "ResolvedGenerateConfig",
    "resolve_benchmark_preset_config",
    "resolve_generate_config",
    "serialize_resolution_events",
]
