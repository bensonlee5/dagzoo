"""Request-run handoff manifest helpers for downstream corpus consumers."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from time import time_ns
from typing import Any, cast

from dagzoo.config import RequestFileConfig
from dagzoo.core.identity import stable_blake2s_hex
from dagzoo.core.staged_artifacts import cleanup_path, promote_staged_path, staged_output_path
from dagzoo.io.lineage_artifact import sha256_hex
from dagzoo.math import sanitize_json

REQUEST_HANDOFF_MANIFEST_FILENAME = "handoff_manifest.json"
REQUEST_HANDOFF_SCHEMA_NAME = "dagzoo_request_handoff_manifest"
REQUEST_HANDOFF_SCHEMA_VERSION = 2
REQUEST_HANDOFF_SOURCE_FAMILY = "dagzoo.fixed_layout_scm"
REQUEST_HANDOFF_RECOMMENDED_TRAINING_CORPUS = "curated"
_BLAKE2S_HEX_LENGTH = 32
_SHA256_HEX_LENGTH = 64


def _raise(path: str, message: str) -> None:
    raise ValueError(f"{path}: {message}")


def _require_mapping(value: object, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _raise(path, "must be a mapping")
    return cast(Mapping[str, Any], value)


def _require_non_empty_string(value: object, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _raise(path, "must be a non-empty string")
    return cast(str, value)


def _require_optional_string(value: object, *, path: str) -> str | None:
    if value is None:
        return None
    return _require_non_empty_string(value, path=path)


def _require_hex_string(value: object, *, path: str, expected_length: int) -> str:
    text = _require_non_empty_string(value, path=path)
    if len(text) != expected_length or any(ch not in "0123456789abcdef" for ch in text):
        _raise(path, f"must be a {expected_length}-character lowercase hexadecimal string")
    return text


def _require_int(value: object, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _raise(path, "must be an integer")
    return int(cast(int, value))


def _require_non_negative_float(value: object, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _raise(path, "must be a finite non-negative number")
    number = float(cast(int | float, value))
    if not math.isfinite(number) or number < 0.0:
        _raise(path, "must be a finite non-negative number")
    return number


def _require_optional_unit_interval(value: object, *, path: str) -> float | None:
    if value is None:
        return None
    number = _require_non_negative_float(value, path=path)
    if number > 1.0:
        _raise(path, "must be <= 1.0")
    return number


def _resolve_path_str(path: str | Path) -> str:
    return str(Path(path).resolve())


def _resolve_optional_path_str(path: str | Path | None) -> str | None:
    if path is None:
        return None
    return _resolve_path_str(path)


def _read_sha256(path: str | Path) -> str:
    return sha256_hex(Path(path).read_bytes())


def _relative_posix_path(path: str | Path, *, start: str | Path) -> str:
    return Path(path).resolve().relative_to(Path(start).resolve()).as_posix()


def _datasets_per_minute(*, datasets: int, elapsed_seconds: float) -> float:
    if elapsed_seconds <= 0.0:
        return 0.0
    return (float(datasets) / float(elapsed_seconds)) * 60.0


def build_request_handoff_manifest(
    *,
    request_path: str | Path,
    request: RequestFileConfig,
    run_root: str | Path,
    generated_dir: str | Path,
    filter_dir: str | Path,
    filtered_corpus_dir: str | Path,
    effective_config_path: str | Path,
    effective_config_trace_path: str | Path,
    filter_manifest_path: str | Path,
    filter_summary_path: str | Path,
    generated_datasets: int,
    generation_elapsed_seconds: float,
    filter_total_datasets: int,
    filter_accepted_datasets: int,
    filter_rejected_datasets: int,
    filter_elapsed_seconds: float,
    requested_device: str,
    resolved_device: str,
    hardware_backend: str,
    hardware_device_name: str,
    hardware_tier: str,
    hardware_policy: str,
    diversity_summary_json_path: str | Path | None = None,
    diversity_summary_md_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the machine-readable request handoff manifest payload."""

    total_datasets = int(filter_total_datasets)
    acceptance_rate = (
        float(filter_accepted_datasets) / float(total_datasets) if total_datasets > 0 else None
    )
    effective_config_sha256 = _read_sha256(effective_config_path)
    effective_config_trace_sha256 = _read_sha256(effective_config_trace_path)
    filter_manifest_sha256 = _read_sha256(filter_manifest_path)
    filter_summary_sha256 = _read_sha256(filter_summary_path)
    request_run_id = stable_blake2s_hex(
        {
            "request_payload": request.to_dict(),
            "effective_config_sha256": effective_config_sha256,
            "requested_device": str(requested_device),
            "resolved_device": str(resolved_device),
            "hardware_policy": str(hardware_policy),
        }
    )
    generated_corpus_id = stable_blake2s_hex(
        {
            "request_run_id": request_run_id,
            "corpus_kind": "generated",
        }
    )
    curated_corpus_id = stable_blake2s_hex(
        {
            "request_run_id": request_run_id,
            "corpus_kind": "curated",
            "filter_summary_sha256": filter_summary_sha256,
        }
    )
    payload: dict[str, Any] = {
        "schema_name": REQUEST_HANDOFF_SCHEMA_NAME,
        "schema_version": REQUEST_HANDOFF_SCHEMA_VERSION,
        "identity": {
            "source_family": REQUEST_HANDOFF_SOURCE_FAMILY,
            "request_run_id": request_run_id,
            "generated_corpus_id": generated_corpus_id,
            "curated_corpus_id": curated_corpus_id,
        },
        "request": {
            "path": _resolve_path_str(request_path),
            "payload": request.to_dict(),
        },
        "artifacts": {
            "run_root": _resolve_path_str(run_root),
            "generated_dir": _resolve_path_str(generated_dir),
            "filter_dir": _resolve_path_str(filter_dir),
            "filtered_corpus_dir": _resolve_path_str(filtered_corpus_dir),
            "effective_config_path": _resolve_path_str(effective_config_path),
            "effective_config_trace_path": _resolve_path_str(effective_config_trace_path),
            "filter_manifest_path": _resolve_path_str(filter_manifest_path),
            "filter_summary_path": _resolve_path_str(filter_summary_path),
        },
        "artifacts_relative": {
            "run_root": ".",
            "generated_dir": _relative_posix_path(generated_dir, start=run_root),
            "filter_dir": _relative_posix_path(filter_dir, start=run_root),
            "filtered_corpus_dir": _relative_posix_path(filtered_corpus_dir, start=run_root),
            "effective_config_path": _relative_posix_path(effective_config_path, start=run_root),
            "effective_config_trace_path": _relative_posix_path(
                effective_config_trace_path, start=run_root
            ),
            "filter_manifest_path": _relative_posix_path(filter_manifest_path, start=run_root),
            "filter_summary_path": _relative_posix_path(filter_summary_path, start=run_root),
        },
        "checksums": {
            "effective_config_sha256": effective_config_sha256,
            "effective_config_trace_sha256": effective_config_trace_sha256,
            "filter_manifest_sha256": filter_manifest_sha256,
            "filter_summary_sha256": filter_summary_sha256,
        },
        "summary": {
            "generated_datasets": int(generated_datasets),
            "accepted_datasets": int(filter_accepted_datasets),
            "rejected_datasets": int(filter_rejected_datasets),
            "acceptance_rate": acceptance_rate,
        },
        "throughput": {
            "generation_stage": {
                "generated_datasets": int(generated_datasets),
                "elapsed_seconds": float(generation_elapsed_seconds),
                "datasets_per_minute": _datasets_per_minute(
                    datasets=int(generated_datasets),
                    elapsed_seconds=float(generation_elapsed_seconds),
                ),
            },
            "filter_stage": {
                "total_datasets": int(filter_total_datasets),
                "accepted_datasets": int(filter_accepted_datasets),
                "rejected_datasets": int(filter_rejected_datasets),
                "elapsed_seconds": float(filter_elapsed_seconds),
                "datasets_per_minute": _datasets_per_minute(
                    datasets=int(filter_total_datasets),
                    elapsed_seconds=float(filter_elapsed_seconds),
                ),
            },
        },
        "hardware": {
            "requested_device": str(requested_device),
            "resolved_device": str(resolved_device),
            "backend": str(hardware_backend),
            "device_name": str(hardware_device_name),
            "tier": str(hardware_tier),
            "hardware_policy": str(hardware_policy),
        },
        "diversity_artifacts": {
            "summary_json_path": _resolve_optional_path_str(diversity_summary_json_path),
            "summary_md_path": _resolve_optional_path_str(diversity_summary_md_path),
        },
        "defaults": {
            "recommended_training_corpus": REQUEST_HANDOFF_RECOMMENDED_TRAINING_CORPUS,
            "recommended_training_artifact_key": "filtered_corpus_dir",
            "curation_policy": "accepted_only",
        },
    }
    validate_request_handoff_manifest(payload)
    return payload


def validate_request_handoff_manifest(payload: Mapping[str, Any]) -> None:
    """Validate the request handoff manifest wire shape."""

    root = _require_mapping(payload, path="handoff_manifest")
    schema_name = _require_non_empty_string(
        root.get("schema_name"),
        path="handoff_manifest.schema_name",
    )
    if schema_name != REQUEST_HANDOFF_SCHEMA_NAME:
        _raise(
            "handoff_manifest.schema_name",
            f"must equal {REQUEST_HANDOFF_SCHEMA_NAME!r}",
        )
    schema_version = _require_int(
        root.get("schema_version"),
        path="handoff_manifest.schema_version",
    )
    if schema_version != REQUEST_HANDOFF_SCHEMA_VERSION:
        _raise(
            "handoff_manifest.schema_version",
            f"must equal {REQUEST_HANDOFF_SCHEMA_VERSION}",
        )

    identity = _require_mapping(root.get("identity"), path="handoff_manifest.identity")
    source_family = _require_non_empty_string(
        identity.get("source_family"),
        path="handoff_manifest.identity.source_family",
    )
    if source_family != REQUEST_HANDOFF_SOURCE_FAMILY:
        _raise(
            "handoff_manifest.identity.source_family",
            f"must equal {REQUEST_HANDOFF_SOURCE_FAMILY!r}",
        )
    _require_hex_string(
        identity.get("request_run_id"),
        path="handoff_manifest.identity.request_run_id",
        expected_length=_BLAKE2S_HEX_LENGTH,
    )
    _require_hex_string(
        identity.get("generated_corpus_id"),
        path="handoff_manifest.identity.generated_corpus_id",
        expected_length=_BLAKE2S_HEX_LENGTH,
    )
    _require_hex_string(
        identity.get("curated_corpus_id"),
        path="handoff_manifest.identity.curated_corpus_id",
        expected_length=_BLAKE2S_HEX_LENGTH,
    )

    request = _require_mapping(root.get("request"), path="handoff_manifest.request")
    _require_non_empty_string(request.get("path"), path="handoff_manifest.request.path")
    request_payload = _require_mapping(
        request.get("payload"),
        path="handoff_manifest.request.payload",
    )
    try:
        RequestFileConfig.from_dict(dict(request_payload))
    except (TypeError, ValueError) as exc:
        _raise("handoff_manifest.request.payload", str(exc))

    artifacts = _require_mapping(root.get("artifacts"), path="handoff_manifest.artifacts")
    for key in (
        "run_root",
        "generated_dir",
        "filter_dir",
        "filtered_corpus_dir",
        "effective_config_path",
        "effective_config_trace_path",
        "filter_manifest_path",
        "filter_summary_path",
    ):
        _require_non_empty_string(
            artifacts.get(key),
            path=f"handoff_manifest.artifacts.{key}",
        )

    artifacts_relative = _require_mapping(
        root.get("artifacts_relative"),
        path="handoff_manifest.artifacts_relative",
    )
    run_root_relative = _require_non_empty_string(
        artifacts_relative.get("run_root"),
        path="handoff_manifest.artifacts_relative.run_root",
    )
    if run_root_relative != ".":
        _raise("handoff_manifest.artifacts_relative.run_root", "must equal '.'")
    for key in (
        "generated_dir",
        "filter_dir",
        "filtered_corpus_dir",
        "effective_config_path",
        "effective_config_trace_path",
        "filter_manifest_path",
        "filter_summary_path",
    ):
        _require_non_empty_string(
            artifacts_relative.get(key),
            path=f"handoff_manifest.artifacts_relative.{key}",
        )

    checksums = _require_mapping(root.get("checksums"), path="handoff_manifest.checksums")
    for key in (
        "effective_config_sha256",
        "effective_config_trace_sha256",
        "filter_manifest_sha256",
        "filter_summary_sha256",
    ):
        _require_hex_string(
            checksums.get(key),
            path=f"handoff_manifest.checksums.{key}",
            expected_length=_SHA256_HEX_LENGTH,
        )

    summary = _require_mapping(root.get("summary"), path="handoff_manifest.summary")
    _require_int(
        summary.get("generated_datasets"), path="handoff_manifest.summary.generated_datasets"
    )
    _require_int(
        summary.get("accepted_datasets"), path="handoff_manifest.summary.accepted_datasets"
    )
    _require_int(
        summary.get("rejected_datasets"), path="handoff_manifest.summary.rejected_datasets"
    )
    _require_optional_unit_interval(
        summary.get("acceptance_rate"),
        path="handoff_manifest.summary.acceptance_rate",
    )

    throughput = _require_mapping(root.get("throughput"), path="handoff_manifest.throughput")
    generation_stage = _require_mapping(
        throughput.get("generation_stage"),
        path="handoff_manifest.throughput.generation_stage",
    )
    _require_int(
        generation_stage.get("generated_datasets"),
        path="handoff_manifest.throughput.generation_stage.generated_datasets",
    )
    _require_non_negative_float(
        generation_stage.get("elapsed_seconds"),
        path="handoff_manifest.throughput.generation_stage.elapsed_seconds",
    )
    _require_non_negative_float(
        generation_stage.get("datasets_per_minute"),
        path="handoff_manifest.throughput.generation_stage.datasets_per_minute",
    )

    filter_stage = _require_mapping(
        throughput.get("filter_stage"),
        path="handoff_manifest.throughput.filter_stage",
    )
    _require_int(
        filter_stage.get("total_datasets"),
        path="handoff_manifest.throughput.filter_stage.total_datasets",
    )
    _require_int(
        filter_stage.get("accepted_datasets"),
        path="handoff_manifest.throughput.filter_stage.accepted_datasets",
    )
    _require_int(
        filter_stage.get("rejected_datasets"),
        path="handoff_manifest.throughput.filter_stage.rejected_datasets",
    )
    _require_non_negative_float(
        filter_stage.get("elapsed_seconds"),
        path="handoff_manifest.throughput.filter_stage.elapsed_seconds",
    )
    _require_non_negative_float(
        filter_stage.get("datasets_per_minute"),
        path="handoff_manifest.throughput.filter_stage.datasets_per_minute",
    )

    hardware = _require_mapping(root.get("hardware"), path="handoff_manifest.hardware")
    for key in (
        "requested_device",
        "resolved_device",
        "backend",
        "device_name",
        "tier",
        "hardware_policy",
    ):
        _require_non_empty_string(
            hardware.get(key),
            path=f"handoff_manifest.hardware.{key}",
        )

    diversity_artifacts = _require_mapping(
        root.get("diversity_artifacts"),
        path="handoff_manifest.diversity_artifacts",
    )
    _require_optional_string(
        diversity_artifacts.get("summary_json_path"),
        path="handoff_manifest.diversity_artifacts.summary_json_path",
    )
    _require_optional_string(
        diversity_artifacts.get("summary_md_path"),
        path="handoff_manifest.diversity_artifacts.summary_md_path",
    )

    defaults = _require_mapping(root.get("defaults"), path="handoff_manifest.defaults")
    recommended_training_corpus = _require_non_empty_string(
        defaults.get("recommended_training_corpus"),
        path="handoff_manifest.defaults.recommended_training_corpus",
    )
    if recommended_training_corpus != REQUEST_HANDOFF_RECOMMENDED_TRAINING_CORPUS:
        _raise(
            "handoff_manifest.defaults.recommended_training_corpus",
            f"must equal {REQUEST_HANDOFF_RECOMMENDED_TRAINING_CORPUS!r}",
        )
    recommended_artifact_key = _require_non_empty_string(
        defaults.get("recommended_training_artifact_key"),
        path="handoff_manifest.defaults.recommended_training_artifact_key",
    )
    if recommended_artifact_key != "filtered_corpus_dir":
        _raise(
            "handoff_manifest.defaults.recommended_training_artifact_key",
            "must equal 'filtered_corpus_dir'",
        )
    curation_policy = _require_non_empty_string(
        defaults.get("curation_policy"),
        path="handoff_manifest.defaults.curation_policy",
    )
    if curation_policy != "accepted_only":
        _raise(
            "handoff_manifest.defaults.curation_policy",
            "must equal 'accepted_only'",
        )


def write_request_handoff_manifest(
    *,
    request_path: str | Path,
    request: RequestFileConfig,
    run_root: str | Path,
    generated_dir: str | Path,
    filter_dir: str | Path,
    filtered_corpus_dir: str | Path,
    effective_config_path: str | Path,
    effective_config_trace_path: str | Path,
    filter_manifest_path: str | Path,
    filter_summary_path: str | Path,
    generated_datasets: int,
    generation_elapsed_seconds: float,
    filter_total_datasets: int,
    filter_accepted_datasets: int,
    filter_rejected_datasets: int,
    filter_elapsed_seconds: float,
    requested_device: str,
    resolved_device: str,
    hardware_backend: str,
    hardware_device_name: str,
    hardware_tier: str,
    hardware_policy: str,
    diversity_summary_json_path: str | Path | None = None,
    diversity_summary_md_path: str | Path | None = None,
    out_path: str | Path | None = None,
) -> Path:
    """Write the request handoff manifest to disk and return its path."""

    manifest_path = (
        Path(out_path)
        if out_path is not None
        else Path(run_root) / REQUEST_HANDOFF_MANIFEST_FILENAME
    )
    payload = build_request_handoff_manifest(
        request_path=request_path,
        request=request,
        run_root=run_root,
        generated_dir=generated_dir,
        filter_dir=filter_dir,
        filtered_corpus_dir=filtered_corpus_dir,
        effective_config_path=effective_config_path,
        effective_config_trace_path=effective_config_trace_path,
        filter_manifest_path=filter_manifest_path,
        filter_summary_path=filter_summary_path,
        generated_datasets=generated_datasets,
        generation_elapsed_seconds=generation_elapsed_seconds,
        filter_total_datasets=filter_total_datasets,
        filter_accepted_datasets=filter_accepted_datasets,
        filter_rejected_datasets=filter_rejected_datasets,
        filter_elapsed_seconds=filter_elapsed_seconds,
        requested_device=requested_device,
        resolved_device=resolved_device,
        hardware_backend=hardware_backend,
        hardware_device_name=hardware_device_name,
        hardware_tier=hardware_tier,
        hardware_policy=hardware_policy,
        diversity_summary_json_path=diversity_summary_json_path,
        diversity_summary_md_path=diversity_summary_md_path,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    staged_manifest_path = staged_output_path(
        parent_dir=manifest_path.parent,
        final_name=manifest_path.name,
        staging_token=str(time_ns()),
    )
    try:
        staged_manifest_path.write_text(
            json.dumps(sanitize_json(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        promote_staged_path(staged_path=staged_manifest_path, final_path=manifest_path)
    finally:
        cleanup_path(staged_manifest_path)
    return manifest_path


__all__ = [
    "REQUEST_HANDOFF_MANIFEST_FILENAME",
    "REQUEST_HANDOFF_SCHEMA_NAME",
    "REQUEST_HANDOFF_SCHEMA_VERSION",
    "build_request_handoff_manifest",
    "validate_request_handoff_manifest",
    "write_request_handoff_manifest",
]
