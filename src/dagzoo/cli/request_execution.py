"""Request-file execution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dagzoo.config import REQUEST_PROFILE_SMOKE, RequestFileConfig, clone_generator_config
from dagzoo.core.config_resolution import (
    append_config_diff_events,
    cap_rows_spec_to_total,
    resolve_request_config,
    serialize_resolution_events,
)
from dagzoo.core.fixed_layout_runtime import realize_generation_config_for_run
from dagzoo.filtering import DeferredFilterRunResult
from dagzoo.filtering.deferred_filter import MANIFEST_FILENAME, SUMMARY_FILENAME
from dagzoo.io.parquet_writer import write_packed_parquet_shards_stream

from .common import get_cli_public_api
from .effective_config import (
    print_effective_config,
    print_resolution_trace,
    write_effective_config,
    write_effective_config_trace,
)


@dataclass(slots=True)
class RequestRunResult:
    """Result payload for one request execution."""

    request_path: Path
    run_root: Path
    generated_dir: Path
    filter_dir: Path
    curated_dir: Path
    effective_config_path: Path
    effective_config_trace_path: Path
    generated_datasets: int
    filter_result: DeferredFilterRunResult


def _request_output_dirs(run_root: Path) -> tuple[Path, Path, Path]:
    """Return the fixed request-run artifact directories."""

    return run_root / "generated", run_root / "filter", run_root / "curated"


def _ensure_request_run_output_safe(run_root: Path) -> tuple[Path, Path, Path]:
    """Fail fast when one request-run output root already holds prior artifacts."""

    if run_root.exists() and not run_root.is_dir():
        raise RuntimeError(f"Request output_root must be a directory path: {run_root}")

    generated_dir, filter_dir, curated_dir = _request_output_dirs(run_root)
    for path in (generated_dir, filter_dir, curated_dir):
        if path.exists() and not path.is_dir():
            raise RuntimeError(f"Request artifact path must be a directory: {path}")

    stale_generated = next(generated_dir.glob("shard_*"), None) if generated_dir.exists() else None
    if stale_generated is not None:
        raise RuntimeError(
            "Request generated output already contains shard data: "
            f"{generated_dir}. Remove existing shard_* folders or choose a new output_root."
        )

    if filter_dir.exists() and (
        (filter_dir / MANIFEST_FILENAME).exists() or (filter_dir / SUMMARY_FILENAME).exists()
    ):
        raise RuntimeError(
            "Request filter output already contains prior artifacts: "
            f"{filter_dir}. Remove prior filter artifacts or choose a new output_root."
        )

    stale_curated = next(curated_dir.glob("shard_*"), None) if curated_dir.exists() else None
    if stale_curated is not None:
        raise RuntimeError(
            "Request curated output already contains shard data: "
            f"{curated_dir}. Remove existing shard_* folders or choose a new output_root."
        )

    return generated_dir, filter_dir, curated_dir


def run_request_execution(
    *,
    request_path: str | Path,
    device_override: str | None,
    hardware_policy: str,
    n_jobs_override: int | None,
    print_effective_config_flag: bool,
    print_resolution_trace_flag: bool,
) -> RequestRunResult:
    """Resolve and execute one request-file run through canonical generate->filter."""

    request_file = RequestFileConfig.from_yaml(request_path)
    run_root = Path(request_file.output_root)
    generated_dir, filter_dir, curated_dir = _ensure_request_run_output_safe(run_root)
    resolved = resolve_request_config(
        request=request_file,
        device_override=device_override,
        hardware_policy=hardware_policy,
    )
    trace_events = list(resolved.trace_events)
    pre_realization_config = clone_generator_config(resolved.config, revalidate=False)
    if request_file.profile == REQUEST_PROFILE_SMOKE:
        cap_rows_spec_to_total(
            pre_realization_config,
            total_rows_cap=int(
                pre_realization_config.dataset.n_train + pre_realization_config.dataset.n_test
            ),
        )
        append_config_diff_events(
            resolved.config,
            pre_realization_config,
            source="request.smoke_rows_cap",
            events=trace_events,
        )

    seed = pre_realization_config.seed
    config, run_seed, requested_device, _resolved_device = realize_generation_config_for_run(
        pre_realization_config,
        seed=seed,
        device=resolved.requested_device,
    )
    if bool(config.filter.enabled):
        raise ValueError(
            "Inline filtering has been removed from request execution. "
            "Request runs must resolve to filter.enabled=false and use deferred filtering."
        )

    append_config_diff_events(
        pre_realization_config,
        config,
        source="request.run_realization",
        events=trace_events,
    )
    trace_payload = serialize_resolution_events(trace_events)

    if print_effective_config_flag:
        print_effective_config(config, header="Effective config:")

    effective_config_path = write_effective_config(config, generated_dir / "effective_config.yaml")
    effective_config_trace_path = write_effective_config_trace(
        trace_payload,
        generated_dir / "effective_config_trace.yaml",
    )
    if print_resolution_trace_flag:
        print_resolution_trace(trace_payload, header="Resolution trace:")

    cli_api = get_cli_public_api()
    print(
        f"Hardware backend={resolved.hardware.backend} device='{resolved.hardware.device_name}' "
        f"memory_gb={resolved.hardware.total_memory_gb} "
        f"peak_flops={resolved.hardware.peak_flops:.3e} tier={resolved.hardware.tier} "
        f"hardware_policy={hardware_policy}"
    )
    stream = cli_api.generate_batch_iter(
        config,
        num_datasets=request_file.dataset_count,
        seed=run_seed,
        device=requested_device,
    )
    written = write_packed_parquet_shards_stream(
        stream,
        out_dir=generated_dir,
        shard_size=config.output.shard_size,
        compression=config.output.compression,
    )
    filter_result = cli_api.run_deferred_filter(
        in_dir=generated_dir,
        out_dir=filter_dir,
        curated_out_dir=curated_dir,
        n_jobs_override=n_jobs_override,
    )

    return RequestRunResult(
        request_path=Path(request_path),
        run_root=run_root,
        generated_dir=generated_dir,
        filter_dir=filter_dir,
        curated_dir=curated_dir,
        effective_config_path=effective_config_path,
        effective_config_trace_path=effective_config_trace_path,
        generated_datasets=int(written),
        filter_result=filter_result,
    )
