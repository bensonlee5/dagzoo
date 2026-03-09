"""Canonical fixed-layout run preparation and execution orchestration."""

from __future__ import annotations

import copy
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch

from dagzoo.config import (
    DatasetRowsSpec,
    GeneratorConfig,
    dataset_rows_is_variable,
    resolve_dataset_total_rows,
)
from dagzoo.core.constants import FIXED_LAYOUT_PLAN_SEED_OFFSET, ROWS_REALIZATION_SEED_OFFSET
from dagzoo.core.fixed_layout import (
    _FixedLayoutPlan,
    _annotate_fixed_layout_metadata,
    _extract_emitted_schema_signature,
    _generate_fixed_layout_graph_batch_with_fallback,
    _generate_fixed_layout_label_batch_with_fallback,
    _layout_signature,
)
from dagzoo.core.fixed_layout_batched import (
    build_fixed_layout_execution_plan,
    fixed_layout_plan_signature,
)
from dagzoo.core.fixed_layout_plan_types import FixedLayoutExecutionPlan
from dagzoo.core.generation_context import (
    _attempt_seed,
    _resolve_device,
    _resolve_run_seed,
    _resolve_split_sizes,
    _split_permutation_seed,
    _torch_dtype,
    _validate_class_split_for_layout,
)
from dagzoo.core.generation_runtime import (
    _build_fixed_schema_finalization_context,
    _finalize_generated_chunk_preserve_schema,
    _finalize_generated_tensors,
)
from dagzoo.core.layout import _sample_layout
from dagzoo.core.layout_types import LayoutPlan
from dagzoo.core.noise_runtime import (
    NoiseRuntimeSelection,
    _noise_sampling_spec,
    _resolve_noise_runtime_selection,
)
from dagzoo.core.shift import resolve_shift_runtime_params
from dagzoo.core.validation import _classification_split_valid, _stratified_split_indices
from dagzoo.rng import SeedManager, offset_seed32
from dagzoo.types import DatasetBundle

_FIXED_LAYOUT_TARGET_CELLS = 4_000_000


@dataclass(slots=True)
class CanonicalFixedLayoutRun:
    """Prepared fixed-layout run context for canonical public generation."""

    config: GeneratorConfig
    plan: _FixedLayoutPlan
    run_seed: int
    requested_device: str
    resolved_device: str
    batch_size: int
    classification_attempt_plan: tuple[int, ...] | None = None


@dataclass(slots=True)
class _NoiseRuntimeGroup:
    """One chunk subgroup that shares the same raw noise sampling contract."""

    chunk_offsets: list[int]
    generation_seeds: list[int]
    selection: NoiseRuntimeSelection
    attempt: int


@dataclass(slots=True)
class _GroupedRawBatch:
    """One raw fixed-layout subgroup generation result."""

    chunk_offsets: list[int]
    selection: NoiseRuntimeSelection
    attempt: int
    x_batch: torch.Tensor
    y_batch: torch.Tensor
    aux_meta_batch: list[dict[str, Any]]
    effective_resolved_device: str
    device_fallback_reason: str | None


def _sample_fixed_layout_candidate(
    config: GeneratorConfig,
    *,
    seed: int,
    requested_device: str,
    resolved_device: str,
) -> _FixedLayoutPlan:
    """Sample one fixed-layout plan candidate without replay validation."""

    return _sample_fixed_layout_once(
        config,
        seed=seed,
        requested_device=requested_device,
        resolved_device=resolved_device,
    )


def _noise_runtime_selection_key(
    selection: NoiseRuntimeSelection,
) -> tuple[str, float, float]:
    return (
        str(selection.family_sampled),
        float(selection.base_scale),
        float(selection.student_t_df),
    )


def _group_noise_runtime_chunk(
    config: GeneratorConfig,
    *,
    dataset_seeds: list[int],
    attempts: list[int] | None = None,
) -> list[_NoiseRuntimeGroup]:
    effective_attempts = attempts or [0] * len(dataset_seeds)
    if len(effective_attempts) != len(dataset_seeds):
        raise ValueError(
            "Fixed-layout attempt plan length must match chunk dataset count: "
            f"dataset_seeds={len(dataset_seeds)} attempts={len(effective_attempts)}"
        )

    grouped: dict[tuple[str, float, float, int], _NoiseRuntimeGroup] = {}
    ordered_keys: list[tuple[str, float, float, int]] = []
    for chunk_offset, (dataset_seed, attempt) in enumerate(
        zip(dataset_seeds, effective_attempts, strict=True)
    ):
        if attempt < 0:
            raise ValueError(f"Fixed-layout attempt plan entries must be >= 0, got {attempt}.")
        data_seed = SeedManager(dataset_seed).child("data")
        selection = _resolve_noise_runtime_selection(config, run_seed=data_seed)
        generation_seed = _attempt_seed(data_seed, int(attempt))
        key = _noise_runtime_selection_key(selection) + (int(attempt),)
        if key not in grouped:
            grouped[key] = _NoiseRuntimeGroup(
                chunk_offsets=[],
                generation_seeds=[],
                selection=selection,
                attempt=int(attempt),
            )
            ordered_keys.append(key)
        group = grouped[key]
        group.chunk_offsets.append(int(chunk_offset))
        group.generation_seeds.append(int(generation_seed))
    return [grouped[key] for key in ordered_keys]


def _generate_grouped_raw_batches(
    config: GeneratorConfig,
    layout: LayoutPlan,
    *,
    execution_plan: FixedLayoutExecutionPlan,
    grouped_noise_runtime: list[_NoiseRuntimeGroup],
    requested_device: str,
    resolved_device: str,
    noise_sigma_multiplier: float,
) -> list[_GroupedRawBatch]:
    grouped_batches: list[_GroupedRawBatch] = []
    for group in grouped_noise_runtime:
        noise_spec = _noise_sampling_spec(group.selection)
        (
            x_batch,
            y_batch,
            aux_meta_batch,
            effective_resolved_device,
            device_fallback_reason,
        ) = _generate_fixed_layout_graph_batch_with_fallback(
            config,
            layout,
            execution_plan=execution_plan,
            dataset_seeds=group.generation_seeds,
            requested_device=requested_device,
            resolved_device=resolved_device,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
        grouped_batches.append(
            _GroupedRawBatch(
                chunk_offsets=list(group.chunk_offsets),
                selection=group.selection,
                attempt=int(group.attempt),
                x_batch=x_batch,
                y_batch=y_batch,
                aux_meta_batch=aux_meta_batch,
                effective_resolved_device=str(effective_resolved_device),
                device_fallback_reason=device_fallback_reason,
            )
        )
    return grouped_batches


def _validate_fixed_layout_rows_mode(config: GeneratorConfig) -> None:
    if dataset_rows_is_variable(config.dataset.rows):
        raise ValueError(
            "Fixed-layout generation requires a fixed split size; variable dataset.rows "
            "modes (range/choices) are not supported."
        )


def _effective_fixed_layout_target_cells(config: GeneratorConfig) -> int:
    """Return the configured fixed-layout auto-batch target cell budget."""

    target_cells = config.runtime.fixed_layout_target_cells
    if target_cells is None:
        return int(_FIXED_LAYOUT_TARGET_CELLS)
    return int(target_cells)


def _resolve_fixed_layout_batch_size(
    plan: _FixedLayoutPlan,
    *,
    num_datasets: int,
    batch_size: int | None,
    target_cells: int | None = None,
) -> int:
    if batch_size is not None:
        return max(1, min(int(batch_size), int(num_datasets)))
    per_dataset_cells = max(
        1, int(plan.n_train + plan.n_test) * max(1, int(plan.layout.n_features))
    )
    auto_batch = max(1, int(target_cells or _FIXED_LAYOUT_TARGET_CELLS) // per_dataset_cells)
    return max(1, min(int(num_datasets), int(auto_batch)))


def realize_generation_config_for_run(
    config: GeneratorConfig,
    *,
    seed: int | None = None,
    device: str | None = None,
) -> tuple[GeneratorConfig, int, str, str]:
    """Resolve one canonical single-run config with rows fixed for the full run."""

    run_seed = _resolve_run_seed(config, seed)
    requested_device = (device or config.runtime.device or "auto").lower()
    resolved_device = _resolve_device(config, device)

    realized = copy.deepcopy(config)
    rows_seed = offset_seed32(run_seed, ROWS_REALIZATION_SEED_OFFSET)
    total_rows = resolve_dataset_total_rows(realized.dataset.rows, dataset_seed=rows_seed)
    if total_rows is not None:
        n_test = int(realized.dataset.n_test)
        n_train = int(total_rows) - n_test
        if n_train <= 0:
            raise ValueError(
                "Resolved rows split is invalid: total rows must be > dataset.n_test "
                f"(total_rows={int(total_rows)}, n_test={n_test})."
            )
        realized.dataset.n_train = int(n_train)
        realized.dataset.rows = DatasetRowsSpec(mode="fixed", value=int(total_rows))

    return realized, int(run_seed), str(requested_device), str(resolved_device)


def prepare_canonical_fixed_layout_run(
    config: GeneratorConfig,
    *,
    num_datasets: int,
    seed: int | None = None,
    device: str | None = None,
    batch_size: int | None = None,
) -> CanonicalFixedLayoutRun:
    """Prepare one internal fixed-layout run context for public generation APIs."""

    if num_datasets < 0:
        raise ValueError(f"num_datasets must be >= 0, got {num_datasets}")

    realized_config, run_seed, requested_device, resolved_device = (
        realize_generation_config_for_run(
            config,
            seed=seed,
            device=device,
        )
    )
    base_plan_seed = offset_seed32(run_seed, FIXED_LAYOUT_PLAN_SEED_OFFSET)
    attempts = max(1, int(realized_config.filter.max_attempts))
    last_error = "unknown"
    classification_attempt_plan: tuple[int, ...] | None = None
    for attempt in range(attempts):
        candidate_seed = _attempt_seed(base_plan_seed, attempt)
        plan = (
            _sample_fixed_layout_candidate(
                realized_config,
                seed=candidate_seed,
                requested_device=requested_device,
                resolved_device=resolved_device,
            )
            if str(realized_config.dataset.task) == "classification"
            else _sample_fixed_layout(
                realized_config,
                seed=candidate_seed,
                device=requested_device,
            )
        )
        effective_batch_size = _resolve_fixed_layout_batch_size(
            plan,
            num_datasets=max(1, int(num_datasets)),
            batch_size=batch_size,
            target_cells=_effective_fixed_layout_target_cells(realized_config),
        )
        if str(realized_config.dataset.task) != "classification":
            break
        attempt_plan = _fixed_layout_plan_classification_attempt_plan(
            realized_config,
            plan=plan,
            requested_device=requested_device,
            resolved_device=resolved_device,
            run_seed=run_seed,
            num_datasets=max(1, int(num_datasets)),
            batch_size=effective_batch_size,
        )
        if attempt_plan is not None:
            classification_attempt_plan = attempt_plan
            break
        last_error = "invalid_class_split"
        if attempt == attempts - 1:
            raise ValueError(
                "Failed to prepare a replayable canonical fixed-layout classification run "
                f"after {attempts} attempts. Last reason: {last_error}."
            )
    else:
        raise ValueError(
            "Failed to prepare a fixed-layout run after "
            f"{attempts} attempts. Last reason: {last_error}."
        )
    return CanonicalFixedLayoutRun(
        config=realized_config,
        plan=plan,
        run_seed=int(run_seed),
        requested_device=str(requested_device),
        resolved_device=str(resolved_device),
        batch_size=int(effective_batch_size),
        classification_attempt_plan=classification_attempt_plan,
    )


def _sample_fixed_layout_once(
    config: GeneratorConfig,
    *,
    seed: int,
    requested_device: str,
    resolved_device: str,
) -> _FixedLayoutPlan:
    """Sample one fixed-layout plan candidate without replay validation retries."""

    _validate_fixed_layout_rows_mode(config)
    manager = SeedManager(seed)
    layout_gen = manager.torch_rng("layout")
    layout = _sample_layout(config, layout_gen, "cpu")
    n_train, n_test = _resolve_split_sizes(config, dataset_seed=seed)
    _validate_class_split_for_layout(config, layout=layout, n_train=n_train, n_test=n_test)
    shift_params = resolve_shift_runtime_params(config)
    execution_plan = build_fixed_layout_execution_plan(
        config,
        layout,
        plan_seed=seed,
        mechanism_logit_tilt=float(shift_params.mechanism_logit_tilt),
    )
    return _FixedLayoutPlan(
        layout=layout,
        requested_device=requested_device,
        resolved_device=resolved_device,
        plan_seed=int(seed),
        n_train=int(n_train),
        n_test=int(n_test),
        layout_signature=_layout_signature(layout),
        execution_plan=execution_plan,
        plan_signature=fixed_layout_plan_signature(execution_plan),
    )


def _raw_classification_labels_support_split(
    y: torch.Tensor,
    *,
    dataset_seed: int,
    attempt: int,
    n_train: int,
) -> bool:
    """Return whether one raw classification label vector can satisfy split constraints."""

    labels = y.to(device="cpu", dtype=torch.int64)
    split_generator = torch.Generator(device="cpu")
    split_generator.manual_seed(_split_permutation_seed(dataset_seed, attempt))
    try:
        train_idx_cpu, test_idx_cpu = _stratified_split_indices(
            labels,
            int(n_train),
            split_generator,
            "cpu",
        )
    except ValueError as exc:
        if str(exc).startswith("infeasible_stratified_split"):
            return False
        raise
    return _classification_split_valid(labels[train_idx_cpu], labels[test_idx_cpu])


def _first_valid_classification_attempt_for_dataset(
    config: GeneratorConfig,
    *,
    plan: _FixedLayoutPlan,
    dataset_seed: int,
    requested_device: str,
    resolved_device: str,
    start_attempt: int = 0,
) -> int | None:
    """Return the first valid replay attempt for one dataset seed, if any."""

    data_seed = SeedManager(dataset_seed).child("data")
    shift_params = resolve_shift_runtime_params(config)
    noise_runtime_selection = _resolve_noise_runtime_selection(config, run_seed=data_seed)
    noise_spec = _noise_sampling_spec(noise_runtime_selection)
    attempts = max(1, int(config.filter.max_attempts))

    for attempt in range(max(0, int(start_attempt)), attempts):
        y_batch, _aux_meta_batch, _effective_resolved_device, _device_fallback_reason = (
            _generate_fixed_layout_label_batch_with_fallback(
                config,
                plan.layout,
                execution_plan=plan.execution_plan,
                dataset_seeds=[_attempt_seed(data_seed, attempt)],
                requested_device=requested_device,
                resolved_device=resolved_device,
                noise_sigma_multiplier=float(shift_params.variance_sigma_multiplier),
                noise_spec=noise_spec,
            )
        )
        if _raw_classification_labels_support_split(
            y_batch[0],
            dataset_seed=dataset_seed,
            attempt=attempt,
            n_train=int(plan.n_train),
        ):
            return int(attempt)
    return None


def _fixed_layout_plan_classification_attempt_plan(
    config: GeneratorConfig,
    *,
    plan: _FixedLayoutPlan,
    requested_device: str,
    resolved_device: str,
    run_seed: int,
    num_datasets: int = 1,
    batch_size: int = 1,
) -> tuple[int, ...] | None:
    """Return the first-valid attempt per dataset for a replayable classification run."""

    manager = SeedManager(run_seed)
    shift_params = resolve_shift_runtime_params(config)
    effective_batch_size = max(1, int(batch_size))
    dataset_index = 0
    attempt_plan: list[int] = []
    while dataset_index < num_datasets:
        chunk_size = min(effective_batch_size, num_datasets - dataset_index)
        dataset_seeds = [
            manager.child("dataset", dataset_index + offset) for offset in range(chunk_size)
        ]
        grouped_noise_runtime = _group_noise_runtime_chunk(
            config,
            dataset_seeds=dataset_seeds,
        )
        raw_batch_by_offset: list[tuple[torch.Tensor, int] | None] = [None] * chunk_size
        for group in grouped_noise_runtime:
            noise_spec = _noise_sampling_spec(group.selection)
            y_batch, _aux_meta_batch, _effective_resolved_device, _device_fallback_reason = (
                _generate_fixed_layout_label_batch_with_fallback(
                    config,
                    plan.layout,
                    execution_plan=plan.execution_plan,
                    dataset_seeds=group.generation_seeds,
                    requested_device=requested_device,
                    resolved_device=resolved_device,
                    noise_sigma_multiplier=float(shift_params.variance_sigma_multiplier),
                    noise_spec=noise_spec,
                )
            )
            for local_index, chunk_offset in enumerate(group.chunk_offsets):
                raw_batch_by_offset[chunk_offset] = (y_batch, int(local_index))
        for offset, dataset_seed in enumerate(dataset_seeds):
            raw_batch_entry = raw_batch_by_offset[offset]
            if raw_batch_entry is None:
                raise RuntimeError("Missing grouped raw batch entry for fixed-layout chunk offset.")
            y_batch, local_index = raw_batch_entry
            if _raw_classification_labels_support_split(
                y_batch[local_index],
                dataset_seed=dataset_seed,
                attempt=0,
                n_train=int(plan.n_train),
            ):
                attempt_plan.append(0)
                continue
            replay_attempt = _first_valid_classification_attempt_for_dataset(
                config,
                plan=plan,
                dataset_seed=dataset_seed,
                requested_device=requested_device,
                resolved_device=resolved_device,
                start_attempt=1,
            )
            if replay_attempt is None:
                return None
            attempt_plan.append(int(replay_attempt))
        dataset_index += chunk_size
    return tuple(attempt_plan)


def _fixed_layout_plan_supports_classification_run(
    config: GeneratorConfig,
    *,
    plan: _FixedLayoutPlan,
    requested_device: str,
    resolved_device: str,
    run_seed: int,
    num_datasets: int = 1,
    batch_size: int = 1,
) -> bool:
    """Return whether a classification plan can replay for the full requested run."""

    return (
        _fixed_layout_plan_classification_attempt_plan(
            config,
            plan=plan,
            requested_device=requested_device,
            resolved_device=resolved_device,
            run_seed=run_seed,
            num_datasets=num_datasets,
            batch_size=batch_size,
        )
        is not None
    )


def _fixed_layout_plan_supports_classification_replay(
    config: GeneratorConfig,
    *,
    plan: _FixedLayoutPlan,
    requested_device: str,
    resolved_device: str,
    validation_seed: int,
) -> bool:
    """Return whether a classification plan can replay under the fixed-layout engine."""

    return _fixed_layout_plan_supports_classification_run(
        config,
        plan=plan,
        requested_device=requested_device,
        resolved_device=resolved_device,
        run_seed=validation_seed,
        num_datasets=1,
        batch_size=1,
    )


def _sample_fixed_layout(
    config: GeneratorConfig,
    *,
    seed: int | None = None,
    device: str | None = None,
) -> _FixedLayoutPlan:
    """Sample one in-process layout plan for canonical fixed-layout generation."""

    run_seed = _resolve_run_seed(config, seed)
    requested_device = (device or config.runtime.device or "auto").lower()
    resolved_device = _resolve_device(config, device)
    attempts = max(1, int(config.filter.max_attempts))
    last_error = "unknown"

    for attempt in range(attempts):
        plan_seed = _attempt_seed(run_seed, attempt)
        plan = _sample_fixed_layout_once(
            config,
            seed=plan_seed,
            requested_device=requested_device,
            resolved_device=resolved_device,
        )
        if str(config.dataset.task) != "classification":
            return plan
        valid = False
        for validation_attempt in range(attempts):
            validation_seed = _attempt_seed(plan_seed, validation_attempt)
            if _fixed_layout_plan_supports_classification_replay(
                config,
                plan=plan,
                requested_device=requested_device,
                resolved_device=resolved_device,
                validation_seed=validation_seed,
            ):
                valid = True
                break
        if valid:
            return plan
        last_error = "invalid_class_split"

    raise ValueError(
        "Failed to sample a replayable fixed-layout classification plan after "
        f"{attempts} attempts. Last reason: {last_error}."
    )


def _generate_fixed_layout_bundle_with_retries(
    config: GeneratorConfig,
    *,
    plan: _FixedLayoutPlan,
    dataset_seed: int,
    requested_device: str,
    resolved_device: str,
    preserve_feature_schema: bool,
    start_attempt: int = 0,
) -> DatasetBundle:
    data_seed = SeedManager(dataset_seed).child("data")
    shift_params = resolve_shift_runtime_params(config)
    noise_runtime_selection = _resolve_noise_runtime_selection(config, run_seed=data_seed)
    noise_spec = _noise_sampling_spec(noise_runtime_selection)
    dtype = _torch_dtype(config)
    attempts = max(1, int(config.filter.max_attempts))
    initial_attempt = max(0, int(start_attempt))
    if initial_attempt >= attempts:
        raise ValueError(
            "Fixed-layout retry start attempt exceeds configured retry budget: "
            f"start_attempt={initial_attempt} max_attempts={attempts}."
        )
    last_error: str = "unknown"

    for attempt in range(initial_attempt, attempts):
        (
            x_batch,
            y_batch,
            aux_meta_batch,
            effective_resolved_device,
            device_fallback_reason,
        ) = _generate_fixed_layout_graph_batch_with_fallback(
            config,
            plan.layout,
            execution_plan=plan.execution_plan,
            dataset_seeds=[_attempt_seed(data_seed, attempt)],
            requested_device=requested_device,
            resolved_device=resolved_device,
            noise_sigma_multiplier=float(shift_params.variance_sigma_multiplier),
            noise_spec=noise_spec,
        )
        try:
            return _finalize_generated_tensors(
                config,
                plan.layout,
                seed=dataset_seed,
                attempt=attempt,
                attempts_used=attempt + 1,
                device=effective_resolved_device,
                n_train=int(plan.n_train),
                n_test=int(plan.n_test),
                requested_device=requested_device,
                resolved_device=effective_resolved_device,
                device_fallback_reason=device_fallback_reason,
                x=x_batch[0],
                y=y_batch[0],
                aux_meta=aux_meta_batch[0],
                shift_params=shift_params,
                noise_runtime_selection=noise_runtime_selection,
                dtype=dtype,
                preserve_feature_schema=preserve_feature_schema,
            )
        except ValueError as exc:
            if str(exc) == "invalid_class_split":
                last_error = "invalid_class_split"
                continue
            raise

    raise ValueError(
        "Failed to generate a valid fixed-layout dataset after "
        f"{attempts} attempts. Last reason: {last_error}."
    )


def _generate_batch_with_plan_iter(
    config: GeneratorConfig,
    *,
    plan: _FixedLayoutPlan,
    num_datasets: int,
    seed: int | None = None,
    batch_size: int | None = None,
    classification_attempt_plan: tuple[int, ...] | None = None,
) -> Iterator[DatasetBundle]:
    """Yield datasets for one in-process fixed-layout plan."""

    if num_datasets < 0:
        raise ValueError(f"num_datasets must be >= 0, got {num_datasets}")
    if num_datasets == 0:
        return
    if classification_attempt_plan is not None and len(classification_attempt_plan) != num_datasets:
        raise ValueError(
            "Fixed-layout classification attempt plan length must match num_datasets: "
            f"attempt_plan={len(classification_attempt_plan)} num_datasets={num_datasets}"
        )

    requested_device = str(plan.requested_device)
    validated_resolved_device = str(plan.resolved_device)
    run_seed = _resolve_run_seed(config, seed)
    manager = SeedManager(run_seed)
    dtype = _torch_dtype(config)
    shift_params = resolve_shift_runtime_params(config)
    expected_schema: tuple[int, tuple[str, ...], tuple[int, ...]] | None = None
    finalization_context = _build_fixed_schema_finalization_context(
        config,
        plan.layout,
        n_train=int(plan.n_train),
        n_test=int(plan.n_test),
        shift_params=shift_params,
    )

    effective_batch_size = _resolve_fixed_layout_batch_size(
        plan,
        num_datasets=num_datasets,
        batch_size=batch_size,
        target_cells=_effective_fixed_layout_target_cells(config),
    )
    dataset_index = 0
    while dataset_index < num_datasets:
        chunk_size = min(effective_batch_size, num_datasets - dataset_index)
        dataset_seeds = [
            manager.child("dataset", dataset_index + offset) for offset in range(chunk_size)
        ]
        chunk_attempts = (
            list(classification_attempt_plan[dataset_index : dataset_index + chunk_size])
            if classification_attempt_plan is not None
            else [0] * chunk_size
        )
        grouped_noise_runtime = _group_noise_runtime_chunk(
            config,
            dataset_seeds=dataset_seeds,
        )
        grouped_raw_batches = _generate_grouped_raw_batches(
            config,
            plan.layout,
            execution_plan=plan.execution_plan,
            grouped_noise_runtime=grouped_noise_runtime,
            requested_device=requested_device,
            resolved_device=validated_resolved_device,
            noise_sigma_multiplier=float(shift_params.variance_sigma_multiplier),
        )
        raw_batch_by_offset: list[tuple[_GroupedRawBatch, int, DatasetBundle | None] | None] = [
            None
        ] * chunk_size
        for grouped_batch in grouped_raw_batches:
            group_dataset_seeds = [
                int(dataset_seeds[int(chunk_offset)])
                for chunk_offset in grouped_batch.chunk_offsets
            ]
            finalized_group = _finalize_generated_chunk_preserve_schema(
                config,
                plan.layout,
                context=finalization_context,
                seeds=group_dataset_seeds,
                attempt=0,
                attempts_used=1,
                device=grouped_batch.effective_resolved_device,
                n_train=int(plan.n_train),
                n_test=int(plan.n_test),
                requested_device=requested_device,
                resolved_device=grouped_batch.effective_resolved_device,
                device_fallback_reason=grouped_batch.device_fallback_reason,
                x=grouped_batch.x_batch,
                y=grouped_batch.y_batch,
                aux_meta_batch=grouped_batch.aux_meta_batch,
                noise_runtime_selection=grouped_batch.selection,
                dtype=dtype,
            )
            for local_index, chunk_offset in enumerate(grouped_batch.chunk_offsets):
                raw_batch_by_offset[chunk_offset] = (
                    grouped_batch,
                    int(local_index),
                    finalized_group[local_index],
                )
        for offset, dataset_seed in enumerate(dataset_seeds):
            raw_batch_entry = raw_batch_by_offset[offset]
            if raw_batch_entry is None:
                raise RuntimeError("Missing grouped raw batch entry for fixed-layout chunk offset.")
            _grouped_batch, _local_index, bundle = raw_batch_entry
            if bundle is None:
                bundle = _generate_fixed_layout_bundle_with_retries(
                    config,
                    plan=plan,
                    dataset_seed=dataset_seed,
                    requested_device=requested_device,
                    resolved_device=validated_resolved_device,
                    preserve_feature_schema=True,
                    start_attempt=chunk_attempts[offset],
                )
            _annotate_fixed_layout_metadata(bundle, plan=plan)
            schema = _extract_emitted_schema_signature(bundle)
            if expected_schema is None:
                expected_schema = schema
            elif schema != expected_schema:
                raise ValueError(
                    "Fixed-layout schema mismatch: emitted dataset does not match "
                    "the first fixed-layout bundle schema."
                )
            yield bundle
        dataset_index += chunk_size


__all__ = [
    "CanonicalFixedLayoutRun",
    "_fixed_layout_plan_supports_classification_run",
    "_generate_batch_with_plan_iter",
    "_resolve_fixed_layout_batch_size",
    "_sample_fixed_layout",
    "prepare_canonical_fixed_layout_run",
    "realize_generation_config_for_run",
]
