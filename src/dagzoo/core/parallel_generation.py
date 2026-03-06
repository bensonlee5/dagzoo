"""Internal helpers for local multi-worker generation orchestration."""

from __future__ import annotations

import copy
from contextlib import contextmanager
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import queue
import sys
import threading

import torch

from dagzoo.config import GeneratorConfig
from dagzoo.core import generation_context as _generation_context
from dagzoo.core import generation_engine as _generation_engine
from dagzoo.core.worker_partition import iter_worker_dataset_seeds
from dagzoo.types import DatasetBundle


class ParallelGenerationConfigError(ValueError):
    """Raised when local parallel generation is requested with an unsupported config."""


@dataclass(slots=True)
class _BundleResult:
    dataset_index: int
    bundle: DatasetBundle


@contextmanager
def _cap_torch_intraop_threads(active_worker_count: int) -> Iterator[None]:
    """Prevent local worker threads from oversubscribing Torch CPU kernels."""

    if active_worker_count <= 1:
        yield
        return

    original_threads = int(torch.get_num_threads())
    if original_threads < active_worker_count:
        raise ParallelGenerationConfigError(
            "Local parallel generation requires torch.get_num_threads() >= the active "
            "worker count. Got "
            f"torch_threads={original_threads} < active_worker_count={active_worker_count}; "
            "lower runtime.worker_count, lower num_datasets, or increase Torch CPU threads."
        )
    capped_threads = max(1, original_threads // active_worker_count)

    # Torch thread settings are process-global, so keep the cap scoped to the local
    # multi-worker benchmark path and restore it after all worker threads have joined.
    torch.set_num_threads(capped_threads)
    try:
        yield
    finally:
        torch.set_num_threads(original_threads)


def _validate_parallel_generation_request(
    config: GeneratorConfig,
    *,
    device: str | None,
) -> tuple[str, str, int]:
    """Validate the local parallel generation request and return resolved device info."""

    requested_device = (device or config.runtime.device or "auto").lower()
    resolved_device = _generation_context._resolve_device(config, device)
    worker_count = int(config.runtime.worker_count)
    worker_index = int(config.runtime.worker_index)

    if worker_count <= 1:
        raise ParallelGenerationConfigError(
            "Parallel generation requires runtime.worker_count > 1."
        )
    if worker_index != 0:
        raise ParallelGenerationConfigError(
            "Local parallel generation requires runtime.worker_index == 0. "
            f"Got worker_index={worker_index} with worker_count={worker_count}."
        )
    if resolved_device != "cpu":
        raise ParallelGenerationConfigError(
            "Local parallel generation currently supports resolved device 'cpu' only. "
            f"Got requested_device='{requested_device}' resolved_device='{resolved_device}'."
        )

    return requested_device, resolved_device, worker_count


def generate_parallel_batch_iter(
    config: GeneratorConfig,
    *,
    num_datasets: int,
    seed: int | None = None,
    device: str | None = None,
    max_buffered_results: int | None = None,
) -> Iterator[DatasetBundle]:
    """Yield datasets in global order using a local thread pool over worker partitions."""

    if num_datasets < 0:
        raise ValueError(f"num_datasets must be >= 0, got {num_datasets}")
    if num_datasets == 0:
        return

    requested_device, resolved_device, worker_count = _validate_parallel_generation_request(
        config,
        device=device,
    )
    active_worker_count = min(worker_count, num_datasets)
    run_seed = _generation_context._resolve_run_seed(config, seed)
    buffer_budget = max(1, int(max_buffered_results or (active_worker_count * 2)))
    per_worker_capacity = max(1, (buffer_budget + active_worker_count - 1) // active_worker_count)
    bundle_queues = [
        queue.Queue[_BundleResult](maxsize=per_worker_capacity) for _ in range(active_worker_count)
    ]
    # Each worker writes its own slot at most once; the consumer only reads stable references.
    worker_errors: list[BaseException | None] = [None] * active_worker_count
    first_recorded_error: list[BaseException | None] = [None]
    first_recorded_error_lock = threading.Lock()
    # Worker failures stop new dataset generation, but in-flight bundles may still be queued.
    stop_event = threading.Event()
    consumer_closed_event = threading.Event()

    def _target_worker_error(target_worker_index: int) -> BaseException | None:
        return worker_errors[target_worker_index]

    def _first_recorded_worker_error() -> BaseException | None:
        return first_recorded_error[0]

    def _try_get_buffered_item(target_queue: queue.Queue[_BundleResult]) -> _BundleResult | None:
        try:
            return target_queue.get_nowait()
        except queue.Empty:
            return None

    def _bundle_from_item(
        item: _BundleResult,
        *,
        target_worker_index: int,
        next_dataset_index: int,
    ) -> DatasetBundle:
        if item.dataset_index != next_dataset_index:
            raise RuntimeError(
                "Parallel generation yielded an unexpected dataset index from worker "
                f"{target_worker_index}: expected {next_dataset_index}, "
                f"got {item.dataset_index}."
            )
        return item.bundle

    def _put_bundle(worker_index: int, item: _BundleResult) -> bool:
        while True:
            if consumer_closed_event.is_set():
                return False
            try:
                bundle_queues[worker_index].put(item, timeout=0.05)
                return True
            except queue.Full:
                continue

    def _run_worker(local_worker_index: int) -> None:
        try:
            worker_config = copy.deepcopy(config)
            for dataset_index, dataset_seed in iter_worker_dataset_seeds(
                run_seed=run_seed,
                num_datasets=num_datasets,
                worker_count=active_worker_count,
                worker_index=local_worker_index,
            ):
                if stop_event.is_set():
                    break
                bundle = _generation_engine._generate_one_seeded(
                    worker_config,
                    seed=dataset_seed,
                    requested_device=requested_device,
                    resolved_device=resolved_device,
                )
                if not _put_bundle(
                    local_worker_index,
                    _BundleResult(dataset_index=dataset_index, bundle=bundle),
                ):
                    break
        except BaseException as exc:  # pragma: no cover - exercised via consumer raise path
            worker_errors[local_worker_index] = exc
            with first_recorded_error_lock:
                if first_recorded_error[0] is None:
                    first_recorded_error[0] = exc
            stop_event.set()

    with _cap_torch_intraop_threads(active_worker_count):
        with ThreadPoolExecutor(
            max_workers=active_worker_count,
            thread_name_prefix="dagzoo-parallel-gen",
        ) as executor:
            futures = [
                executor.submit(_run_worker, worker_idx)
                for worker_idx in range(active_worker_count)
            ]
            try:
                for next_dataset_index in range(num_datasets):
                    target_worker_index = next_dataset_index % active_worker_count
                    target_queue = bundle_queues[target_worker_index]
                    target_future = futures[target_worker_index]
                    while True:
                        item = _try_get_buffered_item(target_queue)
                        if item is not None:
                            yield _bundle_from_item(
                                item,
                                target_worker_index=target_worker_index,
                                next_dataset_index=next_dataset_index,
                            )
                            break
                        target_worker_error = _target_worker_error(target_worker_index)
                        if target_worker_error is not None:
                            raise target_worker_error
                        try:
                            item = target_queue.get(timeout=0.05)
                        except queue.Empty:
                            item = _try_get_buffered_item(target_queue)
                            if item is not None:
                                yield _bundle_from_item(
                                    item,
                                    target_worker_index=target_worker_index,
                                    next_dataset_index=next_dataset_index,
                                )
                                break
                            target_worker_error = _target_worker_error(target_worker_index)
                            if target_worker_error is not None:
                                raise target_worker_error
                            if target_future.done():
                                item = _try_get_buffered_item(target_queue)
                                if item is not None:
                                    yield _bundle_from_item(
                                        item,
                                        target_worker_index=target_worker_index,
                                        next_dataset_index=next_dataset_index,
                                    )
                                    break
                                target_worker_error = _target_worker_error(target_worker_index)
                                if target_worker_error is not None:
                                    raise target_worker_error
                                worker_error = _first_recorded_worker_error()
                                if worker_error is not None:
                                    raise worker_error
                                # Defensive fallback for unexpected thread-body failures that bypass
                                # the recorded worker error path.
                                worker_exc = target_future.exception()
                                if worker_exc is not None:
                                    raise worker_exc
                                raise RuntimeError(
                                    "Parallel generation worker ended before producing the expected "
                                    f"dataset index {next_dataset_index}."
                                )
                            continue
                        yield _bundle_from_item(
                            item,
                            target_worker_index=target_worker_index,
                            next_dataset_index=next_dataset_index,
                        )
                        break
            finally:
                consumer_closed_event.set()
                stop_event.set()
                current_exception = sys.exc_info()[1]
                teardown_error: BaseException | None = None
                for future in futures:
                    try:
                        future.result()
                    except BaseException as exc:  # pragma: no cover - requires teardown timing
                        if current_exception is None and teardown_error is None:
                            teardown_error = exc
                if current_exception is None and teardown_error is not None:
                    raise teardown_error
