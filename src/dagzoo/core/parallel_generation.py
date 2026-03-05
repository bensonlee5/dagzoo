"""Internal helpers for local multi-worker generation orchestration."""

from __future__ import annotations

import copy
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import queue
import threading

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


@dataclass(slots=True)
class _WorkerError:
    worker_index: int
    error: BaseException


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
    run_seed = _generation_context._resolve_run_seed(config, seed)
    buffer_budget = max(1, int(max_buffered_results or (worker_count * 2)))
    per_worker_capacity = max(1, (buffer_budget + worker_count - 1) // worker_count)
    bundle_queues = [
        queue.Queue[_BundleResult](maxsize=per_worker_capacity) for _ in range(worker_count)
    ]
    error_queue: queue.Queue[_WorkerError] = queue.Queue()
    stop_event = threading.Event()
    consumer_closed_event = threading.Event()

    def _put_bundle(worker_index: int, item: _BundleResult) -> bool:
        while True:
            if consumer_closed_event.is_set():
                return False
            if stop_event.is_set():
                return False
            try:
                bundle_queues[worker_index].put(item, timeout=0.05)
                return True
            except queue.Full:
                continue

    def _run_worker(local_worker_index: int) -> None:
        worker_config = copy.deepcopy(config)
        worker_config.runtime.worker_index = local_worker_index
        try:
            for dataset_index, dataset_seed in iter_worker_dataset_seeds(
                run_seed=run_seed,
                num_datasets=num_datasets,
                worker_count=worker_count,
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
            stop_event.set()
            if not consumer_closed_event.is_set():
                error_queue.put_nowait(_WorkerError(worker_index=local_worker_index, error=exc))

    first_error: BaseException | None = None

    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="dagzoo-parallel-gen",
    ) as executor:
        futures = [executor.submit(_run_worker, worker_idx) for worker_idx in range(worker_count)]
        try:
            for next_dataset_index in range(num_datasets):
                target_worker_index = next_dataset_index % worker_count
                target_queue = bundle_queues[target_worker_index]
                target_future = futures[target_worker_index]
                while True:
                    if first_error is None:
                        try:
                            worker_error = error_queue.get_nowait()
                        except queue.Empty:
                            pass
                        else:
                            first_error = worker_error.error
                            stop_event.set()
                    if first_error is not None:
                        raise first_error
                    try:
                        item = target_queue.get(timeout=0.05)
                    except queue.Empty:
                        if target_future.done():
                            worker_exc = target_future.exception()
                            if worker_exc is not None:
                                stop_event.set()
                                raise worker_exc
                            raise RuntimeError(
                                "Parallel generation worker ended before producing the expected "
                                f"dataset index {next_dataset_index}."
                            )
                        continue
                    if item.dataset_index != next_dataset_index:
                        raise RuntimeError(
                            "Parallel generation yielded an unexpected dataset index from worker "
                            f"{target_worker_index}: expected {next_dataset_index}, "
                            f"got {item.dataset_index}."
                        )
                    yield item.bundle
                    break
        finally:
            consumer_closed_event.set()
            stop_event.set()
            for future in futures:
                future.result()

    if first_error is not None:
        raise first_error
