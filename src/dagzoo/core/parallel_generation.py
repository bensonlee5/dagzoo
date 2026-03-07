"""Internal helpers for local multi-worker generation orchestration."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import suppress
from dataclasses import dataclass
import multiprocessing as mp
from multiprocessing.context import BaseContext
from multiprocessing.process import BaseProcess
import os
import queue
import sys
import time
import traceback
from typing import Any

import numpy as np
import torch

from dagzoo.config import GeneratorConfig
from dagzoo.core import generation_context as _generation_context
from dagzoo.core import generation_engine as _generation_engine
from dagzoo.core.worker_partition import iter_worker_dataset_seeds
from dagzoo.math_utils import to_numpy as _to_numpy
from dagzoo.types import DatasetBundle

_QUEUE_POLL_TIMEOUT_S = 0.05
_PROCESS_JOIN_TIMEOUT_S = 0.5
_CLEAN_EXIT_QUEUE_DRAIN_TIMEOUT_S = 1.0


class ParallelGenerationConfigError(ValueError):
    """Raised when local parallel generation is requested with an unsupported config."""


class ParallelGenerationWorkerError(RuntimeError):
    """Raised when one spawned worker fails during local parallel generation."""

    def __init__(
        self,
        *,
        worker_index: int,
        dataset_index: int | None,
        exception_type: str,
        exception_message: str,
        traceback_text: str,
    ) -> None:
        self.worker_index = int(worker_index)
        self.dataset_index = None if dataset_index is None else int(dataset_index)
        self.exception_type = str(exception_type)
        self.exception_message = str(exception_message)
        self.traceback_text = str(traceback_text)

        location = f"worker {self.worker_index}"
        if self.dataset_index is not None:
            location += f" while producing dataset_index={self.dataset_index}"
        message = (
            f"Parallel generation {location} failed with "
            f"{self.exception_type}: {self.exception_message}"
        )
        if self.traceback_text:
            message = f"{message}\nRemote traceback:\n{self.traceback_text}"
        super().__init__(message)


@dataclass(slots=True)
class _WorkerRunSpec:
    worker_index: int
    local_worker_count: int
    num_datasets: int
    run_seed: int
    requested_device: str
    resolved_device: str
    config_payload: dict[str, Any]
    queue_timeout_s: float
    torch_intraop_threads: int


@dataclass(slots=True)
class _WorkerResultMessage:
    worker_index: int
    dataset_index: int
    bundle_payload: dict[str, Any]


@dataclass(slots=True)
class _WorkerDoneMessage:
    worker_index: int


@dataclass(slots=True)
class _WorkerErrorMessage:
    worker_index: int
    dataset_index: int | None
    exception_type: str
    exception_message: str
    traceback_text: str


def active_worker_count(worker_count: int, num_datasets: int) -> int:
    """Return the number of worker partitions that can emit work for a run."""

    return min(int(worker_count), int(num_datasets))


def _local_parallel_worker_capacity() -> int:
    """Return the host-local CPU capacity available to the benchmark coordinator."""

    sched_getaffinity = getattr(os, "sched_getaffinity", None)
    if sched_getaffinity is not None:
        try:
            return max(1, len(sched_getaffinity(0)))
        except OSError:
            pass

    cpu_count = os.cpu_count()
    return max(1, int(cpu_count or 1))


def effective_local_parallel_worker_count(worker_count: int, num_datasets: int) -> int:
    """Return the local worker count after capping to host CPU capacity."""

    return min(active_worker_count(worker_count, num_datasets), _local_parallel_worker_capacity())


def _worker_torch_intraop_threads(local_worker_count: int) -> int:
    """Return the per-process Torch intra-op thread budget for spawned workers."""

    return max(1, _local_parallel_worker_capacity() // max(1, int(local_worker_count)))


def _result_queue_capacities(local_worker_count: int, buffer_budget: int) -> list[int]:
    """Split the total buffered-result budget across local worker queues."""

    worker_count = max(1, int(local_worker_count))
    effective_budget = max(int(buffer_budget), worker_count)
    base_capacity, remainder = divmod(effective_budget, worker_count)
    return [
        base_capacity + (1 if worker_index < remainder else 0)
        for worker_index in range(worker_count)
    ]


def _build_spawn_context() -> BaseContext:
    """Return the multiprocessing context used for local parallel generation."""

    return mp.get_context("spawn")


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


def _configure_worker_torch_threads(*, intraop_threads: int) -> None:
    """Cap Torch CPU threading inside one spawned local worker process."""

    torch.set_num_threads(max(1, int(intraop_threads)))
    with suppress(RuntimeError):
        torch.set_num_interop_threads(1)


def _put_result_message(
    result_queue: Any,
    stop_event: Any,
    *,
    message: _WorkerResultMessage,
    timeout: float,
) -> bool:
    """Put one result payload onto one worker-local result queue with shutdown awareness."""

    while True:
        if stop_event.is_set():
            return False
        try:
            result_queue.put(message, timeout=timeout)
            return True
        except queue.Full:
            continue
        except (BrokenPipeError, EOFError, OSError, ValueError):
            return False


def _put_control_message(
    control_queue: Any,
    *,
    message: _WorkerDoneMessage | _WorkerErrorMessage,
) -> None:
    """Best-effort control-plane message delivery from a spawned worker."""

    with suppress(BrokenPipeError, EOFError, OSError, ValueError):
        control_queue.put(message)


def _serialize_bundle_for_ipc(bundle: DatasetBundle) -> dict[str, Any]:
    """Convert one generated bundle into a multiprocessing-safe payload."""

    return {
        "X_train": np.ascontiguousarray(_to_numpy(bundle.X_train)),
        "y_train": np.ascontiguousarray(_to_numpy(bundle.y_train)),
        "X_test": np.ascontiguousarray(_to_numpy(bundle.X_test)),
        "y_test": np.ascontiguousarray(_to_numpy(bundle.y_test)),
        "feature_types": [str(value) for value in bundle.feature_types],
        "metadata": dict(bundle.metadata),
    }


def _deserialize_bundle_from_ipc(payload: dict[str, Any]) -> DatasetBundle:
    """Rebuild one dataset bundle from a multiprocessing payload."""

    return DatasetBundle(
        X_train=payload["X_train"],
        y_train=payload["y_train"],
        X_test=payload["X_test"],
        y_test=payload["y_test"],
        feature_types=[str(value) for value in payload["feature_types"]],
        metadata=dict(payload["metadata"]),
    )


def _parallel_worker_entrypoint(
    spec: _WorkerRunSpec,
    result_queue: Any,
    control_queue: Any,
    stop_event: Any,
) -> None:
    """Generate one worker partition inside a spawned child process."""

    current_dataset_index: int | None = None
    try:
        _configure_worker_torch_threads(intraop_threads=spec.torch_intraop_threads)
        worker_config = GeneratorConfig.from_dict(spec.config_payload)
        for current_dataset_index, dataset_seed in iter_worker_dataset_seeds(
            run_seed=spec.run_seed,
            num_datasets=spec.num_datasets,
            worker_count=spec.local_worker_count,
            worker_index=spec.worker_index,
        ):
            if stop_event.is_set():
                break
            try:
                bundle = _generation_engine._generate_one_seeded(
                    worker_config,
                    seed=dataset_seed,
                    requested_device=spec.requested_device,
                    resolved_device=spec.resolved_device,
                )
                if not _put_result_message(
                    result_queue,
                    stop_event,
                    message=_WorkerResultMessage(
                        worker_index=spec.worker_index,
                        dataset_index=current_dataset_index,
                        bundle_payload=_serialize_bundle_for_ipc(bundle),
                    ),
                    timeout=spec.queue_timeout_s,
                ):
                    break
            except BaseException:
                raise
        _put_control_message(
            control_queue,
            message=_WorkerDoneMessage(worker_index=spec.worker_index),
        )
    except BaseException as exc:
        stop_event.set()
        _put_control_message(
            control_queue,
            message=_WorkerErrorMessage(
                worker_index=spec.worker_index,
                dataset_index=current_dataset_index,
                exception_type=exc.__class__.__name__,
                exception_message=str(exc),
                traceback_text="".join(traceback.format_exception(exc)),
            ),
        )


def _spawn_parallel_workers(
    *,
    ctx: Any,
    worker_specs: list[_WorkerRunSpec],
    result_queues: list[Any],
    control_queue: Any,
    stop_event: Any,
) -> list[BaseProcess]:
    """Start spawned local generation worker processes."""

    processes: list[BaseProcess] = []
    started_processes: list[BaseProcess] = []
    try:
        for spec in worker_specs:
            process = ctx.Process(
                target=_parallel_worker_entrypoint,
                args=(
                    spec,
                    result_queues[spec.worker_index],
                    control_queue,
                    stop_event,
                ),
                name=f"dagzoo-parallel-gen-{spec.worker_index}",
            )
            process.start()
            processes.append(process)
            started_processes.append(process)
        return processes
    except BaseException:
        stop_event.set()
        _terminate_processes(started_processes)
        raise


def _worker_error_from_message(message: _WorkerErrorMessage) -> ParallelGenerationWorkerError:
    """Rehydrate one remote worker failure into a local exception."""

    return ParallelGenerationWorkerError(
        worker_index=message.worker_index,
        dataset_index=message.dataset_index,
        exception_type=message.exception_type,
        exception_message=message.exception_message,
        traceback_text=message.traceback_text,
    )


def _drain_control_queue(
    control_queue: Any,
    *,
    done_workers: set[int],
) -> ParallelGenerationWorkerError | None:
    """Drain available control-plane messages and return the first worker error, if any."""

    while True:
        try:
            message = control_queue.get_nowait()
        except queue.Empty:
            return None
        if isinstance(message, _WorkerDoneMessage):
            done_workers.add(int(message.worker_index))
            continue
        if isinstance(message, _WorkerErrorMessage):
            return _worker_error_from_message(message)
        raise RuntimeError(
            "Parallel generation received an unexpected control message "
            f"{type(message).__name__!r}."
        )


def _target_worker_exited_cleanly_or_raise(
    *,
    processes: list[BaseProcess],
    target_worker_index: int,
    next_dataset_index: int,
) -> bool:
    """Return whether the target worker has exited cleanly."""

    target_process = processes[target_worker_index]
    exitcode = target_process.exitcode
    if exitcode is None:
        return False

    if exitcode != 0:
        raise RuntimeError(
            "Parallel generation worker exited unexpectedly before producing "
            f"dataset index {next_dataset_index}: "
            f"worker {target_worker_index} exitcode={int(exitcode)}."
        )

    return True


def _drain_target_queue_after_clean_exit(
    *,
    target_queue: Any,
    control_queue: Any,
    done_workers: set[int],
    drain_timeout_s: float,
) -> _WorkerResultMessage | None:
    """Poll one target worker queue for in-flight messages after a clean child exit."""

    deadline = time.monotonic() + max(0.0, float(drain_timeout_s))
    while True:
        worker_error = _drain_control_queue(
            control_queue,
            done_workers=done_workers,
        )
        if worker_error is not None:
            raise worker_error

        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return None

        try:
            return target_queue.get(timeout=min(_QUEUE_POLL_TIMEOUT_S, remaining))
        except queue.Empty:
            continue


def _terminate_processes(processes: list[BaseProcess]) -> None:
    """Stop spawned worker processes, escalating from join to terminate to kill."""

    for process in processes:
        process.join(timeout=_PROCESS_JOIN_TIMEOUT_S)
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=_PROCESS_JOIN_TIMEOUT_S)
    for process in processes:
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
    for process in processes:
        process.join(timeout=_PROCESS_JOIN_TIMEOUT_S)


def _close_process_queue(queue_obj: Any) -> None:
    """Close one multiprocessing queue and join its feeder thread."""

    with suppress(AttributeError, OSError, ValueError):
        queue_obj.close()
    with suppress(AttributeError, OSError, ValueError):
        queue_obj.join_thread()


def generate_parallel_batch_iter(
    config: GeneratorConfig,
    *,
    num_datasets: int,
    seed: int | None = None,
    device: str | None = None,
    max_buffered_results: int | None = None,
) -> Generator[DatasetBundle, None, None]:
    """Yield datasets in global order using spawned local worker processes."""

    if num_datasets < 0:
        raise ValueError(f"num_datasets must be >= 0, got {num_datasets}")
    if num_datasets == 0:
        return

    requested_device, resolved_device, worker_count = _validate_parallel_generation_request(
        config,
        device=device,
    )
    local_worker_count = effective_local_parallel_worker_count(worker_count, num_datasets)
    buffer_budget = max(1, int(max_buffered_results or (local_worker_count * 2)))
    run_seed = _generation_context._resolve_run_seed(config, seed)
    ctx = _build_spawn_context()
    result_queues = [
        ctx.Queue(maxsize=queue_capacity)
        for queue_capacity in _result_queue_capacities(local_worker_count, buffer_budget)
    ]
    control_queue = ctx.Queue()
    stop_event = ctx.Event()

    worker_specs = [
        _WorkerRunSpec(
            worker_index=worker_index,
            local_worker_count=local_worker_count,
            num_datasets=num_datasets,
            run_seed=run_seed,
            requested_device=requested_device,
            resolved_device=resolved_device,
            config_payload=config.to_dict(),
            queue_timeout_s=_QUEUE_POLL_TIMEOUT_S,
            torch_intraop_threads=_worker_torch_intraop_threads(local_worker_count),
        )
        for worker_index in range(local_worker_count)
    ]
    try:
        processes = _spawn_parallel_workers(
            ctx=ctx,
            worker_specs=worker_specs,
            result_queues=result_queues,
            control_queue=control_queue,
            stop_event=stop_event,
        )
    except BaseException:
        for result_queue in result_queues:
            _close_process_queue(result_queue)
        _close_process_queue(control_queue)
        raise

    done_workers: set[int] = set()
    try:
        for next_dataset_index in range(num_datasets):
            target_worker_index = next_dataset_index % local_worker_count
            target_queue = result_queues[target_worker_index]
            while True:
                worker_error = _drain_control_queue(
                    control_queue,
                    done_workers=done_workers,
                )
                if worker_error is not None:
                    raise worker_error
                try:
                    message = target_queue.get(timeout=_QUEUE_POLL_TIMEOUT_S)
                except queue.Empty:
                    worker_error = _drain_control_queue(
                        control_queue,
                        done_workers=done_workers,
                    )
                    if worker_error is not None:
                        raise worker_error
                    if not _target_worker_exited_cleanly_or_raise(
                        processes=processes,
                        target_worker_index=target_worker_index,
                        next_dataset_index=next_dataset_index,
                    ):
                        continue
                    message = _drain_target_queue_after_clean_exit(
                        target_queue=target_queue,
                        control_queue=control_queue,
                        done_workers=done_workers,
                        drain_timeout_s=_CLEAN_EXIT_QUEUE_DRAIN_TIMEOUT_S,
                    )
                    if message is None:
                        if target_worker_index not in done_workers:
                            raise RuntimeError(
                                "Parallel generation worker exited without a completion signal "
                                f"before producing dataset index {next_dataset_index}: worker "
                                f"{target_worker_index}."
                            )
                        raise RuntimeError(
                            "Parallel generation worker completed before producing the expected "
                            f"dataset index {next_dataset_index}: worker {target_worker_index}."
                        )

                if not isinstance(message, _WorkerResultMessage):
                    raise RuntimeError(
                        "Parallel generation received an unexpected result message "
                        f"{type(message).__name__!r}."
                    )
                dataset_index = int(message.dataset_index)
                if dataset_index != next_dataset_index:
                    raise RuntimeError(
                        "Parallel generation yielded an unexpected dataset index from worker "
                        f"{target_worker_index}: expected {next_dataset_index}, "
                        f"got {dataset_index}."
                    )
                yield _deserialize_bundle_from_ipc(message.bundle_payload)
                break
    finally:
        stop_event.set()
        current_exception = sys.exc_info()[1]
        teardown_error: BaseException | None = None
        _terminate_processes(processes)
        if current_exception is None:
            worker_error = _drain_control_queue(
                control_queue,
                done_workers=done_workers,
            )
            if worker_error is not None:
                teardown_error = worker_error
            else:
                for worker_index, process in enumerate(processes):
                    if process.exitcode not in (None, 0):
                        teardown_error = RuntimeError(
                            "Parallel generation worker exited unexpectedly during teardown: "
                            f"worker {worker_index} exitcode={int(process.exitcode or 0)}."
                        )
                        break
        for result_queue in result_queues:
            _close_process_queue(result_queue)
        _close_process_queue(control_queue)
        if current_exception is None and teardown_error is not None:
            raise teardown_error
