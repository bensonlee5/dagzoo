"""Hardware policy registry for explicit config scaling rules."""

from __future__ import annotations

import copy
from collections.abc import Callable

from dagzoo.config import GeneratorConfig
from dagzoo.hardware import HardwareInfo

HardwarePolicy = Callable[[GeneratorConfig, HardwareInfo], GeneratorConfig]


def _policy_none(config: GeneratorConfig, hw: HardwareInfo) -> GeneratorConfig:
    """No-op policy that preserves the provided configuration."""

    _ = hw
    return config


def _policy_cuda_tiered_v1(config: GeneratorConfig, hw: HardwareInfo) -> GeneratorConfig:
    """Apply coarse CUDA tier overrides modeled after prior runtime defaults."""

    if hw.backend != "cuda":
        return config

    if hw.tier == "cuda_h100":
        config.benchmark.preset_name = "cuda_h100_auto"
        config.dataset.n_train = max(config.dataset.n_train, 4096)
        config.dataset.n_test = max(config.dataset.n_test, 1024)
        config.dataset.n_features_max = max(config.dataset.n_features_max, 192)
        config.graph.n_nodes_max = max(config.graph.n_nodes_max, 64)
        config.output.shard_size = max(config.output.shard_size, 512)
        config.benchmark.num_datasets = max(config.benchmark.num_datasets, 5000)
        config.benchmark.warmup_datasets = max(config.benchmark.warmup_datasets, 50)
        return config

    if hw.tier == "cuda_datacenter":
        config.benchmark.preset_name = "cuda_datacenter_auto"
        config.dataset.n_train = max(config.dataset.n_train, 1536)
        config.dataset.n_test = max(config.dataset.n_test, 512)
        config.dataset.n_features_max = max(config.dataset.n_features_max, 96)
        config.graph.n_nodes_max = max(config.graph.n_nodes_max, 48)
        config.output.shard_size = max(config.output.shard_size, 256)
        config.benchmark.num_datasets = max(config.benchmark.num_datasets, 2500)
        config.benchmark.warmup_datasets = max(config.benchmark.warmup_datasets, 30)
        return config

    if hw.tier == "cuda_desktop":
        config.benchmark.preset_name = "cuda_desktop_auto"
        return config

    config.benchmark.preset_name = "cuda_unknown_fallback"
    return config


_HARDWARE_POLICY_REGISTRY: dict[str, HardwarePolicy] = {
    "none": _policy_none,
    "cuda_tiered_v1": _policy_cuda_tiered_v1,
}


def list_hardware_policies() -> tuple[str, ...]:
    """Return all registered hardware policy names."""

    return tuple(sorted(_HARDWARE_POLICY_REGISTRY))


def register_hardware_policy(
    name: str,
    policy: HardwarePolicy,
    *,
    overwrite: bool = False,
) -> None:
    """Register a named hardware policy callback."""

    normalized_name = str(name).strip().lower()
    if not normalized_name:
        raise ValueError("hardware policy name must be a non-empty string.")
    if normalized_name in _HARDWARE_POLICY_REGISTRY and not overwrite:
        raise ValueError(
            f"hardware policy '{normalized_name}' already exists; pass overwrite=True to replace it."
        )
    _HARDWARE_POLICY_REGISTRY[normalized_name] = policy


def apply_hardware_policy(
    config: GeneratorConfig,
    hw: HardwareInfo,
    *,
    policy_name: str = "none",
) -> GeneratorConfig:
    """Apply a named hardware policy without mutating the input config."""

    normalized_name = str(policy_name).strip().lower()
    policy = _HARDWARE_POLICY_REGISTRY.get(normalized_name)
    if policy is None:
        raise ValueError(
            "Unknown hardware policy "
            f"{policy_name!r}. Available: {', '.join(list_hardware_policies()) or '(none)'}."
        )

    base = copy.deepcopy(config)
    result = policy(base, hw)
    if not isinstance(result, GeneratorConfig):
        raise TypeError(
            f"hardware policy '{normalized_name}' must return GeneratorConfig, got {type(result)!r}."
        )
    result.validate_generation_constraints()
    return result


__all__ = [
    "HardwarePolicy",
    "apply_hardware_policy",
    "list_hardware_policies",
    "register_hardware_policy",
]
