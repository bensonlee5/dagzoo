import pytest
import torch

from dagzoo.hardware import HardwareInfo
from dagzoo.rng import KeyedRng, keyed_rng_from_generator


def make_generator(seed: int = 42) -> torch.Generator:
    """Create a seeded torch Generator on CPU for deterministic tests."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return g


def make_keyed_rng(generator: torch.Generator, *components: str | int) -> KeyedRng:
    """Consume one ambient draw and derive the same keyed root used by helpers."""

    return keyed_rng_from_generator(generator, *components)


def _build_mock_hardware(tier: str) -> HardwareInfo:
    normalized_tier = str(tier).strip().lower()
    if normalized_tier == "cpu":
        return HardwareInfo(
            backend="cpu",
            requested_device="cpu",
            device_name="cpu",
            total_memory_gb=None,
            peak_flops=float("inf"),
            tier="cpu",
        )
    if normalized_tier == "cuda_desktop":
        return HardwareInfo(
            backend="cuda",
            requested_device="cuda",
            device_name="NVIDIA RTX 4090",
            total_memory_gb=24.0,
            peak_flops=165.2e12,
            tier="cuda_desktop",
        )
    if normalized_tier == "cuda_datacenter":
        return HardwareInfo(
            backend="cuda",
            requested_device="cuda",
            device_name="NVIDIA A100 80GB",
            total_memory_gb=80.0,
            peak_flops=312e12,
            tier="cuda_datacenter",
        )
    if normalized_tier == "cuda_h100":
        return HardwareInfo(
            backend="cuda",
            requested_device="cuda",
            device_name="NVIDIA H100 SXM",
            total_memory_gb=80.0,
            peak_flops=989e12,
            tier="cuda_h100",
        )
    raise ValueError(f"Unsupported mock hardware tier {tier!r}.")


@pytest.fixture
def hardware_info_factory():
    """Return one helper that builds stable mock hardware profiles by tier."""

    return _build_mock_hardware


@pytest.fixture
def patch_detect_hardware(monkeypatch: pytest.MonkeyPatch, hardware_info_factory):
    """Patch one or more detect_hardware call sites to one stable mock tier."""

    def _patch(tier: str, *targets: str) -> HardwareInfo:
        hw = hardware_info_factory(tier)
        for target in targets:
            monkeypatch.setattr(target, lambda _requested_device, hw=hw: hw)
        return hw

    return _patch
