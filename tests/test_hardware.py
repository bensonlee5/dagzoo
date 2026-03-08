import math
import pytest

from dagzoo.config import GeneratorConfig
from dagzoo.hardware import HardwareInfo, detect_hardware, get_peak_flops
from dagzoo.hardware_policy import apply_hardware_policy, list_hardware_policies


def test_peak_flops_lookup() -> None:
    assert get_peak_flops("NVIDIA H100 PCIe") == 756e12
    assert get_peak_flops("NVIDIA H100 SXM") == 989e12
    assert get_peak_flops("NVIDIA RTX 4090") == 165.2e12


def test_peak_flops_fallback() -> None:
    val = get_peak_flops("Some Unknown GPU XYZ")
    assert math.isinf(val)


def test_hardware_policy_none_is_immutable_and_noop() -> None:
    cfg = GeneratorConfig()
    original = cfg.to_dict()

    hw = HardwareInfo(
        backend="cpu",
        requested_device="auto",
        device_name="cpu",
        total_memory_gb=None,
        peak_flops=float("inf"),
        tier="cpu",
    )
    effective = apply_hardware_policy(cfg, hw, policy_name="none")

    assert cfg.to_dict() == original
    assert effective.to_dict() == original
    assert effective is not cfg


def test_hardware_policy_cuda_tiered_applied_when_h100_profile() -> None:
    cfg = GeneratorConfig()
    cfg.runtime.device = "cuda"
    original_train = cfg.dataset.n_train

    hw = HardwareInfo(
        backend="cuda",
        requested_device="cuda",
        device_name="NVIDIA H100 SXM",
        total_memory_gb=80.0,
        peak_flops=989e12,
        tier="cuda_h100",
    )
    effective = apply_hardware_policy(cfg, hw, policy_name="cuda_tiered_v1")

    assert cfg.dataset.n_train == original_train
    assert effective.dataset.n_train >= 4096
    assert effective.benchmark.preset_name == "cuda_h100_auto"
    assert effective.runtime.fixed_layout_target_cells == 160_000_000


def test_hardware_policy_raises_low_fixed_layout_target_cells_to_memory_scaled_floor() -> None:
    cfg = GeneratorConfig()
    cfg.runtime.device = "cuda"
    cfg.runtime.fixed_layout_target_cells = 16_000_000

    hw = HardwareInfo(
        backend="cuda",
        requested_device="cuda",
        device_name="NVIDIA RTX 4090",
        total_memory_gb=24.0,
        peak_flops=165.2e12,
        tier="cuda_desktop",
    )
    effective = apply_hardware_policy(cfg, hw, policy_name="cuda_tiered_v1")

    assert effective.runtime.fixed_layout_target_cells == 48_000_000


def test_hardware_policy_preserves_explicit_high_fixed_layout_target_cells_override() -> None:
    cfg = GeneratorConfig()
    cfg.runtime.device = "cuda"
    cfg.runtime.fixed_layout_target_cells = 96_000_000

    hw = HardwareInfo(
        backend="cuda",
        requested_device="cuda",
        device_name="NVIDIA RTX 4090",
        total_memory_gb=24.0,
        peak_flops=165.2e12,
        tier="cuda_desktop",
    )
    effective = apply_hardware_policy(cfg, hw, policy_name="cuda_tiered_v1")

    assert effective.runtime.fixed_layout_target_cells == 96_000_000


def test_hardware_policy_cuda_datacenter_scales_fixed_layout_target_cells_with_memory() -> None:
    cfg = GeneratorConfig()
    cfg.runtime.device = "cuda"

    hw = HardwareInfo(
        backend="cuda",
        requested_device="cuda",
        device_name="NVIDIA A100 80GB",
        total_memory_gb=80.0,
        peak_flops=312e12,
        tier="cuda_datacenter",
    )
    effective = apply_hardware_policy(cfg, hw, policy_name="cuda_tiered_v1")

    assert effective.runtime.fixed_layout_target_cells == 160_000_000


def test_hardware_policy_cuda_unknown_fallback_applies_fixed_layout_target_floor() -> None:
    cfg = GeneratorConfig()
    cfg.runtime.device = "cuda"

    hw = HardwareInfo(
        backend="cuda",
        requested_device="cuda",
        device_name="Unknown GPU XYZ",
        total_memory_gb=None,
        peak_flops=float("inf"),
        tier="cuda_unknown_fallback",
    )
    effective = apply_hardware_policy(cfg, hw, policy_name="cuda_tiered_v1")

    assert effective.benchmark.preset_name == "cuda_unknown_fallback"
    assert effective.runtime.fixed_layout_target_cells == 48_000_000


def test_hardware_policy_unknown_name_raises() -> None:
    cfg = GeneratorConfig()
    hw = HardwareInfo(
        backend="cpu",
        requested_device="cpu",
        device_name="cpu",
        total_memory_gb=None,
        peak_flops=float("inf"),
        tier="cpu",
    )
    with pytest.raises(ValueError, match="Unknown hardware policy"):
        _ = apply_hardware_policy(cfg, hw, policy_name="missing")


def test_hardware_policy_registry_includes_builtins() -> None:
    names = list_hardware_policies()
    assert "none" in names
    assert "cuda_tiered_v1" in names


def test_detect_hardware_cpu_backend_label() -> None:
    hw = detect_hardware("cpu")
    assert hw.backend == "cpu"
