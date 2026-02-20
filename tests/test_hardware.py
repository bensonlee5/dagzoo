import numpy as np

from cauchy_generator.config import GeneratorConfig
from cauchy_generator.hardware import HardwareInfo, apply_hardware_profile
from cauchy_generator.hardware import get_peak_flops


def test_peak_flops_lookup() -> None:
    assert get_peak_flops("NVIDIA H100 PCIe") == 756e12
    assert get_peak_flops("NVIDIA H100 SXM") == 989e12
    assert get_peak_flops("NVIDIA RTX 4090") == 165.2e12


def test_peak_flops_fallback() -> None:
    val = get_peak_flops("Some Unknown GPU XYZ")
    assert np.isinf(val)


def test_hardware_profile_not_applied_when_torch_not_intended() -> None:
    cfg = GeneratorConfig()
    cfg.runtime.prefer_torch = False
    cfg.runtime.device = "auto"
    original_train = cfg.dataset.n_train
    original_profile = cfg.benchmark.profile_name

    hw = HardwareInfo(
        backend="cuda",
        requested_device="auto",
        device_name="NVIDIA H100 SXM",
        total_memory_gb=80.0,
        peak_flops=989e12,
        profile="cuda_h100",
    )
    apply_hardware_profile(cfg, hw)

    assert cfg.dataset.n_train == original_train
    assert cfg.benchmark.profile_name == original_profile


def test_hardware_profile_applied_when_cuda_explicit() -> None:
    cfg = GeneratorConfig()
    cfg.runtime.prefer_torch = False
    cfg.runtime.device = "cuda"

    hw = HardwareInfo(
        backend="cuda",
        requested_device="cuda",
        device_name="NVIDIA H100 SXM",
        total_memory_gb=80.0,
        peak_flops=989e12,
        profile="cuda_h100",
    )
    apply_hardware_profile(cfg, hw)

    assert cfg.dataset.n_train >= 4096
    assert cfg.benchmark.profile_name == "cuda_h100_auto"
