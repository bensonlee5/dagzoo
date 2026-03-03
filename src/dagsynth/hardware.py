"""Hardware detection utilities for runtime backend capability."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class HardwareInfo:
    backend: str
    requested_device: str
    device_name: str
    total_memory_gb: float | None
    peak_flops: float
    tier: str


def get_peak_flops(device_name: str) -> float:
    """Return peak FLOPs estimate for known accelerator model names."""

    name = device_name.lower()
    peak_flops_table: tuple[tuple[list[str], float], ...] = (
        (["gb200"], 2.5e15),
        (["grace blackwell"], 2.5e15),
        (["b200"], 2.25e15),
        (["b100"], 1.8e15),
        (["h200", "nvl"], 836e12),
        (["h200", "pcie"], 836e12),
        (["h200"], 989e12),
        (["h100", "nvl"], 835e12),
        (["h100", "pcie"], 756e12),
        (["h100"], 989e12),
        (["h800", "nvl"], 989e12),
        (["h800"], 756e12),
        (["a100"], 312e12),
        (["a800"], 312e12),
        (["a40"], 149.7e12),
        (["a30"], 165e12),
        (["l40s"], 362e12),
        (["l40-s"], 362e12),
        (["l40 s"], 362e12),
        (["l4"], 121e12),
        (["mi355"], 2.5e15),
        (["mi325"], 1.3074e15),
        (["mi300x"], 1.3074e15),
        (["mi300a"], 980.6e12),
        (["mi250x"], 383e12),
        (["mi250"], 362.1e12),
        (["5090"], 209.5e12),
        (["4090"], 165.2e12),
        (["3090"], 71e12),
    )
    for patterns, flops in peak_flops_table:
        if all(p in name for p in patterns):
            return flops

    if "data center gpu max 1550" in name and hasattr(torch, "xpu"):
        try:
            max_comp_units = torch.xpu.get_device_properties("xpu").max_compute_units
            return float(512 * max_comp_units * 1300 * 10**6)
        except Exception:
            pass

    logger.warning("Peak flops undefined for %s; using inf.", device_name)
    return float("inf")


def _recommend_tier(device_name: str, backend: str) -> str:
    """Map detected hardware to a coarse runtime tier."""

    if backend != "cuda":
        return "cpu"
    peak = get_peak_flops(device_name)
    if not math.isfinite(peak):
        return "cuda_unknown_fallback"
    if peak >= 700e12:
        return "cuda_h100"
    if peak >= 300e12:
        return "cuda_datacenter"
    return "cuda_desktop"


def detect_hardware(requested_device: str | None = None) -> HardwareInfo:
    """Detect active accelerator backend/device metadata."""

    requested = (requested_device or "auto").lower()
    if requested in ("cuda", "auto") and torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        name = str(props.name)
        mem_gb = float(props.total_memory) / (1024.0**3)
        peak = get_peak_flops(name)
        return HardwareInfo(
            backend="cuda",
            requested_device=requested,
            device_name=name,
            total_memory_gb=mem_gb,
            peak_flops=peak,
            tier=_recommend_tier(name, "cuda"),
        )

    if (
        requested in ("mps", "auto")
        and getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
    ):
        return HardwareInfo(
            backend="mps",
            requested_device=requested,
            device_name="apple_mps",
            total_memory_gb=None,
            peak_flops=float("inf"),
            tier="cpu",
        )

    if requested in ("cpu", "auto"):
        return HardwareInfo(
            backend="cpu",
            requested_device=requested,
            device_name="cpu",
            total_memory_gb=None,
            peak_flops=float("inf"),
            tier="cpu",
        )

    return HardwareInfo(
        backend="cpu",
        requested_device=requested,
        device_name="cpu",
        total_memory_gb=None,
        peak_flops=float("inf"),
        tier="cpu",
    )
