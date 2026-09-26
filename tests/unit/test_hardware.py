"""Tests for hardware profiler supporting multi-GPU, Apple Silicon, and CPU-only setups."""

from __future__ import annotations

from aglibol.core.hardware import HardwareProfiler
from aglibol.core.types import GPUInfo, HardwareProfile, HardwareTier


def test_hardware_profiler_detect():
    profile = HardwareProfiler.detect()
    assert profile.os_name in ("Windows", "Linux", "Darwin")
    assert profile.cpu_cores_physical >= 1
    assert profile.ram_total_gb > 0
    assert profile.tier in list(HardwareTier)


def test_tier_derivation_across_full_spectrum():
    # 0GB VRAM (CPU only)
    assert HardwareTier.from_vram_gb(0.0, has_discrete_gpu=False) == HardwareTier.TIER_0_CPU
    assert HardwareTier.from_vram_gb(0.5, has_discrete_gpu=True) == HardwareTier.TIER_0_CPU

    # 4GB - 6GB
    assert HardwareTier.from_vram_gb(4.0, has_discrete_gpu=True) == HardwareTier.TIER_1_4GB
    assert HardwareTier.from_vram_gb(6.0, has_discrete_gpu=True) == HardwareTier.TIER_2_6GB

    # 8GB - 12GB
    assert HardwareTier.from_vram_gb(8.0, has_discrete_gpu=True) == HardwareTier.TIER_3_8GB
    assert HardwareTier.from_vram_gb(10.0, has_discrete_gpu=True) == HardwareTier.TIER_4_12GB
    assert HardwareTier.from_vram_gb(12.0, has_discrete_gpu=True) == HardwareTier.TIER_4_12GB

    # 16GB - 20GB
    assert HardwareTier.from_vram_gb(16.0, has_discrete_gpu=True) == HardwareTier.TIER_5_16GB
    assert HardwareTier.from_vram_gb(20.0, has_discrete_gpu=True) == HardwareTier.TIER_5_16GB

    # 24GB - 32GB
    assert HardwareTier.from_vram_gb(24.0, has_discrete_gpu=True) == HardwareTier.TIER_6_24GB
    assert HardwareTier.from_vram_gb(32.0, has_discrete_gpu=True) == HardwareTier.TIER_6_24GB

    # 48GB - 80GB (Enterprise / Multi-GPU)
    assert HardwareTier.from_vram_gb(48.0, has_discrete_gpu=True) == HardwareTier.TIER_7_ENTERPRISE
    assert HardwareTier.from_vram_gb(80.0, has_discrete_gpu=True) == HardwareTier.TIER_7_ENTERPRISE


def test_multi_gpu_aggregation():
    gpu1 = GPUInfo(name="RTX 3090 #1", vendor="NVIDIA", vram_total_mb=24576.0, vram_free_mb=23000.0)
    gpu2 = GPUInfo(name="RTX 3090 #2", vendor="NVIDIA", vram_total_mb=24576.0, vram_free_mb=23000.0)

    profile = HardwareProfile(
        os_name="Linux",
        os_version="6.8",
        cpu_name="Threadripper",
        cpu_cores_physical=32,
        cpu_cores_logical=64,
        ram_total_gb=128.0,
        ram_available_gb=110.0,
        ram_used_percent=14.0,
        gpus=[gpu1, gpu2],
        primary_gpu=gpu1,
        tier=HardwareTier.from_vram_gb(48.0, has_discrete_gpu=True),
        has_discrete_gpu=True,
        total_vram_gb=48.0,
        total_vram_free_gb=44.92,
        gpu_count=2,
    )

    assert profile.tier == HardwareTier.TIER_7_ENTERPRISE
    assert profile.gpu_count == 2
    assert profile.total_vram_gb == 48.0
