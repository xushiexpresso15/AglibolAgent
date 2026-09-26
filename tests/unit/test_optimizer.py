"""Comprehensive tests for continuous mathematical resource optimizer."""

from __future__ import annotations

from aglibol.core.optimizer import ResourceOptimizer
from aglibol.core.types import GPUInfo, HardwareProfile, HardwareTier


def test_weight_and_kv_cache_formulas():
    # 7B Q4 model weight estimation
    weight_7b = ResourceOptimizer.estimate_weight_size_gb(7.0, quant_bits=4.5)
    assert 4.0 <= weight_7b <= 5.0

    # 14B Q4 model weight estimation
    weight_14b = ResourceOptimizer.estimate_weight_size_gb(14.0, quant_bits=4.5)
    assert 8.0 <= weight_14b <= 10.0

    # 70B Q4 model weight estimation
    weight_70b = ResourceOptimizer.estimate_weight_size_gb(70.0, quant_bits=4.5)
    assert 40.0 <= weight_70b <= 48.0

    # KV Cache estimation for 7B with 2048 and 8192 tokens
    kv_2k = ResourceOptimizer.estimate_kv_cache_gb(2048, param_b=7.0)
    assert 0.05 <= kv_2k <= 0.25

    kv_8k = ResourceOptimizer.estimate_kv_cache_gb(8192, param_b=7.0)
    assert 0.3 <= kv_8k <= 1.0


def test_cpu_only_mode_low_ram():
    """Test pure CPU machine with limited RAM (e.g. 8GB laptop, no discrete GPU)."""
    profile = HardwareProfile(
        os_name="Windows",
        os_version="11",
        cpu_name="Intel Core i3",
        cpu_cores_physical=4,
        cpu_cores_logical=8,
        ram_total_gb=8.0,
        ram_available_gb=3.5,
        ram_used_percent=56.0,
        has_discrete_gpu=False,
        tier=HardwareTier.TIER_0_CPU,
    )

    params = ResourceOptimizer.get_params_for_profile(profile, role="coder")
    assert params.num_gpu == 0
    assert params.target_backend == "cpu"
    assert params.num_ctx == 2048
    assert params.concurrency_allowed is False


def test_cpu_only_mode_high_ram():
    """Test workstation CPU server with 64GB RAM and no dedicated GPU."""
    profile = HardwareProfile(
        os_name="Linux",
        os_version="6.5",
        cpu_name="AMD EPYC",
        cpu_cores_physical=16,
        cpu_cores_logical=32,
        ram_total_gb=64.0,
        ram_available_gb=52.0,
        ram_used_percent=18.0,
        has_discrete_gpu=False,
        tier=HardwareTier.TIER_0_CPU,
    )

    params = ResourceOptimizer.get_params_for_profile(profile, role="coder")
    assert params.num_gpu == 0
    assert params.target_backend == "cpu"
    # 52 GB available RAM allows 14B models on CPU!
    assert "14b" in params.recommended_model
    assert params.num_ctx >= 4096


def test_rtx_3050_4gb(mock_tier1_profile):
    """Test 4GB VRAM constrained setup (RTX 3050 Laptop)."""
    params = ResourceOptimizer.get_params_for_profile(mock_tier1_profile, role="coder")
    assert params.concurrency_allowed is False
    assert params.keep_alive == "0"
    assert params.num_ctx == 2048
    assert "7b" in params.recommended_model
    assert params.target_backend == "cuda"


def test_rtx_3080_10gb():
    """Test non-standard 10GB VRAM GPU."""
    gpu = GPUInfo(name="RTX 3080 10GB", vendor="NVIDIA", vram_total_mb=10240.0, vram_free_mb=9000.0)
    profile = HardwareProfile(
        os_name="Windows",
        os_version="11",
        cpu_name="i7-12700K",
        cpu_cores_physical=8,
        cpu_cores_logical=16,
        ram_total_gb=32.0,
        ram_available_gb=24.0,
        ram_used_percent=25.0,
        gpus=[gpu],
        primary_gpu=gpu,
        has_discrete_gpu=True,
        total_vram_gb=10.0,
        total_vram_free_gb=8.79,
        tier=HardwareTier.TIER_4_12GB,
    )

    params = ResourceOptimizer.get_params_for_profile(profile, role="coder")
    assert params.num_gpu == -1  # 100% layers fit into 10GB for 7B model
    assert params.num_ctx >= 4096


def test_rx_7900xt_20gb():
    """Test non-standard 20GB VRAM AMD GPU."""
    gpu = GPUInfo(
        name="AMD Radeon RX 7900 XT", vendor="AMD", vram_total_mb=20480.0, vram_free_mb=19000.0
    )
    profile = HardwareProfile(
        os_name="Linux",
        os_version="6.8",
        cpu_name="Ryzen 7 7800X3D",
        cpu_cores_physical=8,
        cpu_cores_logical=16,
        ram_total_gb=32.0,
        ram_available_gb=26.0,
        ram_used_percent=18.0,
        gpus=[gpu],
        primary_gpu=gpu,
        has_discrete_gpu=True,
        total_vram_gb=20.0,
        total_vram_free_gb=18.55,
        tier=HardwareTier.TIER_5_16GB,
    )

    params = ResourceOptimizer.get_params_for_profile(profile, role="coder")
    assert params.target_backend == "rocm"
    assert "14b" in params.recommended_model
    assert params.concurrency_allowed is True
    assert params.num_ctx >= 8192


def test_enterprise_80gb():
    """Test 80GB VRAM Enterprise Server (NVIDIA A100 / H100)."""
    gpu = GPUInfo(
        name="NVIDIA A100-SXM4-80GB", vendor="NVIDIA", vram_total_mb=81920.0, vram_free_mb=78000.0
    )
    profile = HardwareProfile(
        os_name="Linux",
        os_version="6.5",
        cpu_name="Intel Xeon Platinum",
        cpu_cores_physical=64,
        cpu_cores_logical=128,
        ram_total_gb=256.0,
        ram_available_gb=220.0,
        ram_used_percent=14.0,
        gpus=[gpu],
        primary_gpu=gpu,
        has_discrete_gpu=True,
        total_vram_gb=80.0,
        total_vram_free_gb=76.17,
        tier=HardwareTier.TIER_7_ENTERPRISE,
    )

    params = ResourceOptimizer.get_params_for_profile(profile, role="planner")
    assert "72b" in params.recommended_model
    assert params.num_ctx >= 32768
    assert params.concurrency_allowed is True
    assert params.max_loaded_models >= 3
    assert params.keep_alive == "-1"
