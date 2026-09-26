"""Tests for ModelScheduler adaptive residency behavior."""

from __future__ import annotations

import pytest
from conftest import MockOllamaClient

from aglibol.core.scheduler import ModelScheduler
from aglibol.core.types import GPUInfo, HardwareProfile, HardwareTier
from aglibol.ollama.models import LoadedModel


@pytest.mark.asyncio
async def test_scheduler_constrained_mode_unloads_previous(mock_tier1_profile):
    """Verify that on low VRAM, scheduler evicts old models before acquiring new one."""
    client = MockOllamaClient()
    # Pre-populate with another loaded model
    client.currently_loaded = [
        LoadedModel(name="llama3.1:8b", model="llama3.1:8b", size_vram=4000000000)
    ]

    scheduler = ModelScheduler(client=client, profile=mock_tier1_profile)
    model, options, keep_alive = await scheduler.acquire_model("qwen2.5-coder:7b", role="coder")

    assert model == "qwen2.5-coder:7b"
    assert keep_alive == "0"
    # Verify that the old model was evicted!
    assert "llama3.1:8b" in client.unloaded_models


@pytest.mark.asyncio
async def test_scheduler_high_capacity_mode_allows_concurrency():
    """Verify that on high VRAM (e.g. 48GB), scheduler keeps existing models resident."""
    client = MockOllamaClient()
    client.currently_loaded = [
        LoadedModel(name="qwen2.5:32b", model="qwen2.5:32b", size_vram=20000000000)
    ]

    # Create 48GB high capacity profile
    gpu = GPUInfo(name="RTX 6000 Ada", vendor="NVIDIA", vram_total_mb=49152.0, vram_free_mb=40000.0)
    profile = HardwareProfile(
        os_name="Linux",
        os_version="6.5",
        cpu_name="Threadripper",
        cpu_cores_physical=32,
        cpu_cores_logical=64,
        ram_total_gb=128.0,
        ram_available_gb=96.0,
        ram_used_percent=25.0,
        gpus=[gpu],
        primary_gpu=gpu,
        has_discrete_gpu=True,
        total_vram_gb=48.0,
        total_vram_free_gb=39.06,
        gpu_count=1,
        tier=HardwareTier.TIER_7_ENTERPRISE,
    )

    scheduler = ModelScheduler(client=client, profile=profile)
    model, options, keep_alive = await scheduler.acquire_model("qwen2.5-coder:32b", role="coder")

    assert model == "qwen2.5-coder:32b"
    assert keep_alive == "-1"
    # On high VRAM, existing resident model should NOT be unloaded because capacity is not exceeded
    assert "qwen2.5:32b" not in client.unloaded_models
