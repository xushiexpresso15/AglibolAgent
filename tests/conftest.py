"""Test fixtures and mock helpers for Aglibol Agent."""

from __future__ import annotations

import pytest

from aglibol.core.types import (
    GPUInfo,
    HardwareProfile,
    HardwareTier,
)
from aglibol.ollama.client import OllamaClient


class MockOllamaClient(OllamaClient):
    """Mock OllamaClient for deterministic testing without running live Ollama server."""

    def __init__(self) -> None:
        super().__init__(host="http://mock-ollama:11434")
        self.mock_chat_responses: list[dict] = []
        self.unloaded_models: list[str] = []
        self.currently_loaded: list = []

    async def health_check(self) -> bool:
        return True

    async def unload_model(self, model_name: str) -> bool:
        self.unloaded_models.append(model_name)
        self.currently_loaded = [m for m in self.currently_loaded if m.model != model_name]
        return True

    async def ps(self) -> list:
        return self.currently_loaded

    async def chat(
        self, model, messages, options=None, keep_alive=None, tools=None, format=None, **kwargs
    ):
        if self.mock_chat_responses:
            return self.mock_chat_responses.pop(0)
        return {
            "content": "Mock response",
            "tool_calls": [],
            "done": True,
            "eval_count": 50,
        }


@pytest.fixture
def mock_tier1_profile() -> HardwareProfile:
    """Fixture providing a mock 4 GB VRAM profile."""
    gpu = GPUInfo(
        name="Mock RTX 3050 Laptop",
        vendor="NVIDIA",
        vram_total_mb=4096.0,
        vram_free_mb=3500.0,
        vram_used_mb=596.0,
    )
    return HardwareProfile(
        os_name="Windows",
        os_version="11",
        cpu_name="Mock i5",
        cpu_cores_physical=6,
        cpu_cores_logical=12,
        ram_total_gb=16.0,
        ram_available_gb=5.0,
        ram_used_percent=68.0,
        gpus=[gpu],
        primary_gpu=gpu,
        has_discrete_gpu=True,
        total_vram_gb=4.0,
        total_vram_free_gb=3.42,
        gpu_count=1,
        tier=HardwareTier.TIER_1_4GB,
    )


@pytest.fixture
def mock_tier6_profile() -> HardwareProfile:
    """Fixture providing a mock 24 GB VRAM workstation profile."""
    gpu = GPUInfo(
        name="Mock RTX 4090",
        vendor="NVIDIA",
        vram_total_mb=24576.0,
        vram_free_mb=22000.0,
        vram_used_mb=2576.0,
    )
    return HardwareProfile(
        os_name="Linux",
        os_version="6.8",
        cpu_name="Mock Ryzen 9",
        cpu_cores_physical=16,
        cpu_cores_logical=32,
        ram_total_gb=64.0,
        ram_available_gb=48.0,
        ram_used_percent=25.0,
        gpus=[gpu],
        primary_gpu=gpu,
        has_discrete_gpu=True,
        total_vram_gb=24.0,
        total_vram_free_gb=21.48,
        gpu_count=1,
        tier=HardwareTier.TIER_6_24GB,
    )


@pytest.fixture(autouse=True)
def isolate_user_config(tmp_path, monkeypatch):
    """Isolate user ~/.aglibol config during all test runs."""
    from pathlib import Path

    fake_home = tmp_path / "fake_home"
    fake_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setattr(Path, "home", lambda: fake_home)
