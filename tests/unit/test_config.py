"""Tests for config manager."""

from __future__ import annotations

from aglibol.core.config import ConfigManager
from aglibol.core.types import HardwareTier


def test_load_default_config():
    cfg = ConfigManager.load()
    assert cfg.version == "0.1.0"
    assert "localhost" in cfg.ollama.host
    assert cfg.models.coder == "qwen2.5-coder:7b"


def test_load_tier_profile():
    tier1_prof = ConfigManager.load_tier_profile(HardwareTier.TIER_1_4GB)
    assert "optimization" in tier1_prof
    assert tier1_prof["optimization"]["num_ctx"] == 2048
