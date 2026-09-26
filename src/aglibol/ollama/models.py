"""Ollama model representations and parameter size estimation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    """Information regarding an installed local Ollama model."""

    name: str
    size_bytes: int = 0
    digest: str = ""
    modified_at: str = ""
    parameter_size: str = ""  # e.g., "7B", "14B"
    quantization_level: str = ""  # e.g., "Q4_K_M", "Q8_0"

    @property
    def size_gb(self) -> float:
        return round(self.size_bytes / (1024**3), 2)


class LoadedModel(BaseModel):
    """Information about a model currently resident in RAM/VRAM (/api/ps)."""

    name: str
    model: str
    size: int = 0
    digest: str = ""
    expires_at: str = ""
    size_vram: int = 0

    @property
    def vram_gb(self) -> float:
        return round(self.size_vram / (1024**3), 2)


class ModelDetail(BaseModel):
    """Detailed model configuration returned by /api/show."""

    license: str = ""
    modelfile: str = ""
    parameters: str = ""
    template: str = ""
    system: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
