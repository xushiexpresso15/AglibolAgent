"""Ollama API integration package."""

from aglibol.ollama.client import OllamaClient
from aglibol.ollama.models import ModelDetail, ModelInfo

__all__ = ["OllamaClient", "ModelInfo", "ModelDetail"]
