"""Base tool abstract class and schema definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from aglibol.core.types import ToolResult


class BaseTool(ABC):
    """Abstract base class for all tools available to Aglibol Agent."""

    name: str
    description: str
    parameters: dict[str, Any]

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool action and return a structured ToolResult."""
        pass

    def to_ollama_tool(self) -> dict[str, Any]:
        """Convert tool definition to the JSON Schema expected by Ollama / OpenAI tools format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
