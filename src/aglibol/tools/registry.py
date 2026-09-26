"""Tool registry for registering and discovering agent tools."""

from __future__ import annotations

from typing import Any

from aglibol.core.types import ToolResult
from aglibol.tools.base import BaseTool


class ToolRegistry:
    """Registry holding instantiated tools available to agents."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a new tool instance."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        """Lookup a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[BaseTool]:
        """Return all registered tools."""
        return list(self._tools.values())

    def get_ollama_tools(self, tool_names: list[str] | None = None) -> list[dict[str, Any]]:
        """Get Ollama-compatible function schemas for requested tools (or all if None)."""
        tools_to_include = (
            [self._tools[n] for n in tool_names if n in self._tools]
            if tool_names is not None
            else self._tools.values()
        )
        return [t.to_ollama_tool() for t in tools_to_include]

    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute a tool by name with provided arguments."""
        tool = self.get(name)
        if not tool:
            return ToolResult(
                tool_name=name,
                success=False,
                output="",
                error=f"Tool '{name}' is not registered in ToolRegistry.",
            )
        try:
            return await tool.execute(**arguments)
        except Exception as e:
            return ToolResult(
                tool_name=name,
                success=False,
                output="",
                error=f"Exception during execution of tool '{name}': {str(e)}",
            )
