"""File operations tools: read, write, and list directories."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aglibol.core.types import ToolResult
from aglibol.tools.base import BaseTool
from aglibol.tools.registry import ToolRegistry


class WorkspaceFileTool(BaseTool):
    def __init__(self, workspace_root: Path | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Security: always enforce sandbox — default to cwd if not specified
        self.workspace_root = workspace_root or Path.cwd()

    def _validate_path(self, path: str) -> Path | ToolResult:
        p = Path(path)
        if not p.is_absolute():
            p = self.workspace_root / p
        p = p.resolve()
        root = self.workspace_root.resolve()
        # Symlink-aware check: resolve both to prevent symlink escape
        try:
            p.relative_to(root)
        except ValueError:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="",
                error=f"Access denied: path '{path}' is outside workspace '{self.workspace_root}'",
            )
        return p


class ReadFileTool(WorkspaceFileTool):
    name = "read_file"
    description = "Read the contents of a local file as plain text."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to read (relative or absolute).",
            },
            "max_lines": {
                "type": "integer",
                "description": "Optional limit on lines to read to prevent VRAM context overflow.",
                "default": 500,
            },
        },
        "required": ["path"],
    }

    async def execute(self, path: str, max_lines: int = 500, **kwargs: Any) -> ToolResult:
        p = self._validate_path(path)
        if isinstance(p, ToolResult):
            return p

        if not p.exists():
            return ToolResult(
                tool_name=self.name, success=False, output="", error=f"File not found: {path}"
            )
        if not p.is_file():
            return ToolResult(
                tool_name=self.name, success=False, output="", error=f"Path is not a file: {path}"
            )

        try:
            with open(p, encoding="utf-8", errors="replace") as f:
                lines = [f.readline() for _ in range(max_lines)]
                content = "".join(lines)
                if f.readline():
                    content += f"\n... [Truncated: exceeded {max_lines} lines]"
            return ToolResult(tool_name=self.name, success=True, output=content)
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, output="", error=str(e))


class WriteFileTool(WorkspaceFileTool):
    name = "write_file"
    description = "Write or overwrite text content to a local file."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the destination file.",
            },
            "content": {
                "type": "string",
                "description": "Text content to write.",
            },
        },
        "required": ["path", "content"],
    }

    async def execute(self, path: str, content: str, **kwargs: Any) -> ToolResult:
        p = self._validate_path(path)
        if isinstance(p, ToolResult):
            return p

        try:
            # Strip trailing tool call leakage if model appended next tool call into parameter
            import re

            if re.search(r"<(?:function|tool_call|parameter)=", content):
                content = re.split(r"<(?:function|tool_call|parameter)=", content)[0].rstrip()
            if "\n" not in content and "\\n" in content:
                content = content.replace("\\n", "\n").replace("\\t", "\t")
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(content)
            return ToolResult(
                tool_name=self.name,
                success=True,
                output=f"Successfully wrote {len(content)} characters to {path}",
            )
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, output="", error=str(e))


class ListDirTool(WorkspaceFileTool):
    name = "list_dir"
    description = "List files and subdirectories in a target folder."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path to list. Defaults to current directory ('.').",
                "default": ".",
            }
        },
    }

    async def execute(self, path: str = ".", **kwargs: Any) -> ToolResult:
        p = self._validate_path(path)
        if isinstance(p, ToolResult):
            return p

        if not p.exists() or not p.is_dir():
            return ToolResult(
                tool_name=self.name, success=False, output="", error=f"Invalid directory: {path}"
            )

        try:
            entries = []
            for item in p.iterdir():
                kind = "DIR" if item.is_dir() else "FILE"
                entries.append(f"[{kind}] {item.name}")
            return ToolResult(
                tool_name=self.name, success=True, output="\n".join(entries) or "(Empty directory)"
            )
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, output="", error=str(e))


def register_file_tools(registry: ToolRegistry, workspace_root: Path | None = None) -> None:
    """Register all file manipulation tools with the registry."""
    registry.register(ReadFileTool(workspace_root=workspace_root))
    registry.register(WriteFileTool(workspace_root=workspace_root))
    registry.register(ListDirTool(workspace_root=workspace_root))
