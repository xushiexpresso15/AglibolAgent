"""Search and replace tool with fuzzy matching."""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any

from aglibol.core.types import ToolResult
from aglibol.tools.base import BaseTool
from aglibol.tools.registry import ToolRegistry


class SearchReplaceTool(BaseTool):
    name = "search_replace"
    description = "Apply a search/replace edit to an existing file. Use this instead of write_file when editing files larger than 60 lines."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to edit.",
            },
            "search": {
                "type": "string",
                "description": "Exact text to find in the file.",
            },
            "replace": {
                "type": "string",
                "description": "Replacement text.",
            },
        },
        "required": ["path", "search", "replace"],
    }

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

    def _normalize_whitespace(self, text: str) -> str:
        return "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").split("\n"))

    def _relax_indentation(self, text: str) -> str:
        return "\n".join(line.strip() for line in text.replace("\r\n", "\n").split("\n"))

    def _find_normalized_match(
        self,
        content_lines: list[str],
        search_lines: list[str],
        strip_mode: str = "rstrip",
    ) -> tuple[int | None, int | None]:
        """Find the start and end line indices in content_lines where search_lines match
        after applying the specified normalization. Returns (start_idx, end_idx) where
        end_idx is exclusive, or (None, None) if no match."""
        search_len = len(search_lines)
        if search_len == 0 or search_len > len(content_lines):
            return None, None

        strip_fn = str.strip if strip_mode == "strip" else str.rstrip
        normalized_search = [strip_fn(line) for line in search_lines]

        for i in range(len(content_lines) - search_len + 1):
            window = [strip_fn(content_lines[i + j]) for j in range(search_len)]
            if window == normalized_search:
                return i, i + search_len
        return None, None

    async def execute(self, path: str, search: str, replace: str, **kwargs: Any) -> ToolResult:
        p = self._validate_path(path)
        if isinstance(p, ToolResult):
            return p

        if not p.exists():
            return ToolResult(
                tool_name=self.name, success=False, output="", error=f"File not found: {path}"
            )

        try:
            with open(p, encoding="utf-8") as f:
                content = f.read()

            # 1. Exact match
            if search in content:
                new_content = content.replace(search, replace, 1)
                with open(p, "w", encoding="utf-8") as f:
                    f.write(new_content)
                return ToolResult(
                    tool_name=self.name, success=True, output="Replaced using Exact match"
                )

            # 2. Whitespace-normalized match (preserves original formatting outside match)
            norm_search = self._normalize_whitespace(search)
            content_lines = content.split("\n")
            search_norm_lines = norm_search.split("\n")
            match_start, match_end = self._find_normalized_match(
                content_lines, search_norm_lines, strip_mode="rstrip"
            )
            if match_start is not None:
                before = "\n".join(content_lines[:match_start])
                after = "\n".join(content_lines[match_end:])
                parts = [p for p in [before, replace, after] if p]
                new_content = "\n".join(parts)
                if content.endswith("\n") and not new_content.endswith("\n"):
                    new_content += "\n"
                with open(p, "w", encoding="utf-8") as f:
                    f.write(new_content)
                return ToolResult(
                    tool_name=self.name,
                    success=True,
                    output="Replaced using Whitespace-normalized match",
                )

            # 3. Indentation-relaxed match (preserves original formatting outside match)
            relax_search = self._relax_indentation(search)
            relax_search_lines = relax_search.split("\n")
            match_start, match_end = self._find_normalized_match(
                content_lines, relax_search_lines, strip_mode="strip"
            )
            if match_start is not None:
                before = "\n".join(content_lines[:match_start])
                after = "\n".join(content_lines[match_end:])
                parts = [p for p in [before, replace, after] if p]
                new_content = "\n".join(parts)
                if content.endswith("\n") and not new_content.endswith("\n"):
                    new_content += "\n"
                with open(p, "w", encoding="utf-8") as f:
                    f.write(new_content)
                return ToolResult(
                    tool_name=self.name,
                    success=True,
                    output="Replaced using Indentation-relaxed match",
                )

            # 4. Fuzzy match
            search_lines = search.splitlines()
            content_lines = content.splitlines()
            search_len = len(search_lines)

            best_ratio = 0.0
            best_idx = -1

            if search_len <= len(content_lines) and search_len > 0:
                for i in range(len(content_lines) - search_len + 1):
                    window = "\n".join(content_lines[i : i + search_len])
                    ratio = difflib.SequenceMatcher(None, search, window).ratio()
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_idx = i

            if best_ratio >= 0.85 and best_idx != -1:
                new_lines = (
                    content_lines[:best_idx]
                    + replace.splitlines()
                    + content_lines[best_idx + search_len :]
                )
                new_content = "\n".join(new_lines)
                if replace.endswith("\n") and not new_content.endswith("\n"):
                    new_content += "\n"
                elif content.endswith("\n") and not new_content.endswith("\n"):
                    new_content += "\n"
                with open(p, "w", encoding="utf-8") as f:
                    f.write(new_content)
                return ToolResult(
                    tool_name=self.name,
                    success=True,
                    output=f"Replaced using Fuzzy match (ratio: {best_ratio:.2f})",
                )

            return ToolResult(
                tool_name=self.name,
                success=False,
                output="",
                error=f"Match failed. Best fuzzy match ratio was {best_ratio:.2f}",
            )

        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, output="", error=str(e))


def register_search_replace_tools(
    registry: ToolRegistry, workspace_root: Path | None = None
) -> None:
    """Register search and replace tools with the registry."""
    registry.register(SearchReplaceTool(workspace_root=workspace_root))
