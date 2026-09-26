"""Shell execution tool for sandboxed or local command running."""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from aglibol.core.types import ToolResult
from aglibol.tools.base import BaseTool
from aglibol.tools.registry import ToolRegistry


class ShellCommandTool(BaseTool):
    name = "execute_shell"
    description = "Execute a shell command with a strict timeout and capture its stdout/stderr."
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to run (e.g., 'pytest tests/', 'python my_script.py').",
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Maximum execution time before timeout.",
                "default": 30,
            },
        },
        "required": ["command"],
    }

    BLOCKED_PATTERNS = [
        r"\brm\s+(-rf?|--recursive)\s+[/\\]",
        r"\brmdir\s+/s",
        r"\bformat\s+[a-zA-Z]:",
        r"\bdel\s+/[fs].*[/\\]",
        r"\bmkfs\.",
        r"\bdd\s+.*of\s*=\s*/dev/",
        r":.*\|.*:",
        r"fork\s*bomb",
    ]

    def __init__(self, workspace_root: Path | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Security: always enforce sandbox — default to cwd if not specified
        self.workspace_root = workspace_root or Path.cwd()

    def _check_blocked(self, command: str) -> str | None:
        for pattern in self.BLOCKED_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return f"Command rejected: matches blocked pattern '{pattern}'"
        return None

    @staticmethod
    def normalize_windows_command(command: str) -> str:
        """Translate common POSIX shell commands to Windows cmd equivalents."""
        cmd = command.strip()
        if re.match(r"^ls\b", cmd):
            # Translate ls [-flags] [path] to dir [path]
            translated = re.sub(r"^ls(?:\s+-[a-zA-Z]+)*", "dir", cmd)
            return translated if translated.strip() else "dir"
        if re.match(r"^cat\s+", cmd):
            return re.sub(r"^cat\s+", "type ", cmd)
        if cmd == "pwd":
            return "cd"
        if cmd == "clear":
            return "cls"
        if re.match(r"^which\s+", cmd):
            return re.sub(r"^which\s+", "where ", cmd)
        if re.match(r"^touch\s+", cmd):
            return re.sub(r"^touch\s+(.+)$", r"type nul > \1", cmd)
        return cmd

    async def execute(self, command: str, timeout_seconds: int = 30, **kwargs: Any) -> ToolResult:
        if sys.platform == "win32":
            command = self.normalize_windows_command(command)

        blocked_err = self._check_blocked(command)
        if blocked_err:
            return ToolResult(tool_name=self.name, success=False, output="", error=blocked_err)

        try:
            sensitive_keys = {
                "AWS_SECRET_ACCESS_KEY",
                "AWS_ACCESS_KEY_ID",
                "GITHUB_TOKEN",
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
                "HF_TOKEN",
                "HUGGING_FACE_HUB_TOKEN",
            }
            clean_env = {k: v for k, v in os.environ.items() if k.upper() not in sensitive_keys}

            kwargs_shell = {
                "stdout": asyncio.subprocess.PIPE,
                "stderr": asyncio.subprocess.PIPE,
                "env": clean_env,
                "cwd": str(self.workspace_root),
            }

            proc = await asyncio.create_subprocess_shell(command, **kwargs_shell)
            try:
                MAX_OUTPUT_BYTES = 1_048_576
                stdout_chunks = []
                stderr_chunks = []
                total_read = 0
                limit_reached = False

                async def read_stream(stream, chunks_list):
                    nonlocal total_read, limit_reached
                    while not limit_reached:
                        chunk = await stream.read(8192)
                        if not chunk:
                            break
                        chunks_list.append(chunk)
                        total_read += len(chunk)
                        if total_read >= MAX_OUTPUT_BYTES:
                            limit_reached = True
                            break

                await asyncio.wait_for(
                    asyncio.gather(
                        read_stream(proc.stdout, stdout_chunks),
                        read_stream(proc.stderr, stderr_chunks),
                    ),
                    timeout=float(timeout_seconds),
                )

                if limit_reached:
                    try:
                        if sys.platform == "win32":
                            subprocess.run(
                                ["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True
                            )
                        else:
                            proc.kill()
                    except Exception:
                        pass
                    stdout_chunks.append(b"\n[Output truncated at 1MB]")

                await proc.wait()
                stdout_bytes = b"".join(stdout_chunks)
                stderr_bytes = b"".join(stderr_chunks)

            except TimeoutError:
                try:
                    if sys.platform == "win32":
                        subprocess.run(
                            ["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True
                        )
                    else:
                        proc.kill()
                except Exception:
                    pass
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    output="",
                    error=f"Command timed out after {timeout_seconds} seconds: {command}",
                )

            stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
            stderr = stderr_bytes.decode("utf-8", errors="replace").strip()
            combined = f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}" if stderr else stdout

            return ToolResult(
                tool_name=self.name,
                success=(proc.returncode == 0),
                output=combined or "(Command executed with no output)",
                error=None
                if proc.returncode == 0
                else f"Process exited with code {proc.returncode}",
            )
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, output="", error=str(e))


def register_shell_tools(registry: ToolRegistry, workspace_root: Path | None = None) -> None:
    """Register shell execution tools."""
    registry.register(ShellCommandTool(workspace_root=workspace_root))
