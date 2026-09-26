"""Tests for tool registry and built-in tools."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from aglibol.tools.file_ops import ListDirTool, ReadFileTool, WriteFileTool, register_file_tools
from aglibol.tools.registry import ToolRegistry
from aglibol.tools.shell import ShellCommandTool


@pytest.mark.asyncio
async def test_file_tools():
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = WriteFileTool(workspace_root=Path(tmpdir))
        w_res = await writer.execute(path="sample.txt", content="Aglibol Agent tool test content")
        assert w_res.success is True

        reader = ReadFileTool(workspace_root=Path(tmpdir))
        r_res = await reader.execute(path="sample.txt")
        assert r_res.success is True
        assert r_res.output == "Aglibol Agent tool test content"

        lister = ListDirTool(workspace_root=Path(tmpdir))
        l_res = await lister.execute(path=".")
        assert l_res.success is True
        assert "sample.txt" in l_res.output


@pytest.mark.asyncio
async def test_shell_tool():
    shell = ShellCommandTool()
    res = await shell.execute(command="python -c \"print('Hello from subprocess')\"")
    assert res.success is True
    assert "Hello from subprocess" in res.output


@pytest.mark.asyncio
async def test_tool_registry():
    reg = ToolRegistry()
    register_file_tools(reg)

    assert reg.get("read_file") is not None
    assert reg.get("write_file") is not None
    assert reg.get("list_dir") is not None

    schemas = reg.get_ollama_tools(["read_file"])
    assert len(schemas) == 1
    assert schemas[0]["function"]["name"] == "read_file"


def test_shell_command_windows_normalization():
    # Verify common POSIX commands are translated for Windows cmd.exe
    assert (
        ShellCommandTool.normalize_windows_command('ls -la "C:\\Users\\workspace"')
        == 'dir "C:\\Users\\workspace"'
    )
    assert ShellCommandTool.normalize_windows_command("ls -la") == "dir"
    assert ShellCommandTool.normalize_windows_command("ls -l") == "dir"
    assert ShellCommandTool.normalize_windows_command("ls") == "dir"
    assert ShellCommandTool.normalize_windows_command("cat solution.py") == "type solution.py"
    assert ShellCommandTool.normalize_windows_command("pwd") == "cd"
    assert ShellCommandTool.normalize_windows_command("clear") == "cls"
    assert ShellCommandTool.normalize_windows_command("which python") == "where python"
    assert ShellCommandTool.normalize_windows_command("touch notes.txt") == "type nul > notes.txt"
    # Unrelated commands should remain untouched
    assert ShellCommandTool.normalize_windows_command("python test.py") == "python test.py"
