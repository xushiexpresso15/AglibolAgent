"""Unit tests for Phase C deep optimizations (C1 - C10)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from aglibol.agents.coder import CoderAgent
from aglibol.agents.planner import PlannerAgent
from aglibol.agents.reviewer import ReviewerAgent
from aglibol.core.context import ContextManager
from aglibol.core.scheduler import ModelScheduler
from aglibol.core.types import (
    AgentState,
    ChatMessage,
    GPUInfo,
    HardwareTier,
)
from aglibol.core.workflow import WorkflowEngine
from aglibol.ollama.client import OllamaClient
from aglibol.tools.file_ops import WriteFileTool
from aglibol.tools.registry import ToolRegistry
from aglibol.tools.search_replace import SearchReplaceTool

# --- C1: Sandbox Default CWD Enforcement ---


@pytest.mark.asyncio
async def test_sandbox_default_cwd_enforcement():
    # When initialized without workspace_root, it must default to Path.cwd()
    writer = WriteFileTool()
    assert writer.workspace_root == Path.cwd()

    # Attempting to write outside cwd (e.g. to a root or temp outside cwd) must be rejected
    outside_path = (Path.cwd().parent / "arbitrary_forbidden_outside_path.txt").resolve()
    res = await writer.execute(path=str(outside_path), content="malicious")
    assert res.success is False
    assert "Access denied" in (res.error or "")


# --- C2: SearchReplace Preserves Indentation and Surrounding Lines ---


@pytest.mark.asyncio
async def test_search_replace_preserves_indentation_and_surrounding_formatting():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "code.py"
        original_code = (
            "def calculate(x, y):\n"
            "    # Leading comment with indentation\n"
            "    val = x * 2\n"
            "    res = val + y\n"
            "    return res\n"
        )
        test_file.write_text(original_code, encoding="utf-8")

        # Coder wants to edit 'res = val + y' with relaxed indentation
        tool = SearchReplaceTool(workspace_root=Path(tmpdir))
        result = await tool.execute(
            path=str(test_file),
            search="res = val + y   ",  # Trailing whitespace in search
            replace="    res = val + y + 100",
        )
        assert result.success is True

        content = test_file.read_text(encoding="utf-8")
        # Ensure indentation of other lines was NOT stripped!
        assert "    # Leading comment with indentation" in content
        assert "    val = x * 2" in content
        assert "    res = val + y + 100" in content


# --- C4: Planner JSON Heuristic Repair with Single Quotes ---


def test_planner_json_heuristic_repair_single_quotes():
    malformed_json = "{'summary': 'Build CLI', 'tasks': [{'id': 't1', 'title': 'Core', 'description': 'Write code', 'assigned_agent': 'coder', 'dependencies': []}],}"
    repaired = PlannerAgent._heuristic_json_repair(malformed_json)
    data = json.loads(repaired)
    assert data["summary"] == "Build CLI"
    assert len(data["tasks"]) == 1
    assert data["tasks"][0]["id"] == "t1"


# --- C5: Coder Extract All Code Blocks Preserves Subdirectories ---


def test_coder_extract_preserves_subdirectories():
    text = (
        "Here is the module:\n"
        "```python\n"
        "# File: src/components/button.py\n"
        "class Button:\n"
        "    pass\n"
        "```\n\n"
        "Here is the test:\n"
        "```python:tests/ui/test_button.py\n"
        "def test_button():\n"
        "    pass\n"
        "```\n\n"
        "And a malicious traversal:\n"
        "```python:../../evil.py\n"
        "pass\n"
        "```"
    )
    blocks = CoderAgent._extract_all_code_blocks(text)
    # Valid relative subdirectories must be preserved
    assert "src/components/button.py" in blocks
    assert "tests/ui/test_button.py" in blocks
    # Path traversal ../ must be sanitized to basename
    assert "../../evil.py" not in blocks
    assert "evil.py" in blocks


# --- C6: Reviewer Structured JSON Parsing ---


def test_reviewer_structured_json_parsing():
    json_approved = '{"decision": "approved", "reasons": ["Clean syntax", "Complete implementation"], "feedback": "Great work."}'
    assert ReviewerAgent._parse_decision(json_approved) == "approved"

    json_rejected = '{"decision": "rejected", "reasons": ["Missing edge cases"], "feedback": "Handle zero division."}'
    assert ReviewerAgent._parse_decision(json_rejected) == "rejected"

    # Markdown wrapped json
    md_json = '```json\n{"decision": "approved"}\n```'
    assert ReviewerAgent._parse_decision(md_json) == "approved"


# --- C7: Context Masking Head and Tail Preservation ---


def test_context_masking_head_tail_preservation():
    # Create 30 lines (each ~30 chars) -> ~900 chars (~240 tokens)
    lines = [f"line_{i:02d}: sample text data line {i}" for i in range(30)]
    long_content = "\n".join(lines)

    messages = [
        ChatMessage(role="system", content="System instruction for agent"),
        ChatMessage(role="user", content="Goal task description"),
        ChatMessage(role="tool", content=long_content, tool_name="read_file"),
        ChatMessage(role="assistant", content="Working on it..."),
        ChatMessage(role="user", content="Next step"),
        ChatMessage(role="assistant", content="Final response"),
    ]

    # Token budget: max_context_tokens=300, headroom=0.75 -> target 225 tokens.
    # Initial: ~270 tokens > 225 -> triggers observation masking.
    # After masking: 15 lines (~120 tokens) <= 225 -> tool message preserved with masking!
    compressed = ContextManager.compress_messages(
        messages, max_context_tokens=300, headroom_ratio=0.75
    )
    tool_msg = next((m for m in compressed if m.role == "tool"), None)
    assert tool_msg is not None
    assert "masked" in tool_msg.content
    # Head lines preserved
    assert "line_00:" in tool_msg.content
    assert "line_09:" in tool_msg.content
    # Tail lines preserved
    assert "line_29:" in tool_msg.content
    assert "line_28:" in tool_msg.content
    # Middle lines should be masked
    assert "line_15:" not in tool_msg.content


# --- C8: OllamaClient XML Fallback Tool Call Parsing ---


def test_ollama_client_fallback_xml_tool_call_parsing():
    model_output = (
        "I will write the requested code now.\n"
        "<tool_call>\n"
        '{"name": "write_file", "arguments": {"path": "src/app.py", "content": "print(1)"}}\n'
        "</tool_call>\n"
        "Let me also list files.\n"
        "```tool_call\n"
        '{"name": "list_dir", "arguments": {"path": "."}}\n'
        "```"
    )
    calls = OllamaClient._parse_fallback_tool_calls(model_output)
    assert len(calls) == 2
    assert calls[0].tool_name == "write_file"
    assert calls[0].arguments == {"path": "src/app.py", "content": "print(1)"}
    assert calls[1].tool_name == "list_dir"
    assert calls[1].arguments == {"path": "."}


# --- C9: Workflow Self-Healing Escalation ---


@pytest.mark.asyncio
async def test_workflow_self_healing_escalation(mock_tier1_profile):
    from conftest import MockOllamaClient

    client = MockOllamaClient()
    # Responses: Planner valid -> Coder -> Reviewer REJECTED
    client.mock_chat_responses = [
        {
            "content": '{"summary": "P", "tasks": [{"id": "t1", "title": "T", "description": "D", "assigned_agent": "coder"}]}',
            "tool_calls": [],
            "done": True,
        },
        {"content": "```python\ndef foo(): pass\n```", "tool_calls": [], "done": True},
        {
            "content": '{"decision": "rejected", "reasons": ["Fix foo"]}',
            "tool_calls": [],
            "done": True,
        },
        # Self-healing cycle (attempt 1 of 1 max_cycles)
        {"content": "```python\ndef foo(): return 42\n```", "tool_calls": [], "done": True},
        {
            "content": '{"decision": "approved", "reasons": ["All good"]}',
            "tool_calls": [],
            "done": True,
        },
    ]

    scheduler = ModelScheduler(client=client, profile=mock_tier1_profile)
    tool_registry = ToolRegistry()

    engine = WorkflowEngine(
        client=client,
        scheduler=scheduler,
        tool_registry=tool_registry,
        max_total_steps=10,
    )
    engine.add_node("planner", PlannerAgent())
    engine.add_node("coder", CoderAgent())
    engine.add_node("reviewer", ReviewerAgent())

    engine.add_edge("planner", "coder", condition="always")
    engine.add_edge("coder", "reviewer", condition="always")
    # Set max_cycles to 1: the first retry is the final attempt, triggering self-healing escalation
    engine.add_edge("reviewer", "coder", condition="review_rejected", max_cycles=1)
    engine.add_edge("reviewer", "end", condition="review_approved")

    initial_state = AgentState(session_id="self_heal_test", user_goal="Build foo")
    final_state = await engine.run(initial_state)

    assert final_state.status == "success"
    # Verify temperature bump was set in scratchpad during the retry
    assert final_state.scratchpad.get("temperature_bump") == 0.7


# --- C10: Scheduler Preload Hint ---


@pytest.mark.asyncio
async def test_scheduler_preload_hint(mock_tier1_profile):
    from conftest import MockOllamaClient

    client = MockOllamaClient()
    # Tier 1 (4GB VRAM): concurrency_allowed is False
    scheduler_constrained = ModelScheduler(client=client, profile=mock_tier1_profile)
    # Preload must be a strict no-op
    res = await scheduler_constrained.preload_hint("qwen2.5-coder:7b", role="coder")
    assert res is False

    # High capacity profile (24GB VRAM): concurrency_allowed is True
    high_profile = mock_tier1_profile.model_copy(
        update={
            "tier": HardwareTier.TIER_6_24GB,
            "total_vram_gb": 24.0,
            "gpus": [
                GPUInfo(
                    name="RTX 4090", vendor="NVIDIA", vram_total_mb=24576.0, vram_free_mb=24000.0
                )
            ],
        }
    )
    scheduler_high = ModelScheduler(client=client, profile=high_profile)
    client.currently_loaded = []
    res_high = await scheduler_high.preload_hint("qwen2.5-coder:7b", role="coder")
    assert isinstance(res_high, bool)
