"""Unit tests for enhancements: SearchReplaceTool, Observation Masking, Reviewer Gates, Coder Multi-task, Workflow Failure States, and Security Sandboxing."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from aglibol.agents.coder import CoderAgent
from aglibol.agents.planner import PlannerAgent
from aglibol.agents.reviewer import ReviewerAgent
from aglibol.core.context import ContextManager
from aglibol.core.events import EventBus
from aglibol.core.scheduler import ModelScheduler
from aglibol.core.types import (
    AgentState,
    ChatMessage,
    TaskItem,
    TaskPlan,
    TaskStatus,
)
from aglibol.core.workflow import WorkflowEngine
from aglibol.storage.brain import Brain
from aglibol.tools.file_ops import ReadFileTool, WriteFileTool
from aglibol.tools.registry import ToolRegistry
from aglibol.tools.search_replace import SearchReplaceTool
from aglibol.tools.shell import ShellCommandTool

# --- 1. SearchReplaceTool 4-Tier Matching Tests ---


@pytest.mark.asyncio
async def test_search_replace_exact_match():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        file_path = tmp_path / "hello.py"
        file_path.write_text("def hello():\n    return 'world'\n", encoding="utf-8")

        tool = SearchReplaceTool(workspace_root=tmp_path)
        res = await tool.execute(
            path=str(file_path),
            search="return 'world'",
            replace="return 'antigravity'",
        )
        assert res.success is True
        assert "Exact match" in res.output
        assert "return 'antigravity'" in file_path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_search_replace_whitespace_match():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        file_path = tmp_path / "whitespace.py"
        file_path.write_text("def test():   \n    x = 1   \n", encoding="utf-8")

        tool = SearchReplaceTool(workspace_root=tmp_path)
        res = await tool.execute(
            path=str(file_path),
            search="def test():\n    x = 1",
            replace="def test():\n    x = 2",
        )
        assert res.success is True
        assert "Whitespace-normalized" in res.output
        assert "x = 2" in file_path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_search_replace_fuzzy_match():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        file_path = tmp_path / "fuzzy.py"
        original = (
            "def calculate_total(items, tax_rate):\n"
            "    subtotal = sum(item.price for item in items)\n"
            "    tax = subtotal * tax_rate\n"
            "    return subtotal + tax\n"
        )
        file_path.write_text(original, encoding="utf-8")

        tool = SearchReplaceTool(workspace_root=tmp_path)
        # Search block with slight character difference
        search = (
            "def calculate_total(items, tax_rate):\n"
            "    subtotal = sum(i.price for i in items)\n"
            "    tax = subtotal * tax_rate\n"
            "    return subtotal + tax\n"
        )
        replace = (
            "def calculate_total(items, tax_rate, discount=0):\n"
            "    subtotal = sum(item.price for item in items) - discount\n"
            "    tax = subtotal * tax_rate\n"
            "    return subtotal + tax\n"
        )
        res = await tool.execute(path=str(file_path), search=search, replace=replace)
        assert res.success is True
        assert "Fuzzy match" in res.output
        assert "discount=0" in file_path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_search_replace_sandbox_violation():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        tool = SearchReplaceTool(workspace_root=tmp_path)
        res = await tool.execute(
            path=str(tmp_path.parent / "escape.txt"),
            search="foo",
            replace="bar",
        )
        assert res.success is False
        assert "Access denied" in res.error


# --- 2. Observation Masking in ContextManager ---


def test_observation_masking_preserves_user_goal_and_masks_tools():
    messages = [
        ChatMessage(role="system", content="You are an assistant."),
        ChatMessage(role="user", content="Build a complete REST API with authentication."),
        # Intermediate tool output (> 200 chars)
        ChatMessage(role="tool", content="A" * 500, tool_name="read_file"),
        ChatMessage(role="assistant", content="Analyzing file..."),
        ChatMessage(role="tool", content="B" * 600, tool_name="list_dir"),
        ChatMessage(role="assistant", content="Next step..."),
        ChatMessage(role="user", content="Please proceed with auth router."),
        ChatMessage(role="assistant", content="Working on it."),
    ]

    # Compress with a small budget to trigger masking
    compressed = ContextManager.compress_messages(
        messages, max_context_tokens=300, headroom_ratio=0.75
    )

    # System prompt MUST be preserved
    assert compressed[0].role == "system"
    # User original goal MUST be preserved at index 1
    assert compressed[1].role == "user"
    assert "REST API" in compressed[1].content
    # The last messages must be preserved
    assert compressed[-1].role == "assistant"

    # Any tool message should be masked
    for msg in compressed:
        if msg.role == "tool":
            assert "masked" in msg.content


# --- 3. Reviewer Deterministic AST Gate & False Positive Prevention ---


@pytest.mark.asyncio
async def test_reviewer_ast_gate_rejects_syntax_error(mock_tier1_profile):
    from conftest import MockOllamaClient

    client = MockOllamaClient()
    scheduler = ModelScheduler(client=client, profile=mock_tier1_profile)
    tool_registry = ToolRegistry()

    reviewer = ReviewerAgent()
    state = AgentState(
        session_id="ast_test",
        user_goal="Write a valid python script",
        artifacts={"broken.py": "def broken(\n    return 42\n"},  # Syntax error: unclosed paren
    )

    result_state = await reviewer.execute(state, client, scheduler, tool_registry)
    assert result_state.review_status == "rejected"
    assert "Deterministic Syntax Gate Failed" in result_state.review_feedback
    assert "SyntaxError" in result_state.review_feedback
    # Crucially: client.chat was NEVER called because AST gate intercepted first!
    assert len(client.mock_chat_responses) == 0


def test_reviewer_parse_decision_false_positive_prevention():
    # If the LLM discusses approval without approving
    content_rejected = (
        "I analyzed your code. I cannot grant DECISION: APPROVED because the functions "
        "have missing imports. DECISION: REJECTED\n"
        "1. Fix missing imports."
    )
    decision = ReviewerAgent._parse_decision(content_rejected)
    assert decision == "rejected"

    # Clean approval
    content_approved = "DECISION: APPROVED\nThe implementation meets all requirements."
    assert ReviewerAgent._parse_decision(content_approved) == "approved"


# --- 4. Coder Multi-Task DAG Execution ---


def test_coder_dag_task_dependency_resolution():
    plan = TaskPlan(
        plan_id="p1",
        goal="Build pipeline",
        tasks=[
            TaskItem(id="t1", title="Setup DB", description="Init schema", assigned_agent="coder"),
            TaskItem(
                id="t2",
                title="Create models",
                description="Pydantic models",
                assigned_agent="coder",
                dependencies=["t1"],
            ),
            TaskItem(
                id="t3",
                title="Create endpoints",
                description="FastAPI routes",
                assigned_agent="coder",
                dependencies=["t2"],
            ),
        ],
    )
    state = AgentState(session_id="dag_sess", user_goal="Pipeline", task_plan=plan)

    # First turn: only t1 should be ready
    ready = CoderAgent._get_pending_tasks(state)
    assert len(ready) == 1
    assert ready[0].id == "t1"

    # Complete t1
    ready[0].status = TaskStatus.COMPLETED

    # Second turn: t2 should be ready
    ready2 = CoderAgent._get_pending_tasks(state)
    assert len(ready2) == 1
    assert ready2[0].id == "t2"


def test_coder_extract_all_code_blocks():
    text = (
        "Here is the database file:\n"
        "```python\n"
        "# File: database.py\n"
        "import sqlite3\n"
        "def get_conn(): return sqlite3.connect('app.db')\n"
        "```\n\n"
        "And here is the main script:\n"
        "```python:main.py\n"
        "from database import get_conn\n"
        "print('Ready')\n"
        "```"
    )
    blocks = CoderAgent._extract_all_code_blocks(text)
    assert "database.py" in blocks
    assert "main.py" in blocks
    assert "import sqlite3" in blocks["database.py"]
    assert "from database import get_conn" in blocks["main.py"]


# --- 5. Workflow Failure State Handling ---


@pytest.mark.asyncio
async def test_workflow_max_cycles_sets_failed_status(mock_tier1_profile):
    from conftest import MockOllamaClient

    client = MockOllamaClient()
    # Mock responses: Planner valid -> Coder code -> Reviewer REJECTED repeatedly
    client.mock_chat_responses = [
        {
            "content": '{"summary": "Plan", "tasks": [{"id": "t1", "title": "Do work", "description": "code", "assigned_agent": "coder"}]}',
            "tool_calls": [],
            "done": True,
        },
        {"content": "```python\ndef test(): pass\n```", "tool_calls": [], "done": True},
        {"content": "DECISION: REJECTED\nNeeds more tests.", "tool_calls": [], "done": True},
        # Cycle 1 retry
        {"content": "```python\ndef test(): pass\n```", "tool_calls": [], "done": True},
        {"content": "DECISION: REJECTED\nStill needs tests.", "tool_calls": [], "done": True},
    ]

    event_bus = EventBus()
    scheduler = ModelScheduler(client=client, profile=mock_tier1_profile, event_bus=event_bus)
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
    # Limit to 1 cycle
    engine.add_edge("reviewer", "coder", condition="review_rejected", max_cycles=1)
    engine.add_edge("reviewer", "end", condition="review_approved")
    engine.set_entry_point("planner")

    initial_state = AgentState(session_id="fail_test", user_goal="Pass review")
    final_state = await engine.run(initial_state)

    # Workflow MUST be marked failed when max cycles exceeded, NOT completed successfully!
    assert final_state.status == "failed"
    assert final_state.error is not None
    assert "Max review" in final_state.error


# --- 6. Security Sandboxing & Shell Hardening ---


@pytest.mark.asyncio
async def test_shell_tool_blocks_dangerous_commands():
    tool = ShellCommandTool()
    res = await tool.execute(command="rm -rf /")
    assert res.success is False
    assert "Command rejected" in res.error

    res2 = await tool.execute(command="rmdir /s /q C:\\Windows")
    assert res2.success is False
    assert "Command rejected" in res2.error


@pytest.mark.asyncio
async def test_file_ops_workspace_sandboxing():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        write_tool = WriteFileTool(workspace_root=tmp_path)
        read_tool = ReadFileTool(workspace_root=tmp_path)

        # Attempt to write outside workspace
        outside_file = tmp_path.parent / "hacked.txt"
        res = await write_tool.execute(path=str(outside_file), content="bad")
        assert res.success is False
        assert "Access denied" in res.error

        # Attempt to read outside workspace
        res_read = await read_tool.execute(path=str(outside_file))
        assert res_read.success is False
        assert "Access denied" in res_read.error


# --- 7. Brain Clean Days ---


def test_brain_clean_older_than():
    with tempfile.TemporaryDirectory() as tmpdir:
        brain = Brain(root_dir=tmpdir)
        # Create two dummy sessions
        s1 = brain.get_session_dir("sess_old")
        s2 = brain.get_session_dir("sess_new")
        (s1 / "meta.json").write_text("{}", encoding="utf-8")
        (s2 / "meta.json").write_text("{}", encoding="utf-8")

        # Set s1's modification time to 10 days ago
        import os
        import time

        ten_days_ago = time.time() - (10 * 86400)
        os.utime(str(s1), (ten_days_ago, ten_days_ago))

        # Clean older than 5 days
        cleaned = brain.clean_older_than(days=5)
        assert cleaned == 1
        assert not s1.exists()
        assert s2.exists()


# --- 8. Coder Reasoning & Tool Artifact Tests ---


@pytest.mark.asyncio
async def test_coder_extracts_code_from_thinking(mock_tier1_profile):
    from conftest import MockOllamaClient

    with tempfile.TemporaryDirectory() as tmpdir:
        client = MockOllamaClient()
        # Reasoning model returns code block only in thinking
        client.mock_chat_responses = [
            {
                "content": "",
                "thinking": "<thinking>\nI will write prime_checker.py\n```python:prime_checker.py\ndef is_prime(n):\n    return n > 1 and all(n % i != 0 for i in range(2, int(n**0.5) + 1))\n```\n</thinking>",
                "tool_calls": [],
                "done": True,
                "eval_count": 40,
            }
        ]
        scheduler = ModelScheduler(client=client, profile=mock_tier1_profile)
        registry = ToolRegistry()
        state = AgentState(
            session_id="test_coder_thinking",
            user_goal="Create prime_checker.py",
            workspace_dir=tmpdir,
        )
        coder = CoderAgent()
        updated_state = await coder.execute(state, client, scheduler, registry)

        assert "prime_checker.py" in updated_state.artifacts
        assert "def is_prime" in updated_state.artifacts["prime_checker.py"]
        # Ensure it was persisted to disk in workspace
        saved_file = Path(tmpdir) / "prime_checker.py"
        assert saved_file.exists()
        assert "def is_prime" in saved_file.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_coder_ingests_tool_written_files(mock_tier1_profile):
    from conftest import MockOllamaClient

    with tempfile.TemporaryDirectory() as tmpdir:
        client = MockOllamaClient()
        # Model invokes write_file tool
        client.mock_chat_responses = [
            {
                "content": "I have created the file using tool.",
                "thinking": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "write_file",
                            "arguments": {
                                "path": str(Path(tmpdir) / "script.py"),
                                "content": "print('hello from tool')\n",
                            },
                        }
                    }
                ],
                "done": True,
                "eval_count": 30,
            },
            {
                "content": "Finished implementing script.py.",
                "thinking": "",
                "tool_calls": [],
                "done": True,
                "eval_count": 10,
            },
        ]
        scheduler = ModelScheduler(client=client, profile=mock_tier1_profile)
        registry = ToolRegistry()
        write_tool = WriteFileTool(workspace_root=Path(tmpdir))
        registry.register(write_tool)

        state = AgentState(
            session_id="test_tool_artifacts",
            user_goal="Write script.py",
            workspace_dir=tmpdir,
        )
        coder = CoderAgent()
        updated_state = await coder.execute(state, client, scheduler, registry)

        assert "script.py" in updated_state.artifacts
        assert "print('hello from tool')" in updated_state.artifacts["script.py"]


def test_live_thought_box_lifecycle():
    from aglibol.cli.interactive.tui_renderer import TUIRenderer

    box = TUIRenderer.create_live_thought_box("assistant")
    box.start("Starting analysis...")
    box.update_token("Step 1: Parse input\n")
    box.update_token("Step 2: Generate response\n")
    box.update_status("Finalizing response...")
    box.finish(summary="Completed thinking.")

    assert len(box.lines) == 2
    assert "Step 1: Parse input" in box.lines[0]
