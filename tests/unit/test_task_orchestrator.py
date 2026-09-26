"""Unit tests for TaskOrchestrator sequential DAG task execution engine."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aglibol.agents.coder import CoderAgent
from aglibol.agents.reviewer import ReviewerAgent
from aglibol.core.task_orchestrator import TaskOrchestrator
from aglibol.core.types import (
    AgentState,
    TaskItem,
    TaskPlan,
    TaskStatus,
)
from aglibol.memory.episodic import EpisodicMemory
from aglibol.ollama.client import OllamaClient
from aglibol.storage.artifact_store import ArtifactStore
from aglibol.storage.checkpoint import CheckpointStore
from aglibol.tools.registry import ToolRegistry


@pytest.fixture
def mock_scheduler():
    scheduler = MagicMock()
    scheduler.acquire_model = AsyncMock(return_value=("qwen2.5-coder:7b", {"num_ctx": 4096}, "0"))
    scheduler.release_current_model = AsyncMock(return_value=None)
    scheduler.current_loaded_model = None
    return scheduler


@pytest.fixture
def mock_client():
    client = MagicMock(spec=OllamaClient)
    client.chat = AsyncMock(return_value={"content": "Done", "done_reason": "stop"})
    return client


@pytest.fixture
def mock_tool_registry():
    return ToolRegistry()


@pytest.mark.asyncio
async def test_task_orchestrator_initialization(mock_client, mock_scheduler, mock_tool_registry):
    orchestrator = TaskOrchestrator(
        client=mock_client,
        scheduler=mock_scheduler,
        tool_registry=mock_tool_registry,
    )
    assert orchestrator.max_review_cycles == 3
    assert orchestrator.client == mock_client
    assert orchestrator.scheduler == mock_scheduler


@pytest.mark.asyncio
async def test_task_orchestrator_single_task_fallback(
    mock_client, mock_scheduler, mock_tool_registry
):
    with tempfile.TemporaryDirectory() as tmpdir:
        state = AgentState(
            session_id="test_fallback",
            user_goal="Create a simple script",
            workspace_dir=tmpdir,
        )

        async def mock_coder_exec(*args, **kwargs):
            st = kwargs.get("state") or args[0]
            st.artifacts["script.py"] = "print('hello')"
            return st

        async def mock_reviewer_exec(*args, **kwargs):
            st = kwargs.get("state") or args[0]
            st.review_status = "approved"
            return st

        orchestrator = TaskOrchestrator(
            client=mock_client,
            scheduler=mock_scheduler,
            tool_registry=mock_tool_registry,
        )

        with (
            patch("aglibol.core.task_orchestrator.CoderAgent.execute", side_effect=mock_coder_exec),
            patch(
                "aglibol.core.task_orchestrator.ReviewerAgent.execute",
                side_effect=mock_reviewer_exec,
            ),
        ):
            result = await orchestrator.execute_plan(state)

        assert result.status == "success"
        assert result.is_completed is True
        assert "script.py" in result.artifacts
        assert len(result.task_plan.tasks) == 1
        assert result.task_plan.tasks[0].status == TaskStatus.COMPLETED


@pytest.mark.asyncio
async def test_task_orchestrator_multi_task_sequential(
    mock_client, mock_scheduler, mock_tool_registry
):
    with tempfile.TemporaryDirectory() as tmpdir:
        tasks = [
            TaskItem(
                id="t1",
                title="Create HTML",
                description="index.html",
                assigned_agent="coder",
                dependencies=[],
            ),
            TaskItem(
                id="t2",
                title="Create CSS",
                description="styles.css",
                assigned_agent="coder",
                dependencies=["t1"],
            ),
            TaskItem(
                id="t3",
                title="Create JS",
                description="app.js",
                assigned_agent="coder",
                dependencies=["t2"],
            ),
        ]
        plan = TaskPlan(plan_id="p1", goal="Build web page", tasks=tasks)
        state = AgentState(
            session_id="test_multi",
            user_goal="Build web page",
            workspace_dir=tmpdir,
            task_plan=plan,
        )

        executed_tasks = []

        async def mock_coder_exec(*args, **kwargs):
            st = kwargs.get("state") or args[0]
            executed_tasks.append(st.current_task_id)
            if st.current_task_id == "t1":
                st.artifacts["index.html"] = "<html></html>"
            elif st.current_task_id == "t2":
                st.artifacts["styles.css"] = "body { margin: 0; }"
            elif st.current_task_id == "t3":
                st.artifacts["app.js"] = "console.log('hi');"
            return st

        async def mock_reviewer_exec(*args, **kwargs):
            st = kwargs.get("state") or args[0]
            st.review_status = "approved"
            return st

        orchestrator = TaskOrchestrator(
            client=mock_client,
            scheduler=mock_scheduler,
            tool_registry=mock_tool_registry,
        )

        with (
            patch("aglibol.core.task_orchestrator.CoderAgent.execute", side_effect=mock_coder_exec),
            patch(
                "aglibol.core.task_orchestrator.ReviewerAgent.execute",
                side_effect=mock_reviewer_exec,
            ),
        ):
            result = await orchestrator.execute_plan(state)

        assert executed_tasks == ["t1", "t2", "t3"]
        assert result.status == "success"
        assert all(t.status == TaskStatus.COMPLETED for t in result.task_plan.tasks)
        assert "index.html" in result.artifacts
        assert "styles.css" in result.artifacts
        assert "app.js" in result.artifacts


@pytest.mark.asyncio
async def test_task_orchestrator_skips_already_completed(
    mock_client, mock_scheduler, mock_tool_registry
):
    """Verify resuming from a saved checkpoint where task 1 was already completed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tasks = [
            TaskItem(
                id="t1",
                title="HTML",
                description="index.html",
                assigned_agent="coder",
                status=TaskStatus.COMPLETED,
            ),
            TaskItem(
                id="t2",
                title="CSS",
                description="styles.css",
                assigned_agent="coder",
                dependencies=["t1"],
            ),
        ]
        plan = TaskPlan(plan_id="p2", goal="Resume test", tasks=tasks)
        state = AgentState(
            session_id="test_resume",
            user_goal="Resume test",
            workspace_dir=tmpdir,
            task_plan=plan,
            artifacts={"index.html": "<html></html>"},
        )

        executed_tasks = []

        async def mock_coder_exec(*args, **kwargs):
            st = kwargs.get("state") or args[0]
            executed_tasks.append(st.current_task_id)
            st.artifacts["styles.css"] = "body {}"
            return st

        async def mock_reviewer_exec(*args, **kwargs):
            st = kwargs.get("state") or args[0]
            st.review_status = "approved"
            return st

        orchestrator = TaskOrchestrator(
            client=mock_client,
            scheduler=mock_scheduler,
            tool_registry=mock_tool_registry,
        )

        with (
            patch("aglibol.core.task_orchestrator.CoderAgent.execute", side_effect=mock_coder_exec),
            patch(
                "aglibol.core.task_orchestrator.ReviewerAgent.execute",
                side_effect=mock_reviewer_exec,
            ),
        ):
            result = await orchestrator.execute_plan(state)

        # t1 was skipped, only t2 ran
        assert executed_tasks == ["t2"]
        assert result.status == "success"
        assert result.task_plan.tasks[0].status == TaskStatus.COMPLETED
        assert result.task_plan.tasks[1].status == TaskStatus.COMPLETED


@pytest.mark.asyncio
async def test_task_orchestrator_persists_checkpoints(
    mock_client, mock_scheduler, mock_tool_registry
):
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "checkpoints.db"
        ckpt_store = CheckpointStore(db_path)
        art_store = ArtifactStore(Path(tmpdir))
        ep_mem = EpisodicMemory(Path(tmpdir) / "episodic.jsonl")

        tasks = [
            TaskItem(id="t1", title="Step 1", description="step 1", assigned_agent="coder"),
            TaskItem(
                id="t2",
                title="Step 2",
                description="step 2",
                assigned_agent="coder",
                dependencies=["t1"],
            ),
        ]
        plan = TaskPlan(plan_id="p3", goal="Checkpoints test", tasks=tasks)
        state = AgentState(
            session_id="test_ckpt",
            user_goal="Checkpoints test",
            workspace_dir=tmpdir,
            task_plan=plan,
        )

        async def mock_coder_exec(*args, **kwargs):
            st = kwargs.get("state") or args[0]
            st.artifacts[f"{st.current_task_id}.txt"] = "content"
            return st

        async def mock_reviewer_exec(*args, **kwargs):
            st = kwargs.get("state") or args[0]
            st.review_status = "approved"
            return st

        orchestrator = TaskOrchestrator(
            client=mock_client,
            scheduler=mock_scheduler,
            tool_registry=mock_tool_registry,
            checkpoint_store=ckpt_store,
            artifact_store=art_store,
            episodic_memory=ep_mem,
        )

        with (
            patch("aglibol.core.task_orchestrator.CoderAgent.execute", side_effect=mock_coder_exec),
            patch(
                "aglibol.core.task_orchestrator.ReviewerAgent.execute",
                side_effect=mock_reviewer_exec,
            ),
        ):
            result = await orchestrator.execute_plan(state)

        assert result.status == "success"
        # Check that checkpoints exist in db
        checkpoints = ckpt_store.list_checkpoints("test_ckpt")
        assert len(checkpoints) >= 2

        # Check episodic memory
        records = ep_mem.read_all_turns()
        assert len(records) >= 2


@pytest.mark.asyncio
async def test_task_orchestrator_dependency_failure_skips_unmet(
    mock_client, mock_scheduler, mock_tool_registry
):
    with tempfile.TemporaryDirectory() as tmpdir:
        tasks = [
            TaskItem(id="t1", title="Fail Task", description="fail", assigned_agent="coder"),
            TaskItem(
                id="t2",
                title="Dependent Task",
                description="dep",
                assigned_agent="coder",
                dependencies=["t1"],
            ),
        ]
        plan = TaskPlan(plan_id="p4", goal="Failure test", tasks=tasks)
        state = AgentState(
            session_id="test_fail",
            user_goal="Failure test",
            workspace_dir=tmpdir,
            task_plan=plan,
        )

        async def mock_coder_exec(*args, **kwargs):
            return kwargs.get("state") or args[0]

        async def mock_reviewer_exec(*args, **kwargs):
            st = kwargs.get("state") or args[0]
            st.review_status = "rejected"
            st.review_feedback = "Syntax error in code"
            return st

        orchestrator = TaskOrchestrator(
            client=mock_client,
            scheduler=mock_scheduler,
            tool_registry=mock_tool_registry,
            max_review_cycles=1,
        )

        with (
            patch("aglibol.core.task_orchestrator.CoderAgent.execute", side_effect=mock_coder_exec),
            patch(
                "aglibol.core.task_orchestrator.ReviewerAgent.execute",
                side_effect=mock_reviewer_exec,
            ),
        ):
            result = await orchestrator.execute_plan(state)

        assert result.status == "failed"
        assert result.task_plan.tasks[0].status == TaskStatus.FAILED
        assert result.task_plan.tasks[1].status == TaskStatus.SKIPPED


@pytest.mark.asyncio
async def test_task_orchestrator_writer_agent(mock_client, mock_scheduler, mock_tool_registry):
    with tempfile.TemporaryDirectory() as tmpdir:
        tasks = [
            TaskItem(id="t1", title="Write Guide", description="guide.md", assigned_agent="writer"),
        ]
        plan = TaskPlan(plan_id="p5", goal="Docs", tasks=tasks)
        state = AgentState(
            session_id="test_writer",
            user_goal="Docs",
            workspace_dir=tmpdir,
            task_plan=plan,
        )

        async def mock_writer_exec(*args, **kwargs):
            st = kwargs.get("state") or args[0]
            st.artifacts["guide.md"] = "# User Guide"
            return st

        orchestrator = TaskOrchestrator(
            client=mock_client,
            scheduler=mock_scheduler,
            tool_registry=mock_tool_registry,
        )

        with patch(
            "aglibol.core.task_orchestrator.WriterAgent.execute", side_effect=mock_writer_exec
        ):
            result = await orchestrator.execute_plan(state)

        assert result.status == "success"
        assert result.task_plan.tasks[0].status == TaskStatus.COMPLETED
        assert "guide.md" in result.artifacts


def test_coder_prose_is_never_extracted_as_python_code():
    # 1. English conversational prose containing keywords 'from', 'class', 'import'
    prose_sample = (
        "Let me understand the task. The user wants me to plan a program from scratch.\n"
        "I will define classes of numbers and import nothing.\n"
        "Let me know if you would like me to proceed."
    )
    # _is_valid_python must reject plain conversational English
    assert CoderAgent._is_valid_python(prose_sample) is False

    # 2. Untagged fenced block containing conversational text must not be labeled .py
    untagged_fenced = f"```\n{prose_sample}\n```"
    extracted = CoderAgent._extract_all_code_blocks(untagged_fenced)
    assert not any(k.endswith(".py") for k in extracted.keys())

    # 3. Valid python code block should be extracted properly
    valid_code = "def is_prime(n: int) -> bool:\n    return n > 1\n"
    assert CoderAgent._is_valid_python(valid_code) is True
    valid_fenced = f"```python\n# File: math_utils.py\n{valid_code}\n```"
    valid_extracted = CoderAgent._extract_all_code_blocks(valid_fenced)
    assert "math_utils.py" in valid_extracted
    assert "def is_prime" in valid_extracted["math_utils.py"]


@pytest.mark.asyncio
async def test_reviewer_purges_invalid_ast_artifacts(
    mock_client, mock_scheduler, mock_tool_registry
):
    with tempfile.TemporaryDirectory() as tmpdir:
        broken_filename = "bad_syntax.py"
        broken_code = "def broken(\n   syntax error unterminated"
        disk_path = Path(tmpdir) / broken_filename
        disk_path.write_text(broken_code, encoding="utf-8")

        state = AgentState(
            session_id="test_purge",
            user_goal="Write a script",
            workspace_dir=tmpdir,
            artifacts={broken_filename: broken_code},
        )

        reviewer = ReviewerAgent()
        result_state = await reviewer.execute(
            state=state,
            client=mock_client,
            scheduler=mock_scheduler,
            tool_registry=mock_tool_registry,
        )

        # Reviewer must reject syntax errors deterministically
        assert result_state.review_status == "rejected"
        assert "Deterministic Syntax Gate Failed" in result_state.review_feedback

        # Crucial check: broken artifact must be purged from memory and unlinked from disk
        assert broken_filename not in result_state.artifacts
        assert not disk_path.exists()


@pytest.mark.asyncio
async def test_reviewer_rejects_empty_deliverables(mock_client, mock_scheduler, mock_tool_registry):
    state = AgentState(
        session_id="test_empty",
        user_goal="Write a program",
        artifacts={},
    )
    reviewer = ReviewerAgent()
    result_state = await reviewer.execute(
        state=state,
        client=mock_client,
        scheduler=mock_scheduler,
        tool_registry=mock_tool_registry,
    )
    assert result_state.review_status == "rejected"
    assert "No code deliverables" in result_state.review_feedback
