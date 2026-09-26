"""Tests for WorkflowEngine, ModelScheduler, and multi-agent pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from aglibol.agents.coder import CoderAgent
from aglibol.agents.planner import PlannerAgent
from aglibol.agents.reviewer import ReviewerAgent
from aglibol.core.events import EventBus
from aglibol.core.scheduler import ModelScheduler
from aglibol.core.types import AgentState
from aglibol.core.workflow import WorkflowEngine
from aglibol.storage.artifact_store import ArtifactStore
from aglibol.storage.checkpoint import CheckpointStore
from aglibol.tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_workflow_execution_with_mock_client(mock_tier1_profile):
    from conftest import MockOllamaClient

    client = MockOllamaClient()
    # Mock responses for Planner, Coder, Reviewer
    client.mock_chat_responses = [
        # Planner response (valid json)
        {
            "content": '{"summary": "Plan overview", "tasks": [{"id": "t1", "title": "Implement core", "description": "Write function", "assigned_agent": "coder"}]}',
            "tool_calls": [],
            "done": True,
            "eval_count": 30,
        },
        # Coder response (code block)
        {
            "content": "Here is the code:\n```python\ndef fib(n):\n    return n if n <= 1 else fib(n-1) + fib(n-2)\n```",
            "tool_calls": [],
            "done": True,
            "eval_count": 50,
        },
        # Reviewer response (approval)
        {
            "content": "DECISION: APPROVED\nEverything looks great.",
            "tool_calls": [],
            "done": True,
            "eval_count": 20,
        },
    ]

    event_bus = EventBus()
    events_received: list[str] = []

    async def _tracker(ev):
        events_received.append(ev.event_type)

    event_bus.subscribe("*", _tracker)

    scheduler = ModelScheduler(client=client, profile=mock_tier1_profile, event_bus=event_bus)
    tool_registry = ToolRegistry()

    with tempfile.TemporaryDirectory() as tmpdir:
        session_dir = Path(tmpdir)
        ckpt_store = CheckpointStore(session_dir / "checkpoints.db")
        artifact_store = ArtifactStore(session_dir)

        engine = WorkflowEngine(
            client=client,
            scheduler=scheduler,
            tool_registry=tool_registry,
            checkpoint_store=ckpt_store,
            artifact_store=artifact_store,
            event_bus=event_bus,
            max_total_steps=5,
        )

        engine.add_node("planner", PlannerAgent())
        engine.add_node("coder", CoderAgent())
        engine.add_node("reviewer", ReviewerAgent())

        engine.add_edge("planner", "coder", condition="always")
        engine.add_edge("coder", "reviewer", condition="always")
        engine.add_edge("reviewer", "end", condition="review_approved")
        engine.set_entry_point("planner")

        initial_state = AgentState(
            session_id="test_e2e_sess",
            user_goal="Write a fibonacci function",
        )

        final_state = await engine.run(initial_state)

        # Assertions
        assert final_state.is_completed is True
        assert final_state.review_status == "approved"
        assert len(final_state.artifacts) > 0
        assert "WORKFLOW_STARTED" in events_received
        assert "WORKFLOW_COMPLETED" in events_received
        assert "PLAN_CREATED" in events_received
        assert "CODE_GENERATED" in events_received
        assert "REVIEW_COMPLETED" in events_received

        # Verify disk checkpoint was saved
        restored = ckpt_store.get_latest_checkpoint("test_e2e_sess")
        assert restored is not None
        assert restored.review_status == "approved"

        # Verify artifact was persisted to disk
        artifacts = artifact_store.list_artifacts()
        assert len(artifacts) > 0


@pytest.mark.asyncio
async def test_workflow_resumption_from_checkpoint(mock_tier1_profile):
    from conftest import MockOllamaClient

    client = MockOllamaClient()
    # Mock responses for Coder and Reviewer (Planner already ran before checkpoint)
    client.mock_chat_responses = [
        # Coder response (code block)
        {
            "content": "Here is the code:\n```python\ndef greet(name):\n    return f'Hello, {name}!'\n```",
            "tool_calls": [],
            "done": True,
            "eval_count": 40,
        },
        # Reviewer response (approval)
        {
            "content": "DECISION: APPROVED\nClean implementation.",
            "tool_calls": [],
            "done": True,
            "eval_count": 20,
        },
    ]

    event_bus = EventBus()
    steps_executed: list[tuple[int, str]] = []

    async def _tracker(ev):
        if ev.event_type == "STEP_STARTED":
            steps_executed.append((ev.step, ev.agent))

    event_bus.subscribe("*", _tracker)

    scheduler = ModelScheduler(client=client, profile=mock_tier1_profile, event_bus=event_bus)
    tool_registry = ToolRegistry()

    with tempfile.TemporaryDirectory() as tmpdir:
        session_dir = Path(tmpdir)
        ckpt_store = CheckpointStore(session_dir / "checkpoints.db")
        artifact_store = ArtifactStore(session_dir)

        engine = WorkflowEngine(
            client=client,
            scheduler=scheduler,
            tool_registry=tool_registry,
            checkpoint_store=ckpt_store,
            artifact_store=artifact_store,
            event_bus=event_bus,
            max_total_steps=5,
        )

        engine.add_node("planner", PlannerAgent())
        engine.add_node("coder", CoderAgent())
        engine.add_node("reviewer", ReviewerAgent())

        engine.add_edge("planner", "coder", condition="always")
        engine.add_edge("coder", "reviewer", condition="always")
        engine.add_edge("reviewer", "end", condition="review_approved")
        engine.set_entry_point("planner")

        # Simulate state saved after step 1 (Planner finished)
        interrupted_state = AgentState(
            session_id="resumed_sess_test",
            user_goal="Write a greeting function",
            current_step=1,
            active_agent="planner",
            plan={
                "summary": "Pre-existing plan",
                "tasks": [{"id": "t1", "title": "Implement greet"}],
            },
        )

        final_state = await engine.run(interrupted_state)

        # Assertions
        assert final_state.is_completed is True
        assert final_state.review_status == "approved"
        # Step counter continued from 1 -> 2 -> 3
        assert final_state.current_step == 3
        assert steps_executed == [(2, "coder"), (3, "reviewer")]


@pytest.mark.asyncio
async def test_workflow_build_from_yaml(mock_tier1_profile):
    from conftest import MockOllamaClient

    client = MockOllamaClient()
    client.mock_chat_responses = [
        # Planner
        {
            "content": '{"summary": "YAML plan", "tasks": [{"id": "t1", "title": "Write helper", "assigned_agent": "coder"}]}',
            "tool_calls": [],
            "done": True,
            "eval_count": 25,
        },
        # Coder
        {
            "content": "```python\ndef add(a, b):\n    return a + b\n```",
            "tool_calls": [],
            "done": True,
            "eval_count": 35,
        },
        # Reviewer
        {
            "content": "DECISION: APPROVED\nAll good.",
            "tool_calls": [],
            "done": True,
            "eval_count": 15,
        },
    ]

    event_bus = EventBus()
    scheduler = ModelScheduler(client=client, profile=mock_tier1_profile, event_bus=event_bus)
    tool_registry = ToolRegistry()

    with tempfile.TemporaryDirectory() as tmpdir:
        session_dir = Path(tmpdir)
        ckpt_store = CheckpointStore(session_dir / "checkpoints.db")
        artifact_store = ArtifactStore(session_dir)

        engine = WorkflowEngine.build_from_yaml(
            yaml_path="default.yaml",
            client=client,
            scheduler=scheduler,
            tool_registry=tool_registry,
            checkpoint_store=ckpt_store,
            artifact_store=artifact_store,
            event_bus=event_bus,
            model_overrides={
                "planner": "test-planner",
                "coder": "test-coder",
                "reviewer": "test-reviewer",
            },
            max_total_steps=5,
        )

        assert "planner" in engine.nodes
        assert "coder" in engine.nodes
        assert "reviewer" in engine.nodes
        assert engine.entry_point == "planner"
        assert len(engine.edges) == 4

        state = AgentState(
            session_id="yaml_wf_test",
            user_goal="Create an add function",
        )

        final_state = await engine.run(state)
        assert final_state.is_completed is True
        assert final_state.status == "success"
        assert final_state.review_status == "approved"
