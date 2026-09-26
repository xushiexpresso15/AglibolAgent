"""Task orchestrator for sequential, checkpointed DAG execution."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from aglibol.agents.coder import CoderAgent
from aglibol.agents.reviewer import ReviewerAgent
from aglibol.agents.writer import WriterAgent
from aglibol.core.events import AgentEvent, EventBus
from aglibol.core.scheduler import ModelScheduler
from aglibol.core.types import (
    AgentState,
    ChatMessage,
    TaskItem,
    TaskPlan,
    TaskStatus,
)
from aglibol.memory.episodic import EpisodicMemory
from aglibol.ollama.client import OllamaClient
from aglibol.storage.artifact_store import ArtifactStore
from aglibol.storage.checkpoint import CheckpointStore
from aglibol.tools.registry import ToolRegistry

logger = logging.getLogger("aglibol.core.task_orchestrator")


class TaskOrchestrator:
    """
    Executes a TaskPlan step-by-step in topological/dependency order.
    Features:
    1. Isolated, uncluttered context per subtask so small local models stay focused.
    2. Checkpoints saved to disk after every completed task (instant resumption).
    3. Self-correcting Coder -> Reviewer critique loop per subtask.
    4. Consolidated workspace artifacts across all steps.
    """

    def __init__(
        self,
        client: OllamaClient,
        scheduler: ModelScheduler,
        tool_registry: ToolRegistry,
        checkpoint_store: CheckpointStore | None = None,
        artifact_store: ArtifactStore | None = None,
        event_bus: EventBus | None = None,
        episodic_memory: EpisodicMemory | None = None,
        model_overrides: dict[str, str] | None = None,
        max_review_cycles: int = 3,
    ) -> None:
        self.client = client
        self.scheduler = scheduler
        self.tool_registry = tool_registry
        self.checkpoint_store = checkpoint_store
        self.artifact_store = artifact_store
        self.event_bus = event_bus
        self.episodic_memory = episodic_memory
        self.model_overrides = model_overrides or {}
        self.max_review_cycles = max_review_cycles

    async def execute_plan(
        self,
        state: AgentState,
        on_task_update: Callable[[int, int, TaskItem], None] | None = None,
    ) -> AgentState:
        """
        Execute all tasks in the plan sequentially.
        Skips already completed tasks to allow seamless session resumption.
        """
        # Ensure task plan exists; if not, wrap single goal as a single-step plan
        if not state.task_plan or not state.task_plan.tasks:
            default_task = TaskItem(
                id="task_1",
                title="Implement objective",
                description=state.user_goal,
                assigned_agent="coder",
                dependencies=[],
                status=TaskStatus.PENDING,
            )
            state.task_plan = TaskPlan(
                plan_id="plan_direct",
                goal=state.user_goal,
                summary="Direct single-task execution",
                tasks=[default_task],
            )

        tasks = state.task_plan.tasks
        total_tasks = len(tasks)

        logger.info(
            "TaskOrchestrator starting execution of %d task(s) for session %s",
            total_tasks,
            state.session_id,
        )

        completed_task_ids = {t.id for t in tasks if t.status == TaskStatus.COMPLETED}

        while True:
            # Find next ready pending/retrying task whose dependencies are satisfied
            ready_task: TaskItem | None = None
            for t in tasks:
                if t.status in (TaskStatus.PENDING, TaskStatus.RETRYING):
                    if all(dep in completed_task_ids for dep in t.dependencies):
                        ready_task = t
                        break

            if not ready_task:
                # Check if there are remaining pending tasks blocked by failures
                unresolved = [
                    t for t in tasks if t.status in (TaskStatus.PENDING, TaskStatus.RETRYING)
                ]
                for u in unresolved:
                    u.status = TaskStatus.SKIPPED
                    u.error = "Skipped due to unmet dependency."
                break

            task_index = tasks.index(ready_task) + 1
            ready_task.status = TaskStatus.IN_PROGRESS
            state.current_task_id = ready_task.id
            state.current_step += 1

            logger.info(
                "Executing task %d/%d: [%s] %s",
                task_index,
                total_tasks,
                ready_task.id,
                ready_task.title,
            )

            if on_task_update:
                try:
                    on_task_update(task_index, total_tasks, ready_task)
                except Exception:
                    pass

            if self.event_bus:
                await self.event_bus.emit(
                    AgentEvent(
                        event_type="TASK_STARTED",
                        session_id=state.session_id,
                        step=state.current_step,
                        agent=ready_task.assigned_agent,
                        data={
                            "task_id": ready_task.id,
                            "title": ready_task.title,
                            "task_index": task_index,
                            "total_tasks": total_tasks,
                            "assigned_agent": ready_task.assigned_agent,
                        },
                    )
                )

            # Execute the single task
            state = await self._execute_task(state, ready_task)

            # Persist progress after each task
            if ready_task.status == TaskStatus.COMPLETED:
                completed_task_ids.add(ready_task.id)

            self._persist_step(state, ready_task)

            if self.event_bus:
                ev_type = (
                    "TASK_COMPLETED" if ready_task.status == TaskStatus.COMPLETED else "TASK_FAILED"
                )
                await self.event_bus.emit(
                    AgentEvent(
                        event_type=ev_type,
                        session_id=state.session_id,
                        step=state.current_step,
                        agent=ready_task.assigned_agent,
                        data={
                            "task_id": ready_task.id,
                            "status": ready_task.status.value,
                            "result": ready_task.result,
                            "error": ready_task.error,
                        },
                    )
                )

            # If task failed and subsequent tasks depend on it, loop will naturally skip dependents
            if ready_task.status == TaskStatus.FAILED:
                logger.warning("Task %s failed: %s", ready_task.id, ready_task.error)

        # Evaluate overall plan outcome
        all_completed = all(t.status == TaskStatus.COMPLETED for t in tasks)
        if all_completed:
            state.status = "success"
            state.is_completed = True
        else:
            state.status = "failed"
            state.is_completed = True
            failed_ids = [t.id for t in tasks if t.status == TaskStatus.FAILED]
            state.error = f"Execution stopped with failed task(s): {', '.join(failed_ids)}"

        # Release VRAM
        try:
            await self.scheduler.release_current_model()
        except Exception:
            pass

        return state

    async def _execute_task(self, state: AgentState, task: TaskItem) -> AgentState:
        """Execute a single task with its assigned agent and iterative critique loop."""
        state.scratchpad["current_task_modifications"] = {}
        assigned = task.assigned_agent.lower()

        if assigned == "writer":
            writer_model = self.model_overrides.get("writer", "qwen2.5:7b")
            writer = WriterAgent(model_name=writer_model)
            state = await writer.execute(
                state=state,
                client=self.client,
                scheduler=self.scheduler,
                tool_registry=self.tool_registry,
                event_bus=self.event_bus,
            )
            task.status = TaskStatus.COMPLETED
            task.result = "Documentation generated"
            return state

        elif assigned == "reviewer":
            reviewer_model = self.model_overrides.get("reviewer", "qwen2.5:7b")
            reviewer = ReviewerAgent(model_name=reviewer_model)
            state = await reviewer.execute(
                state=state,
                client=self.client,
                scheduler=self.scheduler,
                tool_registry=self.tool_registry,
                event_bus=self.event_bus,
            )
            if state.review_status == "approved":
                task.status = TaskStatus.COMPLETED
                task.result = state.review_feedback or "Approved"
            else:
                task.status = TaskStatus.FAILED
                task.error = state.review_feedback or "Review rejected"
            return state

        else:
            # Default Coder -> Reviewer critique loop
            coder_model = self.model_overrides.get("coder", "qwen2.5-coder:7b")
            reviewer_model = self.model_overrides.get("reviewer", "qwen2.5:7b")

            for cycle in range(1, self.max_review_cycles + 1):
                coder = CoderAgent(model_name=coder_model)
                state = await coder.execute(
                    state=state,
                    client=self.client,
                    scheduler=self.scheduler,
                    tool_registry=self.tool_registry,
                    event_bus=self.event_bus,
                    episodic_memory=self.episodic_memory,
                )

                reviewer = ReviewerAgent(model_name=reviewer_model)
                state = await reviewer.execute(
                    state=state,
                    client=self.client,
                    scheduler=self.scheduler,
                    tool_registry=self.tool_registry,
                    event_bus=self.event_bus,
                )

                if state.review_status == "approved":
                    task.status = TaskStatus.COMPLETED
                    task.result = state.review_feedback or "Task approved by reviewer"
                    return state

                # If rejected and still have cycles, prepare state for next attempt
                if cycle < self.max_review_cycles:
                    task.status = TaskStatus.RETRYING
                    logger.info(
                        "Task %s rejected (cycle %d/%d). Retrying...",
                        task.id,
                        cycle,
                        self.max_review_cycles,
                    )

            # Max cycles exceeded
            task.status = TaskStatus.FAILED
            task.error = state.review_feedback or "Task failed review after maximum iterations"
            return state

    def _persist_step(self, state: AgentState, task: TaskItem) -> None:
        """Save disk checkpoints and log episodic turns after each task step."""
        # 1. SQLite state snapshot
        if self.checkpoint_store:
            try:
                self.checkpoint_store.save_checkpoint(state)
            except Exception as e:
                logger.warning("Checkpoint save failed for task %s: %s", task.id, e)

        # 2. Persist artifacts to disk store and workspace
        if state.artifacts:
            ws_path = Path(state.workspace_dir) if state.workspace_dir else None
            for fname, content in state.artifacts.items():
                if self.artifact_store:
                    try:
                        self.artifact_store.save_artifact(fname, content)
                    except Exception:
                        pass
                if ws_path:
                    try:
                        dest = ws_path / fname
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        dest.write_text(content, encoding="utf-8")
                    except Exception:
                        pass

        # 3. Episodic memory log
        if self.episodic_memory:
            try:
                summary = f"Task [{task.id}] ({task.status.value}): {task.title}"
                self.episodic_memory.log_turn(
                    agent=task.assigned_agent,
                    step=state.current_step,
                    message=ChatMessage(role="assistant", content=summary),
                    extra={
                        "task_id": task.id,
                        "status": task.status.value,
                        "artifacts": list(state.artifacts.keys()),
                    },
                )
            except Exception as e:
                logger.warning("Episodic log failed for task %s: %s", task.id, e)
