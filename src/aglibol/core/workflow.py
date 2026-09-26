"""Workflow engine: Graph-based state machine for multi-agent coordination."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from aglibol.agents.base import BaseAgent
from aglibol.core.events import AgentEvent, EventBus
from aglibol.core.scheduler import ModelScheduler
from aglibol.core.types import AgentState
from aglibol.memory.episodic import EpisodicMemory
from aglibol.memory.working import WorkingMemory
from aglibol.ollama.client import OllamaClient
from aglibol.storage.artifact_store import ArtifactStore
from aglibol.storage.checkpoint import CheckpointStore
from aglibol.tools.registry import ToolRegistry

logger = logging.getLogger("aglibol.workflow")


class WorkflowEdge:
    """Directed connection between two agent nodes with an optional routing condition."""

    def __init__(
        self,
        from_node: str,
        to_node: str,
        condition: Callable[[AgentState], bool] | str | None = None,
        max_cycles: int = 3,
    ) -> None:
        self.from_node = from_node
        self.to_node = to_node
        self.condition = condition
        self.max_cycles = max_cycles

    @property
    def edge_key(self) -> str:
        """Unique key for per-edge cycle tracking."""
        return f"{self.from_node}->{self.to_node}"


class WorkflowEngine:
    """Graph-based workflow orchestrator executing agent steps with state checkpointing."""

    def __init__(
        self,
        client: OllamaClient,
        scheduler: ModelScheduler,
        tool_registry: ToolRegistry,
        checkpoint_store: CheckpointStore | None = None,
        artifact_store: ArtifactStore | None = None,
        event_bus: EventBus | None = None,
        working_memory: WorkingMemory | None = None,
        episodic_memory: EpisodicMemory | None = None,
        max_total_steps: int = 20,
    ) -> None:
        self.client = client
        self.scheduler = scheduler
        self.tool_registry = tool_registry
        self.checkpoint_store = checkpoint_store
        self.artifact_store = artifact_store
        self.event_bus = event_bus
        self.working_memory = working_memory
        self.episodic_memory = episodic_memory
        self.max_total_steps = max_total_steps

        self.nodes: dict[str, BaseAgent] = {}
        self.edges: list[WorkflowEdge] = []
        self.entry_point: str = "planner"

    def add_node(self, name: str, agent: BaseAgent) -> None:
        """Register an agent node in the workflow graph."""
        self.nodes[name] = agent

    def add_edge(
        self,
        from_node: str,
        to_node: str,
        condition: Callable[[AgentState], bool] | str | None = None,
        max_cycles: int = 3,
    ) -> None:
        """Add a directed transition between two nodes."""
        self.edges.append(
            WorkflowEdge(
                from_node=from_node, to_node=to_node, condition=condition, max_cycles=max_cycles
            )
        )

    def set_entry_point(self, name: str) -> None:
        """Set starting agent node."""
        self.entry_point = name

    async def run(self, initial_state: AgentState) -> AgentState:
        """Execute the workflow graph starting from entry_point until completion.

        Changes from original:
        - Distinguishes success vs failure completion via state.status
        - Per-edge cycle counters instead of single shared iteration_count
        - try/finally ensures scheduler model release even on exceptions
        - Proper failure marking when review exceeds max_cycles
        """
        state = initial_state
        state.status = "running"
        state.is_completed = False

        if state.current_step > 0 and state.active_agent:
            next_node = self._resolve_next_node(state.active_agent, state)
            if next_node and next_node.lower() == "end":
                state.is_completed = True
                state.status = "success"
                return state
            current_node_name = next_node or state.active_agent
        else:
            current_node_name = state.active_agent or self.entry_point

        if self.event_bus:
            await self.event_bus.emit(
                AgentEvent(
                    event_type="WORKFLOW_STARTED",
                    session_id=state.session_id,
                    data={"goal": state.user_goal, "entry_point": current_node_name},
                )
            )

        step_counter = state.current_step
        max_target_step = step_counter + self.max_total_steps

        try:
            while step_counter < max_target_step:
                step_counter += 1
                state.current_step = step_counter

                if current_node_name not in self.nodes:
                    state.error = f"Node '{current_node_name}' not found in workflow graph."
                    state.status = "failed"
                    break

                agent = self.nodes[current_node_name]
                state.active_agent = current_node_name

                if self.event_bus:
                    await self.event_bus.emit(
                        AgentEvent(
                            event_type="STEP_STARTED",
                            session_id=state.session_id,
                            step=step_counter,
                            agent=current_node_name,
                        )
                    )

                # Execute agent step
                try:
                    state = await agent.execute(
                        state=state,
                        client=self.client,
                        scheduler=self.scheduler,
                        tool_registry=self.tool_registry,
                        event_bus=self.event_bus,
                    )
                except Exception as e:
                    state.error = f"Error in agent '{current_node_name}': {str(e)}"
                    state.status = "failed"
                    logger.error(
                        "Agent '%s' raised exception: %s", current_node_name, e, exc_info=True
                    )
                    if self.event_bus:
                        await self.event_bus.emit(
                            AgentEvent(
                                event_type="STEP_ERROR",
                                session_id=state.session_id,
                                step=step_counter,
                                agent=current_node_name,
                                data={"error": str(e)},
                            )
                        )
                    break

                # Persist checkpoint to disk
                try:
                    if self.checkpoint_store:
                        self.checkpoint_store.save_checkpoint(state)
                except Exception as e:
                    logger.warning("Failed to save checkpoint at step %d: %s", step_counter, e)

                # Persist any generated artifacts to disk
                try:
                    if self.artifact_store and state.artifacts:
                        for fname, content in state.artifacts.items():
                            self.artifact_store.save_artifact(fname, content)
                except Exception as e:
                    logger.warning("Failed to save artifacts at step %d: %s", step_counter, e)

                # Update working memory and episodic log
                if self.working_memory:
                    self.working_memory.scratchpad = dict(state.scratchpad)
                if self.episodic_memory:
                    from aglibol.core.types import ChatMessage

                    step_summary = f"Step {step_counter} by {current_node_name}: artifacts={list(state.artifacts.keys())}"
                    self.episodic_memory.log_turn(
                        agent=current_node_name,
                        step=step_counter,
                        message=ChatMessage(role="assistant", content=step_summary),
                        extra={"status": state.status, "review_status": state.review_status},
                    )

                if self.event_bus:
                    await self.event_bus.emit(
                        AgentEvent(
                            event_type="STEP_FINISHED",
                            session_id=state.session_id,
                            step=step_counter,
                            agent=current_node_name,
                        )
                    )

                # Check if workflow reached terminal state
                if state.is_completed:
                    if state.status == "running":
                        state.status = "success"
                    break

                # Resolve next node via transitions
                next_node = self._resolve_next_node(current_node_name, state)
                if next_node is None:
                    # No valid transition — check if it's a cycle limit exceeded
                    if state.review_status == "rejected":
                        state.status = "failed"
                        state.error = (
                            "Max review correction cycles exceeded. The code could not pass review."
                        )
                        state.is_completed = True
                        logger.warning(
                            "Workflow failed: max review cycles exceeded for session %s",
                            state.session_id,
                        )
                    else:
                        state.status = "success"
                        state.is_completed = True
                    break
                elif next_node.lower() == "end":
                    state.is_completed = True
                    state.status = "success"
                    break

                current_node_name = next_node

            else:
                # Exhausted max_total_steps
                state.status = "failed"
                state.error = f"Workflow exceeded maximum steps ({self.max_total_steps})"
                state.is_completed = True

        finally:
            # Always release model to free GPU VRAM
            try:
                await self.scheduler.release_current_model()
            except Exception as e:
                logger.warning("Failed to release model on workflow end: %s", e)

        if self.event_bus:
            await self.event_bus.emit(
                AgentEvent(
                    event_type="WORKFLOW_COMPLETED",
                    session_id=state.session_id,
                    step=state.current_step,
                    data={
                        "status": state.status,
                        "is_completed": state.is_completed,
                        "artifacts_count": len(state.artifacts),
                    },
                )
            )

        return state

    def _resolve_next_node(self, current_node: str, state: AgentState) -> str | None:
        """Evaluate edge conditions from current node with per-edge cycle tracking."""
        for edge in self.edges:
            if edge.from_node != current_node:
                continue

            cond = edge.condition
            if cond is None or cond == "always":
                return edge.to_node
            elif callable(cond):
                if cond(state):
                    return edge.to_node
            elif cond == "review_approved" and state.review_status == "approved":
                return edge.to_node
            elif cond == "review_rejected" and state.review_status == "rejected":
                # Use per-edge cycle counter
                edge_key = edge.edge_key
                current_cycles = state.edge_cycles.get(edge_key, 0)
                if current_cycles < edge.max_cycles:
                    state.edge_cycles[edge_key] = current_cycles + 1

                    # C9 Self-healing: On the last allowed attempt, rollback artifacts and bump temperature
                    if state.edge_cycles[edge_key] == edge.max_cycles:
                        state.scratchpad["temperature_bump"] = 0.7
                        if self.checkpoint_store:
                            try:
                                ckpts = self.checkpoint_store.list_checkpoints(state.session_id)
                                if len(ckpts) >= 2:
                                    prev_ckpt = self.checkpoint_store.get_checkpoint_by_id(
                                        ckpts[-2]["checkpoint_id"]
                                    )
                                    if prev_ckpt and prev_ckpt.artifacts:
                                        state.artifacts = dict(prev_ckpt.artifacts)
                            except Exception as e:
                                logger.warning("Failed to rollback checkpoint on retry: %s", e)

                        notice = (
                            "\n\n[SYSTEM RECOVERY ALERT]: Final attempt before workflow termination. "
                            "Artifacts rolled back to prior checkpoint baseline. Temperature escalated to 0.7. "
                            "You must take a fresh approach and strictly fix all reported defects without regressions."
                        )
                        state.review_feedback = (state.review_feedback or "") + notice

                    return edge.to_node
                # else: cycle limit exceeded, don't follow this edge

        return None

    @classmethod
    def build_from_yaml(
        cls,
        yaml_path: str | Path,
        client: OllamaClient,
        scheduler: ModelScheduler,
        tool_registry: ToolRegistry,
        checkpoint_store: CheckpointStore | None = None,
        artifact_store: ArtifactStore | None = None,
        event_bus: EventBus | None = None,
        working_memory: WorkingMemory | None = None,
        episodic_memory: EpisodicMemory | None = None,
        model_overrides: dict[str, str] | None = None,
        max_total_steps: int = 20,
    ) -> WorkflowEngine:
        """Construct a WorkflowEngine by parsing a YAML workflow definition with ${...} variable interpolation."""
        import yaml

        from aglibol.agents.registry import AgentRegistry

        p = Path(yaml_path)
        candidates = [
            p,
            Path(__file__).resolve().parent.parent / "config" / "workflows" / p.name,
            Path(__file__).resolve().parent.parent / "config" / p.name,
            Path(__file__).resolve().parent.parent.parent.parent / "config" / "workflows" / p.name,
            Path.cwd() / "config" / "workflows" / p.name,
            Path.cwd() / p,
        ]
        chosen_path: Path | None = None
        for cand in candidates:
            if cand.is_file():
                chosen_path = cand
                break

        engine = cls(
            client=client,
            scheduler=scheduler,
            tool_registry=tool_registry,
            checkpoint_store=checkpoint_store,
            artifact_store=artifact_store,
            event_bus=event_bus,
            working_memory=working_memory,
            episodic_memory=episodic_memory,
            max_total_steps=max_total_steps,
        )

        overrides = model_overrides or {}

        if not chosen_path:
            logger.warning(
                "Workflow YAML %s does not exist; using default nodes and edges.", yaml_path
            )
            cls._populate_default_nodes_and_edges(engine, overrides)
            return engine

        raw_text = chosen_path.read_text(encoding="utf-8")

        # Variable interpolation for ${models.<role>} or ${role}
        for role, model in overrides.items():
            raw_text = raw_text.replace(f"${{models.{role}}}", model)
            raw_text = raw_text.replace(f"${{{role}}}", model)

        try:
            data = yaml.safe_load(raw_text) or {}
            wf = data.get("workflow", {})
            nodes = wf.get("nodes", {})
            edges = wf.get("edges", [])

            if not nodes:
                cls._populate_default_nodes_and_edges(engine, overrides)
                return engine

            registry = AgentRegistry()
            for node_name, node_cfg in nodes.items():
                agent_type = node_cfg.get("agent", node_name)
                agent_cls = registry.get(agent_type)
                if not agent_cls:
                    logger.warning(
                        "Agent type '%s' not registered, skipping node '%s'", agent_type, node_name
                    )
                    continue

                model_name = node_cfg.get("model")
                if model_name and model_name.startswith("${"):
                    model_name = overrides.get(agent_type, "qwen2.5:7b")

                agent = agent_cls(model_name=model_name)
                engine.add_node(node_name, agent)
                if node_cfg.get("entry_point"):
                    engine.set_entry_point(node_name)

            for edge in edges:
                engine.add_edge(
                    from_node=edge["from"],
                    to_node=edge["to"],
                    condition=edge.get("condition", "always"),
                    max_cycles=edge.get("max_cycles", 3),
                )

        except Exception as e:
            logger.warning(
                "Failed to parse workflow YAML %s: %s; falling back to default graph.", yaml_path, e
            )
            cls._populate_default_nodes_and_edges(engine, overrides)

        return engine

    @staticmethod
    def _populate_default_nodes_and_edges(
        engine: WorkflowEngine, model_overrides: dict[str, str] | None = None
    ) -> None:
        """Populate standard planner -> coder -> reviewer graph as fallback."""
        from aglibol.agents.coder import CoderAgent
        from aglibol.agents.planner import PlannerAgent
        from aglibol.agents.reviewer import ReviewerAgent

        overrides = model_overrides or {}
        p_model = overrides.get("planner", "qwen2.5:7b")
        c_model = overrides.get("coder", "qwen2.5-coder:7b")
        r_model = overrides.get("reviewer", "qwen2.5:7b")

        engine.add_node("planner", PlannerAgent(model_name=p_model))
        engine.add_node("coder", CoderAgent(model_name=c_model))
        engine.add_node("reviewer", ReviewerAgent(model_name=r_model))

        engine.add_edge("planner", "coder", condition="always")
        engine.add_edge("coder", "reviewer", condition="always")
        engine.add_edge("reviewer", "coder", condition="review_rejected", max_cycles=3)
        engine.add_edge("reviewer", "end", condition="review_approved")
        engine.set_entry_point("planner")
