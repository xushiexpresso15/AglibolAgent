"""Planner agent for decomposing user goals into an executable DAG task plan."""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

from aglibol.agents.base import BaseAgent
from aglibol.core.events import AgentEvent, EventBus
from aglibol.core.scheduler import ModelScheduler
from aglibol.core.types import AgentState, ChatMessage, TaskItem, TaskPlan, TaskStatus
from aglibol.ollama.client import OllamaClient
from aglibol.tools.registry import ToolRegistry

logger = logging.getLogger("aglibol.agents.planner")


class PlannerAgent(BaseAgent):
    """Decomposes a complex objective into sequential or parallel actionable steps."""

    name = "planner"
    role = "planner"
    default_model = "qwen2.5:7b"

    system_prompt = """You are an expert Chief Software Architect and Planning Agent.
Your job is to analyze the user's objective and break it down into a clear, minimal, and concrete execution plan.
You must output a single valid JSON object representing the plan, without any markdown fences or extraneous conversation.

Operational Guidelines:
1. Break down the user objective into sequential, modular tasks.
2. assigned_agent MUST be one of: "coder", "reviewer", "writer". Do NOT use non-existent roles (such as "content_writer", "qa_engineer", or "tester").
3. Each task description MUST specify exact output file paths to create or modify.
4. Each task MUST have concrete, verifiable requirements.

Example input:
Objective: Build a CLI calculator with add and subtract functions in Python.

Example output:
{
  "summary": "Implement a modular Python CLI calculator with add/subtract operations and unit tests.",
  "tasks": [
    {
      "id": "task_1",
      "title": "Create calculator core functions",
      "description": "Write src/calculator.py containing add(a, b) and subtract(a, b) functions with type annotations and docstrings.",
      "assigned_agent": "coder",
      "dependencies": []
    },
    {
      "id": "task_2",
      "title": "Add unit tests for calculator",
      "description": "Write tests/test_calculator.py covering valid operations and edge cases.",
      "assigned_agent": "coder",
      "dependencies": ["task_1"]
    }
  ]
}

The JSON schema must strictly match the structure above with "summary" and "tasks".
"""

    async def execute(
        self,
        state: AgentState,
        client: OllamaClient,
        scheduler: ModelScheduler,
        tool_registry: ToolRegistry,
        event_bus: EventBus | None = None,
    ) -> AgentState:
        prompt = (
            f"Objective:\n{state.user_goal}\n\n"
            "Please decompose this objective into an executable task plan in the requested JSON format."
        )
        messages = [ChatMessage(role="user", content=prompt)]

        # Pass TaskPlan JSON schema for Ollama GBNF grammar-guided structured decoding
        plan_schema = {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "title": {"type": "string"},
                            "description": {"type": "string"},
                            "assigned_agent": {
                                "type": "string",
                                "enum": ["coder", "reviewer", "writer"],
                            },
                            "dependencies": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["id", "title", "description", "assigned_agent"],
                    },
                },
            },
            "required": ["summary", "tasks"],
        }

        resp = await self._run_agent_loop(
            messages=messages,
            client=client,
            scheduler=scheduler,
            tool_registry=tool_registry,
            event_bus=event_bus,
            format=plan_schema,
        )

        plan = self._parse_plan(resp.content, state.user_goal)
        self._validate_dependencies(plan)

        state.task_plan = plan
        # Note: Do not hardcode state.active_agent = "coder" here;
        # the WorkflowEngine determines transition edges.

        if event_bus:
            await event_bus.emit(
                AgentEvent(
                    event_type="PLAN_CREATED",
                    session_id=state.session_id,
                    step=state.current_step,
                    agent=self.name,
                    data={"task_count": len(plan.tasks), "summary": plan.summary},
                )
            )

        return state

    def _parse_plan(self, text: str, goal: str) -> TaskPlan:
        """Robustly extract JSON plan from raw LLM output with multi-tier fallback repair."""
        plan_id = f"plan_{uuid.uuid4().hex[:8]}"

        # Tier 1: Direct JSON parsing
        data = self._try_parse_json(text.strip())
        if data and isinstance(data, dict) and "tasks" in data:
            return self._build_plan(plan_id, goal, data)

        # Tier 2: Extract content from markdown code fences ```json ... ```
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if fence_match:
            data = self._try_parse_json(fence_match.group(1).strip())
            if data and isinstance(data, dict) and "tasks" in data:
                return self._build_plan(plan_id, goal, data)

        # Tier 3: Extract first outermost curly bracket block { ... }
        bracket_match = re.search(r"(\{[\s\S]*\})", text)
        if bracket_match:
            data = self._try_parse_json(bracket_match.group(1).strip())
            if data and isinstance(data, dict) and "tasks" in data:
                return self._build_plan(plan_id, goal, data)

        # Tier 4: Heuristic AST repair (fix trailing commas, unquoted keys, single quotes)
        repaired = self._heuristic_json_repair(text)
        data = self._try_parse_json(repaired)
        if data and isinstance(data, dict) and "tasks" in data:
            return self._build_plan(plan_id, goal, data)

        # Tier 5: Safe fallback single-step plan
        logger.warning(
            "PlannerAgent failed to parse structured JSON plan from model response (%d chars). Falling back to direct execution.",
            len(text),
        )
        fallback_task = TaskItem(
            id="task_1",
            title="Implement solution",
            description=goal,
            assigned_agent="coder",
            dependencies=[],
            status=TaskStatus.PENDING,
        )
        return TaskPlan(
            plan_id=plan_id,
            goal=goal,
            summary="Direct single-step execution plan",
            tasks=[fallback_task],
        )

    @staticmethod
    def _try_parse_json(text: str) -> dict[str, Any] | None:
        """Attempt json.loads, optionally falling back to json_repair if installed."""
        try:
            return json.loads(text)
        except Exception:
            pass

        try:
            import json_repair  # type: ignore[import-not-found]

            repaired = json_repair.repair_json(text, return_objects=True)
            if isinstance(repaired, dict):
                return repaired
        except ImportError:
            pass
        except Exception:
            pass

        return None

    @staticmethod
    def _heuristic_json_repair(text: str) -> str:
        """Best-effort heuristic repair for common LLM JSON errors."""
        # Find outermost brackets
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return text
        s = text[start : end + 1]

        # Fix single quotes to double quotes for keys and string values
        s = re.sub(r"(?<=[{\[,:])\s*'([^'\\]*(?:\\.[^'\\]*)*)'(?=\s*[:,\]}])", r'"\1"', s)
        s = re.sub(r"'([a-zA-Z0-9_\-]+)'\s*:", r'"\1":', s)

        # Replace Python booleans/null
        s = re.sub(r"\bTrue\b", "true", s)
        s = re.sub(r"\bFalse\b", "false", s)
        s = re.sub(r"\bNone\b", "null", s)

        # Remove trailing commas before closing brackets/braces
        s = re.sub(r",\s*([\]}])", r"\1", s)
        return s

    @staticmethod
    def _build_plan(plan_id: str, goal: str, data: dict[str, Any]) -> TaskPlan:
        """Construct TaskPlan from validated JSON dict with role normalization."""
        valid_agents = {"coder", "reviewer", "writer"}
        tasks_data = data.get("tasks", [])
        tasks: list[TaskItem] = []
        for i, t in enumerate(tasks_data):
            agent = t.get("assigned_agent", "coder")
            if agent not in valid_agents:
                agent = "coder"  # Safe normalization fallback
            tasks.append(
                TaskItem(
                    id=t.get("id", f"task_{i + 1}"),
                    title=t.get("title", f"Step {i + 1}"),
                    description=t.get("description", ""),
                    assigned_agent=agent,
                    dependencies=t.get("dependencies", []),
                    status=TaskStatus.PENDING,
                )
            )
        return TaskPlan(
            plan_id=plan_id,
            goal=goal,
            summary=data.get("summary", ""),
            tasks=tasks,
        )

    @staticmethod
    def _validate_dependencies(plan: TaskPlan) -> None:
        """Validate that task dependencies form a valid DAG without self-loops or missing IDs."""
        task_ids = {t.id for t in plan.tasks}
        for task in plan.tasks:
            # Filter out self-dependencies or non-existent dependencies
            task.dependencies = [
                dep_id for dep_id in task.dependencies if dep_id in task_ids and dep_id != task.id
            ]
