"""Reviewer agent for inspecting artifacts, validating correctness, and providing critique."""

from __future__ import annotations

import ast
import json
import logging
import re

from aglibol.agents.base import BaseAgent
from aglibol.core.events import AgentEvent, EventBus
from aglibol.core.scheduler import ModelScheduler
from aglibol.core.types import AgentState, ChatMessage
from aglibol.ollama.client import OllamaClient
from aglibol.tools.registry import ToolRegistry

logger = logging.getLogger("aglibol.agents.reviewer")


class ReviewerAgent(BaseAgent):
    """Reviews code artifacts against quality, security, and requirement criteria with deterministic linter gates."""

    name = "reviewer"
    role = "reviewer"
    default_model = "qwen2.5:7b"

    system_prompt = """You are a Principal Software Quality and Security Reviewer.
Your job is to critically review the code or solution produced by the Coder agent.

Criteria:
1. Does it completely solve the user's objective?
2. Are there syntax errors, missing imports, or unhandled exceptions?
3. Is it clean, readable, and idiomatic?

Decision Contract:
Output a single valid JSON object strictly matching:
{
  "decision": "approved" | "rejected",
  "reasons": ["Detailed point 1", "Detailed point 2"],
  "feedback": "Actionable instructions for the Coder to fix any issues, or confirmation of success."
}
"""

    def __init__(self, model_name: str | None = None) -> None:
        # Give reviewer tools to inspect files and project directory if needed
        super().__init__(model_name=model_name, tools=["read_file", "list_dir"])

    async def execute(
        self,
        state: AgentState,
        client: OllamaClient,
        scheduler: ModelScheduler,
        tool_registry: ToolRegistry,
        event_bus: EventBus | None = None,
    ) -> AgentState:
        # --- Gate 1: Deterministic Syntax & AST Check ---
        new_mods: dict[str, str] = state.scratchpad.get("current_task_modifications", {})
        target_syntax_check = new_mods if new_mods else state.artifacts

        if not target_syntax_check and not state.artifacts:
            error_feedback = (
                "DECISION: REJECTED\n\n"
                "No code deliverables or files were produced by Coder.\n"
                "Please use the 'write_file' tool to write and save your implementation directly to disk."
            )
            state.review_status = "rejected"
            state.review_feedback = error_feedback
            state.iteration_count += 1
            if event_bus:
                await event_bus.emit(
                    AgentEvent(
                        event_type="REVIEW_COMPLETED",
                        session_id=state.session_id,
                        step=state.current_step,
                        agent=self.name,
                        data={
                            "decision": "rejected",
                            "gate": "empty_deliverables",
                            "iteration": state.iteration_count,
                            "errors": ["No code artifacts produced"],
                        },
                    )
                )
            return state

        syntax_errors = self._run_ast_checks(target_syntax_check)
        if syntax_errors:
            # Deterministic failure: immediately reject without wasting LLM inference
            error_feedback = (
                "DECISION: REJECTED\n\n"
                "Deterministic Syntax Gate Failed:\n"
                + "\n".join(f"- {err}" for err in syntax_errors)
                + "\nPlease fix these syntax errors before resubmitting."
            )
            state.review_status = "rejected"
            state.review_feedback = error_feedback
            state.iteration_count += 1

            # Purge corrupted artifacts from state and disk so they do not poison retry cycles
            from pathlib import Path

            for err in syntax_errors:
                bad_file = err.split(":", 1)[0].strip()
                state.artifacts.pop(bad_file, None)
                if state.workspace_dir:
                    disk_p = Path(state.workspace_dir) / bad_file
                    if disk_p.exists():
                        try:
                            disk_p.unlink()
                        except Exception:
                            pass

            if event_bus:
                await event_bus.emit(
                    AgentEvent(
                        event_type="REVIEW_COMPLETED",
                        session_id=state.session_id,
                        step=state.current_step,
                        agent=self.name,
                        data={
                            "decision": "rejected",
                            "gate": "deterministic_ast",
                            "iteration": state.iteration_count,
                            "errors": syntax_errors,
                        },
                    )
                )
            return state

        # --- Gate 2: LLM Semantic & Quality Review ---
        if new_mods:
            mod_summary = "\n\n".join(
                [
                    f"--- Modified/Created Deliverable: {name} ---\n{content}"
                    for name, content in new_mods.items()
                ]
            )
            other_files = [f for f in state.artifacts if f not in new_mods]
            bg_summary = (
                f"\n\nExisting Background Project Files in Workspace: {', '.join(other_files)}"
                if other_files
                else ""
            )
            artifacts_summary = f"Deliverables Produced in Current Step:\n{mod_summary}{bg_summary}"
        elif state.artifacts:
            artifacts_summary = "\n\n".join(
                [
                    f"--- Artifact: {name} ---\n{content}"
                    for name, content in state.artifacts.items()
                ]
            )
        else:
            artifacts_summary = "(No file artifacts generated in memory. Inspect disk workspace if tools were used.)"

        target_criteria = f"Goal: {state.user_goal}"
        if state.current_task_id and state.task_plan:
            for t in state.task_plan.tasks:
                if t.id == state.current_task_id:
                    target_criteria = (
                        f"Current Task Being Reviewed: [{t.id}] {t.title}\n"
                        f"Task Requirements: {t.description}\n"
                        f"Overall Project Goal: {state.user_goal}\n"
                        f"(Note: Evaluate whether the deliverables satisfy this specific task's requirements.)"
                    )
                    break

        prompt = (
            f"{target_criteria}\n\n"
            f"{artifacts_summary}\n\n"
            "Please review this solution. Check whether the deliverables meet the requirements and integrate properly with existing workspace files.\n"
            "Output your decision strictly as a JSON object with 'decision' ('approved' or 'rejected'), 'reasons', and 'feedback'."
        )
        messages = [ChatMessage(role="user", content=prompt)]

        review_schema = {
            "type": "object",
            "properties": {
                "decision": {"type": "string", "enum": ["approved", "rejected"]},
                "reasons": {"type": "array", "items": {"type": "string"}},
                "feedback": {"type": "string"},
            },
            "required": ["decision"],
        }

        resp = await self._run_agent_loop(
            messages=messages,
            client=client,
            scheduler=scheduler,
            tool_registry=tool_registry,
            event_bus=event_bus,
            format=review_schema,
        )

        content = resp.content.strip()
        decision = self._parse_decision(content)

        # Format clean review feedback for downstream Coder
        feedback_text = content
        try:
            cleaned = content
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.DOTALL).strip()
            data = json.loads(cleaned)
            if isinstance(data, dict) and "decision" in data:
                reasons = data.get("reasons", [])
                reasons_str = "\n".join(f"- {r}" for r in reasons) if reasons else ""
                fb = data.get("feedback", "")
                parts = [f"DECISION: {decision.upper()}"]
                if reasons_str:
                    parts.append(f"Reasons:\n{reasons_str}")
                if fb:
                    parts.append(f"Feedback:\n{fb}")
                feedback_text = "\n\n".join(parts)
        except Exception:
            pass

        if decision == "approved":
            state.review_status = "approved"
            state.review_feedback = feedback_text
            state.is_completed = True
        else:
            state.review_status = "rejected"
            state.review_feedback = feedback_text
            state.iteration_count += 1
            # Note: Do not override cycle limit or mark completed here;
            # WorkflowEngine evaluates edge.max_cycles and sets state.status="failed" if exceeded.

        if event_bus:
            await event_bus.emit(
                AgentEvent(
                    event_type="REVIEW_COMPLETED",
                    session_id=state.session_id,
                    step=state.current_step,
                    agent=self.name,
                    data={"decision": state.review_status, "iteration": state.iteration_count},
                )
            )

        return state

    @staticmethod
    def _run_ast_checks(artifacts: dict[str, str]) -> list[str]:
        """Statically validate Python syntax and concurrency safety using ast.parse before invoking LLM."""
        errors: list[str] = []
        for name, code in artifacts.items():
            if name.endswith(".py") or name.endswith(".pyw"):
                try:
                    tree = ast.parse(code, filename=name)
                    # Check for potential self-deadlocks on non-reentrant locks
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            non_reentrant = set()
                            for stmt in node.body:
                                if isinstance(stmt, ast.FunctionDef) and stmt.name == "__init__":
                                    for sub in ast.walk(stmt):
                                        if isinstance(sub, ast.Assign):
                                            for target in sub.targets:
                                                if (
                                                    isinstance(target, ast.Attribute)
                                                    and isinstance(target.value, ast.Name)
                                                    and target.value.id == "self"
                                                ):
                                                    if isinstance(sub.value, ast.Call):
                                                        func = sub.value.func
                                                        if (
                                                            isinstance(func, ast.Name)
                                                            and func.id == "Lock"
                                                        ) or (
                                                            isinstance(func, ast.Attribute)
                                                            and func.attr == "Lock"
                                                        ):
                                                            non_reentrant.add(target.attr)
                            if not non_reentrant:
                                continue
                            methods_acquiring: dict[str, set[str]] = {}
                            for stmt in node.body:
                                if isinstance(stmt, ast.FunctionDef):
                                    acquired = set()
                                    for sub in ast.walk(stmt):
                                        if isinstance(sub, ast.With):
                                            for item in sub.items:
                                                expr = item.context_expr
                                                if (
                                                    isinstance(expr, ast.Attribute)
                                                    and isinstance(expr.value, ast.Name)
                                                    and expr.value.id == "self"
                                                ):
                                                    if expr.attr in non_reentrant:
                                                        acquired.add(expr.attr)
                                    methods_acquiring[stmt.name] = acquired
                            for stmt in node.body:
                                if isinstance(stmt, ast.FunctionDef):
                                    for sub in ast.walk(stmt):
                                        if isinstance(sub, ast.With):
                                            held = set()
                                            for item in sub.items:
                                                expr = item.context_expr
                                                if (
                                                    isinstance(expr, ast.Attribute)
                                                    and isinstance(expr.value, ast.Name)
                                                    and expr.value.id == "self"
                                                ):
                                                    if expr.attr in non_reentrant:
                                                        held.add(expr.attr)
                                            if held:
                                                for call in ast.walk(sub):
                                                    if isinstance(call, ast.Call) and isinstance(
                                                        call.func, ast.Attribute
                                                    ):
                                                        if (
                                                            isinstance(call.func.value, ast.Name)
                                                            and call.func.value.id == "self"
                                                        ):
                                                            called_m = call.func.attr
                                                            if called_m in methods_acquiring:
                                                                overlap = held.intersection(
                                                                    methods_acquiring[called_m]
                                                                )
                                                                if overlap:
                                                                    errors.append(
                                                                        f"{name}: DeadlockError: Class '{node.name}' method '{stmt.name}' holds non-reentrant lock(s) {sorted(overlap)} while calling 'self.{called_m}()', which also acquires the same lock. Use threading.RLock() or avoid nested lock acquisition."
                                                                    )
                except SyntaxError as e:
                    errors.append(f"{name}:{e.lineno}:{e.offset}: SyntaxError: {e.msg}")
                except Exception as e:
                    errors.append(f"{name}: ParseError: {str(e)}")
        return errors

    @staticmethod
    def _parse_decision(content: str) -> str:
        """
        Strictly parse review decision from structured JSON or text fallback.
        Matches only when DECISION: APPROVED or DECISION: REJECTED appears as an intentional statement,
        avoiding false positives from conversational quotes.
        """
        # Tier 1: Parse structured JSON
        try:
            cleaned = content.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.DOTALL).strip()
            data = json.loads(cleaned)
            if isinstance(data, dict) and "decision" in data:
                dec = str(data["decision"]).strip().lower()
                if "approved" in dec and "rejected" not in dec:
                    return "approved"
                return "rejected"
        except Exception:
            pass

        # Tier 2: Look for DECISION at beginning of lines or start of text
        match = re.search(
            r"^\s*DECISION:\s*(APPROVED|REJECTED)", content, re.MULTILINE | re.IGNORECASE
        )
        if match:
            return match.group(1).lower()

        # Tier 3: Fallback check: look for unambiguous standalone approval
        if re.search(r"\bDECISION:\s*APPROVED\b", content, re.IGNORECASE) and not re.search(
            r"\bDECISION:\s*REJECTED\b", content, re.IGNORECASE
        ):
            return "approved"

        return "rejected"
