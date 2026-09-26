"""Coder agent for generating implementations, scripts, and code artifacts."""

from __future__ import annotations

import ast
import logging
import os
import re
from pathlib import Path
from typing import Any

from aglibol.agents.base import BaseAgent
from aglibol.core.events import AgentEvent, EventBus
from aglibol.core.scheduler import ModelScheduler
from aglibol.core.types import AgentState, ChatMessage, TaskItem, TaskStatus
from aglibol.ollama.client import OllamaClient
from aglibol.tools.registry import ToolRegistry

logger = logging.getLogger("aglibol.agents.coder")


class CoderAgent(BaseAgent):
    """Generates code solutions, executes builds/tests, and writes code artifacts to disk."""

    name = "coder"
    role = "coder"
    default_model = "qwen2.5-coder:7b"

    system_prompt = """You are an elite Senior Staff Software Engineer and the Coder specialist of Aglibol Agent.
About Aglibol Agent:
Aglibol Agent is an open-source, local-first multi-agent developer system designed to run on consumer GPUs via Ollama. It orchestrates a specialized agent team (Planner for architecture, Coder for implementation, Reviewer for verification, Writer for documentation), with disk-first persistent memory (Brain, SQLite checkpoints, JSONL episodic logs), sandboxed workspace tools, and a rich interactive terminal interface.

Your job is to implement complete, robust, well-documented, and production-ready code based on the task description.

Operational Guidelines:
1. Always analyze the requirements and structure your plan inside <thinking>...</thinking> tags first, before calling tools or outputting code.
2. Provide COMPLETE implementations (never omit logic, write '// TODO', or leave placeholder stubs).
3. Persist your code directly to disk using the available tools (`write_file` for new/short files, `search_replace` for editing existing files).
4. If the task is to build a new feature, web page, or script, create the new files directly using `write_file`. Only read existing files if you specifically need to modify or test them. Do NOT repeatedly inspect unrelated files.
5. CRITICAL ENVIRONMENT RULE: When running on Windows, do NOT use Unix-specific commands like `ls -la`, `cat`, `touch`, or `grep`.
   - To inspect directory contents, ALWAYS use `list_dir`.
   - To inspect files, ALWAYS use `read_file`.
   - To create or overwrite files, ALWAYS use `write_file`.
6. If writing code directly in your response, use standard markdown code fences with a relative file path indicator:
   ```python
   # File: src/module.py
   <complete code>
   ```
7. Ensure all code compiles cleanly and passes syntax checks. Do NOT output conversational prose as a substitute for writing code files.
8. Thread Safety & Concurrency: When implementing thread-safe classes where methods may call each other while synchronized, always use 'threading.RLock()' rather than non-reentrant 'threading.Lock()' to avoid self-deadlocks, or release the lock before calling internal helper methods.
"""

    max_tool_iterations = 15

    def __init__(self, model_name: str | None = None) -> None:
        super().__init__(
            model_name=model_name,
            tools=["write_file", "read_file", "list_dir", "search_replace", "execute_shell"],
        )

    async def execute(
        self,
        state: AgentState,
        client: OllamaClient,
        scheduler: ModelScheduler,
        tool_registry: ToolRegistry,
        event_bus: EventBus | None = None,
        episodic_memory: Any | None = None,
    ) -> AgentState:
        # Determine tasks to execute
        tasks_to_run = self._get_pending_tasks(state)
        workspace = state.workspace_dir or os.getcwd()
        active_task: TaskItem | None = None

        existing_artifacts = list(state.artifacts.keys())
        existing_str = ""
        if existing_artifacts:
            existing_str = (
                f"Existing Workspace Files: {', '.join(existing_artifacts)}\n"
                f"NOTE: Only read or edit existing files if your task specifically requires updating or extending them. "
                f"If you are creating new deliverables (like a new website, script, or feature), immediately create and persist them using `write_file`.\n\n"
            )

        if state.current_task_id and tasks_to_run:
            active_task = tasks_to_run[0]
            prompt = (
                f"Overall Project Goal: {state.user_goal}\n"
                f"Active Workspace Directory: {workspace}\n\n"
                f"{existing_str}"
                f"CURRENT TASK TO IMPLEMENT:\n"
                f"Task ID: {active_task.id}\n"
                f"Title: {active_task.title}\n"
                f"Requirements: {active_task.description}\n\n"
                f"EXECUTION DIRECTIVES:\n"
                f"1. Focus strictly on completing THIS SPECIFIC TASK ({active_task.id}).\n"
                f"2. Create Deliverables: Create and persist all necessary implementation files directly to disk using `write_file`.\n"
                f"3. Do NOT inspect unrelated workspace files. If creating a new website, script, or application, immediately generate and save the required files (e.g. index.html, style.css).\n"
                f"4. Provide complete, production-ready deliverables without placeholder comments.\n"
            )
        elif not tasks_to_run:
            # Fallback to general user goal
            prompt = (
                f"Overall Objective: {state.user_goal}\n"
                f"Active Workspace Directory: {workspace}\n\n"
                f"{existing_str}"
                f"Goal: {state.user_goal}\n\n"
                f"EXECUTION DIRECTIVES:\n"
                f"1. Create and persist all necessary deliverables to disk using `write_file`.\n"
                f"2. Do NOT get stuck inspecting unrelated files.\n"
            )
        else:
            task_descriptions = [
                f"Task [{t.id}]: {t.title}\nDetails: {t.description}" for t in tasks_to_run
            ]
            tasks_summary = "\n\n".join(task_descriptions)
            prompt = (
                f"Overall Objective: {state.user_goal}\n"
                f"Active Workspace Directory: {workspace}\n\n"
                f"{existing_str}"
                f"Tasks to Implement:\n{tasks_summary}\n\n"
                f"EXECUTION DIRECTIVES:\n"
                f"1. Implement tasks and persist deliverables directly using `write_file`.\n"
                f"2. Do NOT get stuck inspecting unrelated workspace files.\n"
            )

        review_history = state.scratchpad.get("review_history", [])
        if state.review_feedback and state.review_status == "rejected":
            if state.review_feedback not in review_history:
                review_history.append(state.review_feedback)
            state.scratchpad["review_history"] = review_history

            history_str = "\n---\n".join(
                f"[Cycle {i + 1}] {item}" for i, item in enumerate(review_history)
            )
            prompt += (
                f"\n\nCRITICAL - Previous Review Feedback History (fix all errors without regressions):\n"
                f"{history_str}\n"
                f"Ensure you address ALL feedback cycles above and do NOT re-introduce earlier bugs."
            )

        goal_and_task = (
            state.user_goal + " " + (active_task.description if active_task else "")
        ).lower()
        is_web_task = any(
            kw in goal_and_task
            for kw in (
                "\u7db2\u7ad9",
                "\u7f51\u7ad9",
                "\u7db2\u9801",
                "\u7f51\u9875",
                "website",
                "web page",
                "html",
                "landing page",
            )
        )
        if is_web_task:
            prompt += (
                "\n\nSPECIAL DIRECTIVE FOR WEB IMPLEMENTATION:\n"
                "You MUST create the website deliverables. Immediately invoke `write_file` with path='index.html' "
                "containing the complete HTML5 implementation (with modern embedded CSS/JS styling or separate style.css), "
                "or output the code inside ```html:index.html ... ```.\n"
                "Do NOT spend turns inspecting unrelated Python files in the workspace."
            )

        messages = [ChatMessage(role="user", content=prompt)]

        custom_options: dict[str, Any] = {}
        if "temperature_bump" in state.scratchpad:
            custom_options["temperature"] = float(state.scratchpad["temperature_bump"])

        resp = await self._run_agent_loop(
            messages=messages,
            client=client,
            scheduler=scheduler,
            tool_registry=tool_registry,
            event_bus=event_bus,
            custom_options=custom_options if custom_options else None,
            episodic_memory=episodic_memory,
        )

        newly_modified_files: dict[str, str] = {}

        # 1. Ingest any files written directly via tools during loop
        written_files = resp.raw_response.get("written_files", {})
        for filepath, content in written_files.items():
            if filepath and content:
                if "\n" not in content and "\\n" in content:
                    content = content.replace("\\n", "\n").replace("\\t", "\t")
                fname = Path(filepath).name
                state.artifacts[fname] = content
                if filepath != fname:
                    state.artifacts[filepath] = content
                newly_modified_files[fname] = content

        # 2. Extract code blocks from content
        extracted_blocks = self._extract_all_code_blocks(resp.content, context_hint=state.user_goal)

        # 3. Fallback: If no blocks in content, check thinking (reasoning models output code in thinking)
        if not extracted_blocks and resp.thinking:
            extracted_blocks = self._extract_all_code_blocks(
                resp.thinking, context_hint=state.user_goal
            )

        # 4. Fallback: Raw HTML or Python code without markdown fences in content or thinking
        if not extracted_blocks and not state.artifacts:
            for text_source in (resp.content, resp.thinking):
                candidate = text_source.strip()
                if not candidate:
                    continue
                if "<!DOCTYPE html" in candidate or "<html" in candidate:
                    m_html = re.search(
                        r"(<!DOCTYPE html[\s\S]*?</html>|<html[\s\S]*?</html>)",
                        candidate,
                        re.IGNORECASE,
                    )
                    if m_html:
                        extracted_blocks["index.html"] = m_html.group(1).strip()
                        break
                elif self._is_valid_python(candidate):
                    task_fname = "solution.py"
                    for t in tasks_to_run:
                        m = re.search(r"([\w\-./\\]+\.py)\b", t.description + " " + t.title)
                        if m:
                            task_fname = Path(m.group(1)).name
                            break
                    extracted_blocks[task_fname] = candidate
                    break

        for filename, code in extracted_blocks.items():
            if not filename:
                filename = f"solution_step_{state.current_step}_{len(state.artifacts) + 1}.py"
            # Ensure only syntactically valid Python code is saved under a .py name
            if filename.endswith(".py") and not self._is_valid_python(code):
                continue
            state.artifacts[filename] = code
            newly_modified_files[filename] = code

        state.scratchpad["current_task_modifications"] = newly_modified_files

        # 5. Ensure all newly produced artifacts are persisted to disk in workspace
        if state.workspace_dir:
            ws_path = Path(state.workspace_dir)
            for filename, code in list(state.artifacts.items()):
                try:
                    target_path = ws_path / filename
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    clean_code = code
                    if "\n" not in clean_code and "\\n" in clean_code:
                        clean_code = clean_code.replace("\\n", "\n").replace("\\t", "\t")
                    target_path.write_text(clean_code, encoding="utf-8")
                except Exception as e:
                    logger.warning(
                        "Could not auto-persist artifact %s to workspace: %s", filename, e
                    )

        # Update active task metadata; let Reviewer and Orchestrator determine final completion status
        has_artifacts = bool(state.artifacts) or bool(written_files) or bool(extracted_blocks)
        for t in tasks_to_run:
            if has_artifacts or (
                resp.is_done and resp.content.strip() and "reached max" not in resp.content
            ):
                t.status = TaskStatus.IN_PROGRESS
                t.result = resp.content[:300] if resp.content else "Artifacts created"
            else:
                t.status = TaskStatus.IN_PROGRESS
                t.error = "Agent reached iteration limit without generating code or writing files."

        # Note: Do not hardcode state.active_agent = "reviewer";
        # the WorkflowEngine manages the next node in the graph.

        if event_bus:
            await event_bus.emit(
                AgentEvent(
                    event_type="CODE_GENERATED",
                    session_id=state.session_id,
                    step=state.current_step,
                    agent=self.name,
                    data={
                        "length": len(resp.content),
                        "artifacts": list(state.artifacts.keys()),
                        "tasks_completed": [
                            t.id for t in tasks_to_run if t.status == TaskStatus.COMPLETED
                        ],
                    },
                )
            )

        return state

    @staticmethod
    def _get_pending_tasks(state: AgentState) -> list[Any]:
        """Find pending tasks in the current plan whose dependencies are completed."""
        if not state.task_plan or not state.task_plan.tasks:
            return []

        # If a specific task is designated via current_task_id, target only that task
        if state.current_task_id:
            for t in state.task_plan.tasks:
                if t.id == state.current_task_id:
                    t.status = TaskStatus.IN_PROGRESS
                    return [t]

        completed_ids = {t.id for t in state.task_plan.tasks if t.status == TaskStatus.COMPLETED}

        ready_tasks = []
        for t in state.task_plan.tasks:
            if t.status in (TaskStatus.PENDING, TaskStatus.RETRYING):
                # Check if all dependencies are satisfied
                if all(dep in completed_ids for dep in t.dependencies):
                    t.status = TaskStatus.IN_PROGRESS
                    ready_tasks.append(t)

        return ready_tasks

    @staticmethod
    def _is_valid_python(code: str) -> bool:
        """Check if string parses into valid Python AST with executable statements."""
        try:
            tree = ast.parse(code)
            if not tree.body:
                return False
            # Filter out lone string literals (common when models wrap thoughts in backticks)
            if len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr):
                if isinstance(tree.body[0].value, (ast.Constant, ast.Str)):
                    return False
            return True
        except (SyntaxError, ValueError):
            return False

    @classmethod
    def _extract_all_code_blocks(cls, text: str, context_hint: str = "") -> dict[str, str]:
        """
        Extract markdown code blocks from text, attempting to deduce filename
        from comments, headers, or task context hints, with AST validation to reject conversational prose.
        """
        blocks: dict[str, str] = {}
        pattern = re.compile(r"```(?:(\w+))?(?::([^\n\r]+))?\s*\n([\s\S]*?)```")

        idx = 1
        for match in pattern.finditer(text):
            lang = (match.group(1) or "").strip().lower()
            file_hint = (match.group(2) or "").strip()
            code = match.group(3).rstrip()

            filename = ""
            if file_hint:
                raw_path = file_hint.strip().lstrip("/\\")
                norm_p = Path(raw_path)
                filename = (
                    norm_p.name
                    if (".." in norm_p.parts or norm_p.is_absolute())
                    else norm_p.as_posix()
                )
            else:
                first_line = code.split("\n", 1)[0].strip()
                comment_match = (
                    re.match(r"^#\s*(?:File|filename|path):\s*([^\s]+)", first_line, re.IGNORECASE)
                    or re.match(
                        r"^//\s*(?:File|filename|path):\s*([^\s]+)", first_line, re.IGNORECASE
                    )
                    or re.match(
                        r"^/\*\s*(?:File|filename|path):\s*([^\s*]+)\s*\*/",
                        first_line,
                        re.IGNORECASE,
                    )
                    or re.match(
                        r"^<!--\s*(?:File|filename|path):\s*([^\s>]+)\s*-->",
                        first_line,
                        re.IGNORECASE,
                    )
                    or re.match(
                        r"^<!--\s*([a-zA-Z0-9_\-.]+\.[a-zA-Z0-9]+)\s*-->", first_line, re.IGNORECASE
                    )
                    or re.match(
                        r"^(?:[a-zA-Z0-9_]+:)?([a-zA-Z0-9_\-.]+\.[a-zA-Z0-9]+)$", first_line
                    )
                )
                if comment_match:
                    raw_path = comment_match.group(1).strip().lstrip("/\\")
                    norm_p = Path(raw_path)
                    filename = (
                        norm_p.name
                        if (".." in norm_p.parts or norm_p.is_absolute())
                        else norm_p.as_posix()
                    )

            if not filename:
                if lang in ("python", "py"):
                    ext = "py"
                elif lang in ("html", "htm"):
                    ext = "html"
                elif lang in ("css",):
                    ext = "css"
                elif lang in ("javascript", "js"):
                    ext = "js"
                elif not lang:
                    ext = "py" if cls._is_valid_python(code) else "txt"
                else:
                    ext = lang if len(lang) <= 4 else "txt"

                # Check if a specific target filename was requested in the task context
                target_match = re.search(
                    r"\b([\w\-.]+\." + re.escape(ext) + r")\b", context_hint, re.IGNORECASE
                )
                if target_match:
                    filename = target_match.group(1)
                elif ext == "html":
                    filename = "index.html"
                elif ext == "css":
                    filename = "style.css"
                else:
                    filename = f"code_block_{idx}.{ext}"

            # Clean first line if it's a file header
            code_lines = code.splitlines()
            if code_lines and (
                re.match(r"^#\s*(?:File|filename|path):\s*", code_lines[0], re.IGNORECASE)
                or re.match(r"^//\s*(?:File|filename|path):\s*", code_lines[0], re.IGNORECASE)
                or re.match(r"^/\*\s*(?:File|filename|path):\s*", code_lines[0], re.IGNORECASE)
                or re.match(r"^<!--\s*(?:File|filename|path):\s*", code_lines[0], re.IGNORECASE)
                or re.match(
                    r"^<!--\s*[a-zA-Z0-9_\-.]+\.[a-zA-Z0-9]+\s*-->", code_lines[0], re.IGNORECASE
                )
                or re.match(
                    r"^(?:[a-zA-Z0-9_]+:)?([a-zA-Z0-9_\-.]+\.[a-zA-Z0-9]+)$", code_lines[0].strip()
                )
            ):
                code = "\n".join(code_lines[1:]).strip()

            # Clean simulated write_file call if model wrapped content in one
            m_pseudo = re.search(
                r'write_file\s*\(?\{?\s*["\']?(?:path|filename)["\']?\s*:\s*["\'][^"\']+["\'],\s*["\']?content["\']?\s*:\s*"""([\s\S]*?)"""\s*\}?\)?',
                code,
            )
            if m_pseudo:
                code = m_pseudo.group(1).strip()
            else:
                m_pseudo2 = re.search(
                    r'write_file\s*\(?\{?\s*["\']?(?:path|filename)["\']?\s*:\s*["\'][^"\']+["\'],\s*["\']?content["\']?\s*:\s*"([\s\S]*?)"\s*\}?\)?',
                    code,
                )
                if m_pseudo2:
                    code = m_pseudo2.group(1).strip()

            # Strip trailing tool call leakage if model appended next tool call into code
            if re.search(r"<(?:function|tool_call|parameter)=", code):
                code = re.split(r"<(?:function|tool_call|parameter)=", code)[0].rstrip()

            # If block contains a raw JSON tool call, do not register as a deliverable artifact
            if filename.endswith(".json"):
                try:
                    import json

                    parsed = json.loads(code)
                    if (
                        isinstance(parsed, dict)
                        and "name" in parsed
                        and ("arguments" in parsed or "parameters" in parsed)
                    ):
                        idx += 1
                        continue
                except Exception:
                    pass

            # Only accept python files that compile cleanly into AST
            if filename.endswith(".py") and not cls._is_valid_python(code):
                idx += 1
                continue

            if filename not in blocks or len(code) > len(blocks[filename]):
                blocks[filename] = code
            idx += 1

        return blocks
