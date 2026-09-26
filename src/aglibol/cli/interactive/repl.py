"""Core interactive REPL session orchestrating prompt_toolkit, TUI, and multi-agent workflow."""

from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.prompt import Confirm

REPL_STYLE = Style.from_dict(
    {
        "completion-menu.completion": "bg:#1e1e2e #cdd6f4",
        "completion-menu.completion.current": "bg:#313244 #89b4fa bold",
        "completion-menu.meta.completion": "bg:#181825 #a6adc8",
        "completion-menu.meta.completion.current": "bg:#45475a #f9e2af italic",
        "scrollbar.background": "bg:#181825",
        "scrollbar.button": "bg:#45475a",
        "bottom-toolbar": "bg:#181825 #cdd6f4",
        "bottom-toolbar.text": "#cdd6f4",
        "frame": "#89b4fa",
        "frame.border": "#89b4fa",
        "frame.label": "#89b4fa bold",
    }
)

from aglibol.cli.interactive.clipboard import OSClipboard

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_pt_mod = sys.modules.get("prompt_toolkit.shortcuts.prompt")
if _pt_mod and hasattr(_pt_mod, "Frame"):
    _orig_pt_frame = _pt_mod.Frame

    def _titled_frame(body, *args, **kwargs):
        if not kwargs.get("title"):
            kwargs["title"] = " Prompt (Shift+Enter: newline, Enter: send) "
        return _orig_pt_frame(body, *args, **kwargs)

    _pt_mod.Frame = _titled_frame

from aglibol.agents.coder import CoderAgent  # noqa: F401
from aglibol.agents.planner import PlannerAgent
from aglibol.agents.reviewer import ReviewerAgent
from aglibol.agents.router import IntentRouter
from aglibol.agents.writer import WriterAgent
from aglibol.cli.interactive.commands import SlashCommandHandler
from aglibol.cli.interactive.completer import AgentInputCompleter
from aglibol.cli.interactive.hitl import SafetyMode, SecurityGate
from aglibol.cli.interactive.keybindings import build_editor_keybindings
from aglibol.cli.interactive.logo import render_logo
from aglibol.cli.interactive.selector import ModelCatalogHelper
from aglibol.cli.interactive.tui_renderer import TUIRenderer
from aglibol.core.config import ConfigManager
from aglibol.core.context import ContextManager
from aglibol.core.events import AgentEvent, EventBus
from aglibol.core.hardware import HardwareProfiler
from aglibol.core.optimizer import ResourceOptimizer
from aglibol.core.scheduler import ModelScheduler
from aglibol.core.task_orchestrator import TaskOrchestrator
from aglibol.core.types import AgentMode, AgentState, ChatMessage, HardwareProfile, TaskPlan
from aglibol.memory.episodic import EpisodicMemory
from aglibol.memory.working import WorkingMemory
from aglibol.ollama.client import OllamaClient
from aglibol.storage.artifact_store import ArtifactStore
from aglibol.storage.brain import Brain
from aglibol.storage.checkpoint import CheckpointStore
from aglibol.storage.session import SessionManager
from aglibol.tools.file_ops import register_file_tools
from aglibol.tools.registry import ToolRegistry
from aglibol.tools.search_replace import register_search_replace_tools
from aglibol.tools.shell import register_shell_tools

console = Console()

BINARY_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".ico",
    ".svg",
    ".webp",
    ".mp3",
    ".mp4",
    ".wav",
    ".avi",
    ".mov",
    ".mkv",
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".7z",
    ".rar",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".bin",
    ".dat",
    ".db",
    ".sqlite",
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".class",
    ".jar",
    ".pyc",
    ".pyo",
    ".o",
    ".obj",
}


class InteractiveSession:
    """Manages the full lifecycle of an interactive developer CLI session."""

    def __init__(
        self,
        workspace_root: Path | None = None,
        config_path: str | None = None,
        workflow_name: str = "default",
        safety_mode: SafetyMode | None = None,
    ) -> None:
        self.config = ConfigManager.load(config_path)

        default_ws = getattr(self.config, "default_workspace", None)
        chosen_ws = (
            workspace_root
            or (Path(default_ws) if default_ws and Path(default_ws).exists() else None)
            or Path.cwd()
        )
        raw_ws = chosen_ws.resolve()
        home_resolved = Path.home().resolve()
        if raw_ws == home_resolved:
            safe_ws = home_resolved / "aglibol_workspace"
            safe_ws.mkdir(parents=True, exist_ok=True)
            self.workspace_root = safe_ws
            console.print(
                f"  [dim cyan][Workspace][/dim cyan] Safeguarding user root. Using default workspace at: [bold]{safe_ws}[/bold]"
            )
        else:
            self.workspace_root = raw_ws

        saved_mode = getattr(self.config, "agent_mode", "auto")
        try:
            self.agent_mode = AgentMode(saved_mode)
        except ValueError:
            self.agent_mode = AgentMode.AUTO

        self.pending_plan: TaskPlan | None = None
        self.awaiting_plan_approval: bool = False
        self.profile: HardwareProfile = HardwareProfiler.detect()
        self.client = OllamaClient(host=self.config.ollama.host)

        if (
            workflow_name == "default"
            and getattr(self.config.workflow, "name", "default") != "default"
        ):
            self.workflow_name = self.config.workflow.name
        else:
            self.workflow_name = workflow_name

        self.should_exit = False

        if safety_mode is not None:
            effective_safety = safety_mode
        else:
            saved_safety = getattr(self.config, "safety_mode", None)
            effective_safety = SafetyMode.BALANCED
            if saved_safety:
                try:
                    effective_safety = SafetyMode(saved_safety.lower())
                except ValueError:
                    pass
        self.security_gate = SecurityGate(mode=effective_safety)

        # Storage & persistence
        self.brain = Brain(self.config.storage.brain_dir)
        self.sm = SessionManager(self.brain.sessions_dir)
        self.meta = self.sm.create_session(goal="Interactive REPL Session", tier=self.profile.tier)
        self.session_id = self.meta.session_id

        self.session_dir = self.brain.get_session_dir(self.session_id)
        self.checkpoint_store = CheckpointStore(self.session_dir / "checkpoints.db")
        self.artifact_store = ArtifactStore(self.session_dir)
        self.working_memory = WorkingMemory()
        self.episodic_memory = EpisodicMemory(self.session_dir / "episodic.jsonl")

        # Context & conversation
        self.history: list[ChatMessage] = []
        self.pinned_files: dict[str, str] = {}
        self.current_state: AgentState | None = None

        # Model bindings
        self.model_overrides = {
            "chat": getattr(self.config.models, "chat", "qwen2.5:7b"),
            "planner": self.config.models.planner,
            "coder": self.config.models.coder,
            "reviewer": self.config.models.reviewer,
            "writer": getattr(self.config.models, "writer", "qwen2.5:7b"),
        }
        self.installed_models_cache: list[str] = []
        self.status_notice: str = ""
        self.active_live_thought_box: Any = None
        self.active_tool_container: Any = None
        self.active_activity_view: Any = None

        # Event bus
        self.event_bus = EventBus()
        self._setup_event_listeners()

        # Tools
        self.tool_registry = ToolRegistry()
        register_file_tools(self.tool_registry, workspace_root=self.workspace_root)
        register_shell_tools(self.tool_registry, workspace_root=self.workspace_root)
        register_search_replace_tools(self.tool_registry, workspace_root=self.workspace_root)

        # Model Scheduler
        self.scheduler = ModelScheduler(
            client=self.client, profile=self.profile, event_bus=self.event_bus
        )

        # Slash command handler
        self.command_handler = SlashCommandHandler(self)

    def switch_workspace(self, new_path: Path) -> Path:
        """Switch active workspace directory and re-register sandboxed tools."""
        resolved = new_path.resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        self.workspace_root = resolved

        try:
            ConfigManager.set_user_config_value("default_workspace", str(self.workspace_root))
        except Exception:
            pass

        # Re-register tools with updated workspace_root
        self.tool_registry = ToolRegistry()
        register_file_tools(self.tool_registry, workspace_root=self.workspace_root)
        register_shell_tools(self.tool_registry, workspace_root=self.workspace_root)
        register_search_replace_tools(self.tool_registry, workspace_root=self.workspace_root)

        if self.current_state:
            self.current_state.workspace_dir = str(self.workspace_root)

        return self.workspace_root

    def auto_resolve_models(self) -> None:
        """Auto-resolve configured models against installed local models to prevent 404 errors."""
        if not self.installed_models_cache:
            return
        for role in ("chat", "planner", "coder", "reviewer", "writer"):
            current = self.model_overrides.get(role, "")
            resolved, was_auto = ModelCatalogHelper.resolve_best_model_for_role(
                requested_model=current,
                role=role,
                installed_models=self.installed_models_cache,
            )
            if was_auto and resolved != current:
                self.model_overrides[role] = resolved
                try:
                    ConfigManager.set_user_config_value(f"models.{role}", resolved)
                except Exception:
                    pass
                console.print(
                    f"  [dim yellow][Notice] Model '{current}' not installed for {role.capitalize()}. "
                    f"Auto-routed to installed model '[bold cyan]{resolved}[/bold cyan]' (saved to config).[/dim yellow]"
                )

    async def refresh_installed_models(self) -> None:
        """Refresh cache of installed models from local Ollama daemon."""
        try:
            models = await self.client.list_models()
            self.installed_models_cache = [m.name for m in models]
            self.auto_resolve_models()
        except Exception:
            pass

    @property
    def current_num_ctx(self) -> int:
        """Derive active context window size based on current hardware."""
        try:
            opt = ResourceOptimizer.get_params_for_profile(
                profile=self.profile,
                role="coder",
                requested_model=self.model_overrides["coder"],
            )
            return opt.num_ctx
        except Exception:
            return 4096

    def _setup_event_listeners(self) -> None:
        """Attach live UI feedback handlers to EventBus."""

        async def _on_event(ev: AgentEvent) -> None:
            if ev.event_type == "MODEL_LOAD_STARTED":
                num_ctx = ev.data.get("num_ctx")
                ctx_desc = f" (context: {num_ctx:,} tokens)" if num_ctx else ""
                model_str = ev.data.get("model", "")
                if self.active_activity_view:
                    self.active_activity_view.set_notice(f"Model: {model_str}{ctx_desc}")
                else:
                    console.print(
                        f"  [dim cyan][Model][/dim cyan] [bold]{model_str}[/bold]{ctx_desc}"
                    )
            elif ev.event_type == "MODEL_UNLOADED":
                model_str = ev.data.get("model", "")
                if not self.active_activity_view:
                    console.print(f"  [dim][VRAM Freed] Evicted {model_str}[/dim]")
            elif ev.event_type in ("AGENT_THOUGHT", "AGENT_THOUGHT_TOKEN"):
                content = ev.data.get("token") or ev.data.get("content", "")
                stream_type = ev.data.get("stream_type", "thinking")
                is_thinking = stream_type == "thinking"
                if self.active_activity_view:
                    self.active_activity_view.update_thought(
                        content,
                        is_thinking=is_thinking,
                        agent_name=ev.agent or self.active_activity_view.current_agent,
                    )
                elif self.active_live_thought_box:
                    for line in content.splitlines():
                        if line.strip():
                            self.active_live_thought_box.update_token(line.strip() + "\n")
                elif self.active_tool_container:
                    self.active_tool_container.finish()
                    self.active_tool_container = None
                    TUIRenderer.render_thought_box(content, ev.agent or "agent")
                else:
                    TUIRenderer.render_thought_box(content, ev.agent or "agent")
            elif ev.event_type == "TOOL_EXEC_STARTED":
                if self.active_activity_view:
                    self.active_activity_view.add_tool_start(
                        tool_name=ev.data.get("tool", ""),
                        args=ev.data.get("args", {}),
                    )
                else:
                    if self.active_live_thought_box:
                        self.active_live_thought_box.finish()
                        self.active_live_thought_box = None
                    if not self.active_tool_container:
                        self.active_tool_container = TUIRenderer.create_live_tool_container()
                    self.active_tool_container.add_tool_start(
                        tool_name=ev.data.get("tool", ""),
                        args=ev.data.get("args", {}),
                    )
            elif ev.event_type == "TOOL_EXEC_FINISHED":
                if self.active_activity_view:
                    self.active_activity_view.update_tool_finish(
                        tool_name=ev.data.get("tool", ""),
                        success=ev.data.get("success", False),
                        output=str(ev.data.get("output", "") or ""),
                        error=ev.data.get("error"),
                    )
                elif self.active_tool_container:
                    self.active_tool_container.update_tool_finish(
                        tool_name=ev.data.get("tool", ""),
                        success=ev.data.get("success", False),
                        output=str(ev.data.get("output", "") or ""),
                        error=ev.data.get("error"),
                    )
            elif ev.event_type == "REVIEW_COMPLETED":
                dec = str(ev.data.get("decision", "")).upper()
                if self.active_activity_view:
                    self.active_activity_view.set_review_outcome(dec)
                else:
                    color = "bold green" if dec == "APPROVED" else "bold yellow"
                    console.print(f"  [{color}][Review Outcome] {dec}[/{color}]")
            elif ev.event_type == "TASK_STARTED":
                t_idx = ev.data.get("task_index", 1)
                t_tot = ev.data.get("total_tasks", 1)
                t_title = ev.data.get("title", "")
                if self.active_activity_view:
                    self.active_activity_view.set_task_progress(t_idx, t_tot, t_title)

        self.event_bus.subscribe("*", _on_event)

    def get_prompt_html(self) -> HTML:
        """Construct Grok Build CLI-style active prompt input prefix."""
        import html

        mode_tag = html.escape(self.agent_mode.value.lower())
        return HTML(
            f"<b><cyan>aglibol</cyan></b> "
            f"<ansibrightblack>[</ansibrightblack><b>{mode_tag}</b><ansibrightblack>]</ansibrightblack> "
            f"<b><cyan>&gt;</cyan></b> "
        )

    def get_bottom_toolbar(self) -> HTML:
        """Dynamic bottom telemetry status bar with Grok Build CLI styling."""
        import html

        raw_gpu = self.profile.primary_gpu.name if self.profile.primary_gpu else "CPU"
        gpu_name = html.escape(raw_gpu)
        resident = html.escape(self.scheduler.current_loaded_model or "None")
        used_tokens = ContextManager.estimate_messages_tokens(self.history)
        max_ctx = self.current_num_ctx

        percent = int(min(1.0, used_tokens / max(1, max_ctx)) * 100)
        mode_str = html.escape(self.security_gate.mode.value.upper())
        agent_mode_str = html.escape(self.agent_mode.value.upper())
        tier_str = html.escape(self.profile.tier.value.upper())
        notice_prefix = ""
        if self.status_notice:
            notice_prefix = (
                f"<b><ansiyellow>[{html.escape(self.status_notice)}]</ansiyellow></b> | "
            )

        return HTML(
            f"{notice_prefix}"
            f" <b>{tier_str}</b> ({gpu_name}) | "
            f"Mode: <b>{agent_mode_str}</b> | "
            f"Safety: <b>{mode_str}</b> | "
            f"Model: <b>{resident}</b> | "
            f"Context: <b>{used_tokens:,}/{max_ctx:,} ({percent}%)</b> | "
            f"<b>/help</b>"
        )

    def _extract_and_pin_at_mentions(self, text: str) -> None:
        """Detect '@path/to/file' in user input and pin it automatically."""
        matches = re.findall(r"@([^\s\"']+)", text)
        for m in matches:
            candidate = (self.workspace_root / m).resolve()
            try:
                candidate.relative_to(self.workspace_root.resolve())
                if candidate.is_file():
                    if candidate.suffix.lower() in BINARY_EXTENSIONS:
                        console.print(
                            f"  [dim yellow][Notice] Skipping binary file @{m}[/dim yellow]"
                        )
                        continue
                    content = candidate.read_text(encoding="utf-8", errors="replace")
                    rel_path = candidate.relative_to(self.workspace_root).as_posix()
                    self.pinned_files[rel_path] = content
                    console.print(f"  [dim green][Context Attached] @{rel_path}[/dim green]")
            except ValueError:
                console.print(
                    f"  [dim yellow][Notice] @{m} is outside the workspace boundary[/dim yellow]"
                )
            except Exception as e:
                console.print(f"  [dim yellow][Notice] Could not attach @{m}: {e}[/dim yellow]")

    async def execute_turn(self, user_goal: str) -> None:
        """Execute a full agent turn based on user input, routing by mode or approval state."""
        # Update session goal if this is the first turn
        if self.meta.user_goal in ("Interactive REPL Session", ""):
            clean_goal = user_goal.strip().split("\n")[0][:80]
            self.meta.user_goal = clean_goal or "Interactive REPL Session"
            self.sm.save_meta(self.meta)

        # 1. Check if we are waiting for user plan approval
        if self.awaiting_plan_approval and self.pending_plan:
            decision = IntentRouter.detect_plan_approval(user_goal)
            if decision == "approve":
                console.print(
                    "\n[bold green][Plan Approved][/bold green] Transitioning to [bold yellow]Coder[/bold yellow] to implement plan..."
                )
                self.awaiting_plan_approval = False
                if self.agent_mode != AgentMode.AUTO:
                    self.agent_mode = AgentMode.CODER
                plan = self.pending_plan
                self.pending_plan = None
                state = AgentState(
                    session_id=self.session_id,
                    user_goal=plan.goal,
                    workspace_dir=str(self.workspace_root),
                    task_plan=plan,
                )
                if self.current_state and self.current_state.artifacts:
                    state.artifacts = dict(self.current_state.artifacts)
                    state.scratchpad = dict(self.current_state.scratchpad)
                await self._execute_coder_turn(state)
                return
            elif decision == "reject":
                console.print(
                    "\n[bold yellow][Plan Cancelled][/bold yellow] Plan discarded. You can state a new objective or ask questions.\n"
                )
                self.awaiting_plan_approval = False
                self.pending_plan = None
                return
            else:
                console.print(
                    f"\n[bold cyan][Plan Revision][/bold cyan] Revising plan with feedback: {user_goal}"
                )
                revised_goal = f"Original Plan Goal: {self.pending_plan.goal}\nUser feedback/modifications to incorporate:\n{user_goal}"
                state = AgentState(
                    session_id=self.session_id,
                    user_goal=revised_goal,
                    workspace_dir=str(self.workspace_root),
                )
                if self.current_state and self.current_state.artifacts:
                    state.artifacts = dict(self.current_state.artifacts)
                    state.scratchpad = dict(self.current_state.scratchpad)
                await self._execute_planner_turn(state)
                return

        # 2. Extract pinned context and sanitize objective
        clean_goal = IntentRouter.extract_clean_goal(user_goal)
        self._extract_and_pin_at_mentions(user_goal)
        prompt_parts = []
        if self.pinned_files:
            prompt_parts.append("--- Pinned Project Context ---")
            for fpath, fcontent in self.pinned_files.items():
                prompt_parts.append(f"File [{fpath}]:\n{fcontent}\n")
            prompt_parts.append("------------------------------\n")
        prompt_parts.append(clean_goal)
        full_prompt = "\n".join(prompt_parts)

        # 3. Dynamic Mode Classification
        explicit_mode = IntentRouter.detect_explicit_mode_request(user_goal)
        target_mode = IntentRouter.classify(user_goal, current_mode=self.agent_mode)

        if explicit_mode is not None:
            # Explicit user command: switch mode directly without redundant prompt modal
            self.auto_resolve_models()
            self.agent_mode = explicit_mode
            target_mode = explicit_mode
            target_model_name = self.model_overrides.get(target_mode.value, "default")
            console.print(
                f"  [bold green][Mode Switched][/bold green] Active mode is now "
                f"[bold yellow]{target_mode.value.upper()}[/bold yellow] (Model: [bold magenta]{target_model_name}[/bold magenta])\n"
            )
        elif self.agent_mode != AgentMode.AUTO and target_mode != self.agent_mode:
            # Implicit suggestion while pinned to a manual mode: confirm with user
            self.auto_resolve_models()
            current_mode_name = self.agent_mode.value
            current_model_name = self.model_overrides.get(current_mode_name, "default")
            target_mode_name = target_mode.value
            target_model_name = self.model_overrides.get(target_mode_name, "default")

            reason = f"Objective matched {target_mode.value.upper()} capabilities"
            if target_mode == AgentMode.PLANNER:
                reason = "Objective requires project architecture planning before implementation"
            elif target_mode == AgentMode.CODER:
                reason = "Objective requires writing or modifying code"
            elif target_mode == AgentMode.REVIEWER:
                reason = "Objective requires code review or bug audit"
            elif target_mode == AgentMode.WRITER:
                reason = "Objective requires technical documentation or copywriting"

            approved = TUIRenderer.confirm_mode_switch(
                current_mode=current_mode_name,
                current_model=current_model_name,
                target_mode=target_mode_name,
                target_model=target_model_name,
                reason=reason,
            )
            if approved:
                console.print(
                    f"  [bold green][Mode Switch Approved][/bold green] Active mode is now "
                    f"[bold yellow]{target_mode.value.upper()}[/bold yellow] (Model: [bold magenta]{target_model_name}[/bold magenta])\n"
                )
                self.agent_mode = target_mode
            else:
                console.print(
                    f"  [dim yellow]Keeping current mode [{self.agent_mode.value.upper()}].[/dim yellow]\n"
                )
                target_mode = self.agent_mode

        # Prepare state
        state = AgentState(
            session_id=self.session_id,
            user_goal=full_prompt,
            workspace_dir=str(self.workspace_root),
        )
        if self.current_state and self.current_state.artifacts:
            state.artifacts = dict(self.current_state.artifacts)
            state.scratchpad = dict(self.current_state.scratchpad)

        # 4. Dispatch by determined mode
        if target_mode == AgentMode.CHAT:
            await self._execute_chat_turn(full_prompt)
        elif target_mode == AgentMode.PLANNER:
            await self._execute_planner_turn(state)
        elif target_mode == AgentMode.CODER:
            await self._execute_coder_turn(state)
        elif target_mode == AgentMode.REVIEWER:
            await self._execute_reviewer_turn(state)
        elif target_mode == AgentMode.WRITER:
            await self._execute_writer_turn(state)
        else:
            await self._execute_chat_turn(full_prompt)

    async def _execute_chat_turn(self, user_goal: str) -> None:
        """Execute a conversational assistant turn without triggering heavy multi-agent workflows."""
        self.auto_resolve_models()
        chat_model = self.model_overrides.get("chat", "qwen2.5:7b")

        model, options, keep_alive = await self.scheduler.acquire_model(
            model_name=chat_model,
            role="chat",
        )

        system_msg = ChatMessage(
            role="system",
            content=(
                "You are Aglibol Agent, an intelligent, open-source multi-agent developer assistant running locally on personal hardware. "
                "Answer questions clearly, concisely, and helpfully in English or the requested language. "
                "If the user greets you or asks who you are, introduce Aglibol Agent, its local multi-agent capabilities (Planner, Coder, Reviewer, Writer), and slash commands (/help, /model, /mode, /workspace). "
                "Keep answers direct and friendly."
            ),
        )

        # 1. Log incoming user turn to episodic memory on disk
        self.episodic_memory.log_turn(
            agent="user",
            step=self.meta.step_count + 1,
            message=ChatMessage(role="user", content=user_goal),
        )

        messages = [system_msg]
        if self.history:
            messages.extend(self.history)
        messages.append(ChatMessage(role="user", content=user_goal))

        # Dynamic context compression and episodic offload to disk
        messages = ContextManager.compress_messages(
            messages=messages,
            max_context_tokens=self.current_num_ctx,
            headroom_ratio=0.60,
            episodic_memory=self.episodic_memory,
        )

        act_view = TUIRenderer.create_live_activity_view()
        self.active_activity_view = act_view
        act_view.start(agent_name="assistant", initial_phase="Thinking and preparing response...")

        def _on_chat_token(token_type: str, token_str: str) -> None:
            act_view.update_thought(
                token_str,
                is_thinking=(token_type == "thinking"),
                agent_name="assistant",
            )

        try:
            res = await self.client.chat(
                model=model,
                messages=messages,
                options=options,
                keep_alive=keep_alive,
                on_token=_on_chat_token,
            )
            content = res.get("content", "").strip()
            thinking = res.get("thinking", "").strip()
            if thinking and not act_view.current_thought:
                act_view.update_thought(thinking, is_thinking=True, agent_name="assistant")

            # Auto-continuation loop to prevent truncation on context or eval limit
            continuation_turns = 0
            while res.get("done_reason") == "length" and continuation_turns < 3:
                continuation_turns += 1
                act_view.set_notice("Auto-continuing generation to prevent truncation...")
                messages.append(ChatMessage(role="assistant", content=content))
                messages.append(
                    ChatMessage(
                        role="user",
                        content="Continue generation directly from where you left off. Output only the continuation without repeating previous text.",
                    )
                )
                messages = ContextManager.compress_messages(
                    messages=messages,
                    max_context_tokens=self.current_num_ctx,
                    headroom_ratio=0.60,
                    episodic_memory=self.episodic_memory,
                )
                res = await self.client.chat(
                    model=model,
                    messages=messages,
                    options=options,
                    keep_alive=keep_alive,
                    on_token=_on_chat_token,
                )
                chunk = res.get("content", "").strip()
                if not chunk:
                    break
                content += ("\n" + chunk) if not content.endswith(" ") else chunk

            if not content:
                if thinking:
                    content = thinking
                else:
                    content = "Hello! I am Aglibol Agent. How can I assist you today? (Type /help for available commands)"

            if act_view:
                act_view.finish()
            self.active_activity_view = None

            TUIRenderer.render_assistant_response(content, title="Aglibol Assistant")
            self.history.append(ChatMessage(role="user", content=user_goal))
            self.history.append(ChatMessage(role="assistant", content=content))
            self.episodic_memory.log_turn(
                agent="assistant",
                step=self.meta.step_count + 1,
                message=ChatMessage(role="assistant", content=content),
            )
            self.meta.step_count += 1
            self.sm.save_meta(self.meta)
            chat_state = AgentState(
                session_id=self.session_id,
                user_goal=self.meta.user_goal,
                current_step=self.meta.step_count,
                active_agent="chat",
                messages=list(self.history),
                workspace_dir=str(self.workspace_root),
                status="running",
            )
            self.checkpoint_store.save_checkpoint(chat_state)
            self.current_state = chat_state
            console.print("[bold green]Turn completed successfully![/bold green]\n")
        except Exception as e:
            if act_view:
                act_view.finish()
            self.active_activity_view = None
            console.print(f"[bold red]Chat error: {e}[/bold red]")

    async def _execute_planner_turn(self, state: AgentState) -> None:
        """Execute PlannerAgent to decompose objective into an architecture and task plan with an approval gate."""
        self.auto_resolve_models()
        planner_model = self.model_overrides.get("planner", "qwen2.5:7b")
        planner = PlannerAgent(model_name=planner_model)

        console.print(f"\n[bold cyan][Objective][/bold cyan] {state.user_goal}")
        act_view = TUIRenderer.create_live_activity_view()
        self.active_activity_view = act_view
        act_view.start(
            agent_name="planner", initial_phase="Analyzing architecture and decomposing tasks..."
        )

        try:
            state = await planner.execute(
                state=state,
                client=self.client,
                scheduler=self.scheduler,
                tool_registry=self.tool_registry,
                event_bus=self.event_bus,
            )
        finally:
            if act_view:
                act_view.finish()
            self.active_activity_view = None

        self.current_state = state
        self.meta.step_count += 1
        self.sm.save_meta(self.meta)
        state.current_step = self.meta.step_count
        self.checkpoint_store.save_checkpoint(state)
        self.episodic_memory.log_turn(
            agent="planner",
            step=self.meta.step_count,
            message=ChatMessage(
                role="assistant",
                content=str(
                    state.task_plan.summary
                    if state.task_plan
                    else "Planner formulated initial plan."
                ),
            ),
        )

        if state.task_plan and state.task_plan.tasks:
            self.pending_plan = state.task_plan
            self.awaiting_plan_approval = True
            TUIRenderer.render_plan_card(state.task_plan)
        else:
            summary = (
                state.task_plan.summary
                if state.task_plan
                else "Planner formulated initial direction."
            )
            TUIRenderer.render_assistant_response(summary, title="Planner")

        console.print("[bold green]Turn completed successfully![/bold green]\n")

    async def _execute_coder_turn(self, state: AgentState) -> None:
        """Execute TaskOrchestrator to process plan tasks sequentially with checkpointing and critique loops."""
        self.auto_resolve_models()

        console.print(f"\n[bold cyan][Objective][/bold cyan] {state.user_goal}")

        act_view = TUIRenderer.create_live_activity_view()
        self.active_activity_view = act_view
        act_view.start(agent_name="coder", initial_phase="Starting task execution engine...")

        orchestrator = TaskOrchestrator(
            client=self.client,
            scheduler=self.scheduler,
            tool_registry=self.tool_registry,
            checkpoint_store=self.checkpoint_store,
            artifact_store=self.artifact_store,
            event_bus=self.event_bus,
            episodic_memory=self.episodic_memory,
            model_overrides=self.model_overrides,
            max_review_cycles=3,
        )

        def _on_task_update(current: int, total: int, task: Any) -> None:
            if act_view:
                act_view.set_task_progress(current, total, task.title)
                act_view.set_agent(
                    task.assigned_agent,
                    f"Step {current}/{total}: {task.title}",
                )

        try:
            state = await orchestrator.execute_plan(state, on_task_update=_on_task_update)
        finally:
            if act_view:
                act_view.finish()
            self.active_activity_view = None

        self.current_state = state
        self.meta.step_count += 1
        self.sm.save_meta(self.meta)
        state.current_step = self.meta.step_count
        self.checkpoint_store.save_checkpoint(state)

        # Auto-write artifacts to workspace and render with syntax highlighting
        if state.artifacts:
            for fname, content in list(state.artifacts.items()):
                if "\n" not in content and "\\n" in content:
                    content = content.replace("\\n", "\n").replace("\\t", "\t")
                    state.artifacts[fname] = content
                dest = self.workspace_root / fname
                try:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(content, encoding="utf-8")
                except Exception as e:
                    console.print(f"[dim yellow]Warning writing {fname}: {e}[/dim yellow]")
                TUIRenderer.render_code_artifact(
                    fname, content, title=f"Generated Artifact: {fname}"
                )

        # Report turn outcome
        if state.status == "success" or state.review_status == "approved":
            console.print(
                "\n[bold green]All tasks completed and verified successfully![/bold green]"
            )
        else:
            console.print(
                f"\n[bold yellow][Turn Ended] Status [{state.status}]: {state.error or state.review_feedback or 'Done'}[/bold yellow]"
            )

        if state.artifacts:
            console.print(
                "[dim]Active workspace artifacts: " + ", ".join(state.artifacts.keys()) + "[/dim]\n"
            )

    async def _execute_reviewer_turn(self, state: AgentState) -> None:
        """Execute ReviewerAgent directly on current workspace or artifacts."""
        self.auto_resolve_models()
        reviewer_model = self.model_overrides.get("reviewer", "qwen2.5:7b")
        reviewer = ReviewerAgent(model_name=reviewer_model)

        console.print(f"\n[bold cyan][Objective][/bold cyan] {state.user_goal}")
        act_view = TUIRenderer.create_live_activity_view()
        self.active_activity_view = act_view
        act_view.start(
            agent_name="reviewer",
            initial_phase="Auditing code syntax, correctness, and security...",
        )
        try:
            state = await reviewer.execute(
                state=state,
                client=self.client,
                scheduler=self.scheduler,
                tool_registry=self.tool_registry,
                event_bus=self.event_bus,
            )
        finally:
            if act_view:
                act_view.finish()
            self.active_activity_view = None

        self.current_state = state
        self.meta.step_count += 1
        self.sm.save_meta(self.meta)
        state.current_step = self.meta.step_count
        self.checkpoint_store.save_checkpoint(state)
        self.episodic_memory.log_turn(
            agent="reviewer",
            step=self.meta.step_count,
            message=ChatMessage(
                role="assistant", content=state.review_feedback or "Review completed."
            ),
        )
        TUIRenderer.render_assistant_response(state.review_feedback, title="Code Reviewer")
        console.print("[bold green]Turn completed successfully![/bold green]\n")

    async def _execute_writer_turn(self, state: AgentState) -> None:
        """Execute WriterAgent for documentation, articles, and copywriting."""
        self.auto_resolve_models()
        writer_model = self.model_overrides.get("writer", "qwen2.5:7b")
        writer = WriterAgent(model_name=writer_model)

        console.print(f"\n[bold cyan][Objective][/bold cyan] {state.user_goal}")
        act_view = TUIRenderer.create_live_activity_view()
        self.active_activity_view = act_view
        act_view.start(agent_name="writer", initial_phase="Drafting technical documentation...")
        try:
            state = await writer.execute(
                state=state,
                client=self.client,
                scheduler=self.scheduler,
                tool_registry=self.tool_registry,
                event_bus=self.event_bus,
            )
        finally:
            if act_view:
                act_view.finish()
            self.active_activity_view = None

        self.current_state = state
        self.meta.step_count += 1
        self.sm.save_meta(self.meta)
        state.current_step = self.meta.step_count
        self.checkpoint_store.save_checkpoint(state)
        self.episodic_memory.log_turn(
            agent="writer",
            step=self.meta.step_count,
            message=ChatMessage(
                role="assistant",
                content=f"Writer generated {len(state.artifacts)} document(s): {', '.join(state.artifacts.keys()) if state.artifacts else 'none'}",
            ),
        )

        # Auto-write artifacts to workspace
        if state.artifacts:
            for fname, content in state.artifacts.items():
                dest = self.workspace_root / fname
                try:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(content, encoding="utf-8")
                except Exception as e:
                    console.print(f"[dim yellow]Warning writing {fname}: {e}[/dim yellow]")

        last_msg = ""
        if state.messages:
            last_msg = state.messages[-1].content
        if last_msg:
            TUIRenderer.render_assistant_response(last_msg, title="Writer")

        console.print("[bold green]Turn completed successfully![/bold green]\n")
        if state.artifacts:
            console.print(
                "[dim]Active workspace artifacts: " + ", ".join(state.artifacts.keys()) + "[/dim]\n"
            )

    async def start(self) -> None:
        """Main REPL loop."""
        history_file = Path.home() / ".aglibol" / "repl_history"
        history_file.parent.mkdir(parents=True, exist_ok=True)

        # Preflight check for Ollama status FIRST (avoids 14s retry loop if daemon is offline)
        is_ollama_alive = await self.client.health_check()
        if not is_ollama_alive:
            console.print(
                f"\n[bold yellow][Notice] Ollama daemon is offline at {self.config.ollama.host}.[/bold yellow]"
            )
            has_ollama_bin = bool(shutil.which("ollama")) or (
                sys.platform == "win32"
                and (
                    Path(os.environ.get("LOCALAPPDATA", ""))
                    / "Programs"
                    / "Ollama"
                    / "ollama app.exe"
                ).exists()
            )
            if has_ollama_bin:
                try:
                    if Confirm.ask(
                        "[cyan]Start Ollama service in the background automatically?[/cyan]",
                        default=True,
                    ):
                        with console.status(
                            "[cyan]Starting Ollama service...[/cyan]", spinner="dots"
                        ):
                            started = await self.client.start_daemon(timeout_seconds=10.0)
                        if started:
                            console.print(
                                "[bold green]Ollama service started successfully![/bold green]\n"
                            )
                            is_ollama_alive = True
                        else:
                            console.print(
                                "[bold red]Could not start Ollama automatically. Run 'aglibol doctor' to inspect.[/bold red]\n"
                            )
                except (KeyboardInterrupt, EOFError):
                    pass

        if is_ollama_alive:
            # Preload installed model cache for smart completer
            await self.refresh_installed_models()
            if not self.installed_models_cache:
                opt = ResourceOptimizer.get_params_for_profile(self.profile)
                console.print(
                    "\n[bold yellow][Notice] No Ollama models found locally.[/bold yellow]"
                )
                console.print(
                    f"[dim]Run 'ollama pull {opt.recommended_model}' or '/doctor' to pull the recommended model.[/dim]\n"
                )
        else:
            console.print(
                "[dim]REPL launched in offline mode. Run 'aglibol doctor' or '/doctor' once Ollama is running.[/dim]\n"
            )

        # Render full truecolor ASCII brand logo on startup
        render_logo(console)

        session: PromptSession = PromptSession(
            completer=AgentInputCompleter(session=self, workspace_root=self.workspace_root),
            history=FileHistory(str(history_file)),
            bottom_toolbar=self.get_bottom_toolbar,
            complete_while_typing=True,
            style=REPL_STYLE,
            key_bindings=build_editor_keybindings(session=self),
            clipboard=OSClipboard(),
            mouse_support=False,
            multiline=True,
            show_frame=True,
        )

        gpu_name = self.profile.primary_gpu.name if self.profile.primary_gpu else "CPU"
        TUIRenderer.render_welcome_banner(
            session_id=self.session_id,
            tier=self.profile.tier.value,
            gpu_name=gpu_name,
            workspace=str(self.workspace_root),
        )

        saved_sessions = self.sm.list_sessions(limit=5, filter_empty=True)
        if saved_sessions:
            console.print(
                f"  [dim cyan][Session][/dim cyan] Found [bold]{len(saved_sessions)}[/bold] saved session(s). Type [bold green]/session[/bold green] to resume conversation history.\n"
            )

        try:
            while not self.should_exit:
                try:
                    # Prompt input inside dynamic boxed frame pinned at bottom
                    user_input = await session.prompt_async(
                        self.get_prompt_html(),
                        show_frame=True,
                        multiline=True,
                        handle_sigint=False,
                    )
                except KeyboardInterrupt:
                    # Direct interrupt handling
                    if self.should_exit:
                        break
                    continue
                except EOFError:
                    # Ctrl+D exits
                    break

                if not user_input.strip():
                    continue

                # Handle slash commands
                handled = await self.command_handler.handle(user_input)
                if handled:
                    continue

                # Execute multi-agent objective
                try:
                    await self.execute_turn(user_input.strip())
                except Exception as e:
                    console.print(f"[bold red]Execution error: {e}[/bold red]")

        finally:
            # Clean exit
            console.print("\n[dim]Cleaning up VRAM models and saving session...[/dim]")
            try:
                # If session has no user turns and no history, discard empty session
                if not self.history and self.meta.step_count == 0:
                    self.brain.clean_session(self.session_id)
                else:
                    self.meta.status = "completed"
                    self.sm.save_meta(self.meta)
            except Exception:
                pass

            try:
                await self.scheduler.release_current_model()
                await self.client.close()
            except Exception:
                pass
            console.print("[bold green]Goodbye![/bold green]\n")
