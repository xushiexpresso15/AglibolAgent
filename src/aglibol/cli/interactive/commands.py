"""Slash command handlers and dispatcher for the interactive REPL."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from aglibol.cli.display import Display
from aglibol.cli.interactive.hitl import SafetyMode
from aglibol.cli.interactive.selector import ModelCatalogHelper, SelectorItem, TUISelector
from aglibol.cli.interactive.tui_renderer import TUIRenderer
from aglibol.core.config import ConfigManager
from aglibol.core.context import ContextManager
from aglibol.core.optimizer import ResourceOptimizer
from aglibol.core.types import AgentMode, AgentState
from aglibol.memory.episodic import EpisodicMemory
from aglibol.storage.artifact_store import ArtifactStore
from aglibol.storage.checkpoint import CheckpointStore

if TYPE_CHECKING:
    from aglibol.cli.interactive.repl import InteractiveSession

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


class SlashCommandHandler:
    """Dispatches and handles all interactive slash commands."""

    def __init__(self, session: InteractiveSession) -> None:
        self.session = session

    async def handle(self, line: str) -> bool:
        """
        Parse and dispatch slash command.
        Returns True if command was handled, False if regular user prompt.
        """
        stripped = line.strip()
        if not stripped.startswith("/"):
            return False

        parts = stripped.split(maxsplit=2)
        cmd = parts[0].lower()
        arg1 = parts[1] if len(parts) > 1 else ""
        arg2 = parts[2] if len(parts) > 2 else ""

        if cmd in ("/help", "/?"):
            self._cmd_help()
        elif cmd == "/status":
            await self._cmd_status()
        elif cmd == "/models":
            await self._cmd_models()
        elif cmd == "/model":
            await self._cmd_model(arg1, arg2)
        elif cmd == "/mode":
            await self._cmd_mode(arg1)
        elif cmd == "/safety":
            await self._cmd_safety(arg1)
        elif cmd == "/workspace":
            await self._cmd_workspace(arg1, arg2)
        elif cmd == "/session":
            await self._cmd_session(arg1, arg2)
        elif cmd == "/sessions":
            await self._cmd_sessions(arg1, arg2)
        elif cmd == "/resume":
            await self._cmd_session("resume", arg1)
        elif cmd == "/add":
            self._cmd_add(arg1)
        elif cmd == "/drop":
            self._cmd_drop(arg1)
        elif cmd in ("/context", "/tokens"):
            self._cmd_context()
        elif cmd == "/diff":
            self._cmd_diff()
        elif cmd == "/undo":
            self._cmd_undo()
        elif cmd == "/commit":
            self._cmd_commit(arg1 + (" " + arg2 if arg2 else ""))
        elif cmd == "/compact":
            self._cmd_compact()
        elif cmd == "/workflow":
            self._cmd_workflow(arg1)
        elif cmd == "/init":
            self._cmd_init()
        elif cmd == "/doctor":
            await self._cmd_doctor()
        elif cmd == "/update":
            await self._cmd_update()
        elif cmd in ("/clear", "/cls"):
            os.system("cls" if sys.platform == "win32" else "clear")
        elif cmd in ("/exit", "/quit", "/q"):
            self.session.should_exit = True
        else:
            console.print(
                f"[yellow]Unknown command: {cmd}. Type /help for available commands.[/yellow]"
            )

        return True

    def _cmd_help(self) -> None:
        """Display slash commands manual."""
        table = Table(title="Interactive Slash Commands Palette", border_style="cyan")
        table.add_column("Command", style="bold yellow", no_wrap=True)
        table.add_column("Description", style="white")

        commands = [
            ("/help", "Show this commands reference list"),
            ("/status", "Display live hardware specs, VRAM residency, and Ollama status"),
            ("/doctor", "Run comprehensive system diagnostics and check Ollama/tools"),
            ("/update", "Check GitHub for updates and self-upgrade Aglibol Agent"),
            ("/models", "List installed Ollama models with size and quantization details"),
            (
                "/model [role] [name]",
                "View or switch active model for chat, planner, coder, reviewer, or writer",
            ),
            (
                "/mode [auto|chat|planner|coder|reviewer|writer]",
                "Switch agent operating mode or dynamic intent routing",
            ),
            (
                "/safety [balanced|strict|autonomous]",
                "Switch tool execution human-in-the-loop safety mode",
            ),
            (
                "/workspace [list|new|<path>]",
                "Inspect, list files, or switch active project workspace",
            ),
            (
                "/session [resume|delete|prune|list]",
                "Interactive manager to resume, delete, or prune conversation sessions",
            ),
            ("/resume [session_id]", "Shortcut alias for /session resume"),
            ("/add <file>", "Pin a file or folder into active agent context memory"),
            ("/drop <file>", "Unpin a file from active agent context memory"),
            ("/context", "Show active token budget, pinned files, and memory allocation"),
            ("/diff", "Display git/workspace diff of generated code modifications"),
            ("/undo", "Revert the most recent step / file changes via CheckpointStore"),
            ("/commit [msg]", "Stage and commit changes to local git repository"),
            ("/compact", "Force immediate context sliding-window compression"),
            ("/workflow [path]", "Inspect or switch active workflow definition"),
            ("/init", "Initialize local project .aglibol.yaml and AGENTS.md"),
            ("/clear", "Clear screen while preserving conversation state"),
            ("/exit, /quit", "Unload resident models and exit interactive session"),
        ]

        for c, d in commands:
            table.add_row(c, d)

        console.print(table)
        console.print(
            "[dim]Tip: Type '@' anywhere to autocomplete workspace files into your prompt.[/dim]\n"
        )

    async def _cmd_status(self) -> None:
        """Display live hardware and Ollama status."""
        Display.print_hardware_profile(self.session.profile)
        is_alive = await self.session.client.health_check()
        status_text = (
            "[bold green]ONLINE[/bold green]" if is_alive else "[bold red]OFFLINE[/bold red]"
        )
        console.print(f"Ollama Server: {self.session.client.host} • Status: {status_text}")

        if not is_alive:
            console.print(
                "[dim](Start Ollama via 'aglibol doctor' or 'ollama serve' to inspect VRAM)[/dim]"
            )
            return

        loaded = await self.session.client.ps()
        if loaded:
            table = Table(title="Resident Models in VRAM", border_style="green")
            table.add_column("Model", style="bold white")
            table.add_column("VRAM Size", style="cyan")
            table.add_column("Expires At", style="dim")
            for m in loaded:
                table.add_row(m.model, f"{m.size_vram_gb:.2f} GB", m.expires_at)
            console.print(table)
        else:
            console.print("[dim](No models currently resident in VRAM)[/dim]")

    async def _cmd_models(self) -> None:
        """List installed Ollama models."""
        if not await self.session.client.health_check():
            console.print(
                "[yellow]Ollama server is offline. Please start it with 'ollama serve' or run '/doctor'.[/yellow]"
            )
            return

        models = await self.session.client.list_models()
        if not models:
            console.print("[yellow]No models found or Ollama server is offline.[/yellow]")
            return

        table = Table(title="Installed Local Models", border_style="cyan")
        table.add_column("Model Name", style="bold white")
        table.add_column("Size (GB)", style="green")
        table.add_column("Params", style="magenta")
        table.add_column("Quant", style="yellow")

        for m in models:
            table.add_row(m.name, f"{m.size_gb:.1f} GB", m.parameter_size, m.quantization_level)

        console.print(table)

    async def _cmd_model(self, role: str, model_name: str) -> None:
        """Inspect or switch model binding for a role with interactive TUI selector."""
        if not role:
            while True:
                table = Table(title="Active Role Model Assignments", border_style="cyan")
                table.add_column("Role", style="bold white", width=14)
                table.add_column("Assigned Model", style="bold green")

                for r, m in self.session.model_overrides.items():
                    table.add_row(r.capitalize(), m)
                console.print(table)

                role_items = [
                    SelectorItem(
                        key="chat",
                        label="Chat",
                        specs=f"Current: {self.session.model_overrides.get('chat', 'default')}",
                        badge="[Conversational Assistant]",
                        badge_style="cyan",
                    ),
                    SelectorItem(
                        key="planner",
                        label="Planner",
                        specs=f"Current: {self.session.model_overrides.get('planner', 'default')}",
                        badge="[Decomposition & Architecture]",
                        badge_style="magenta",
                    ),
                    SelectorItem(
                        key="coder",
                        label="Coder",
                        specs=f"Current: {self.session.model_overrides.get('coder', 'default')}",
                        badge="[Code Implementation]",
                        badge_style="bold green",
                    ),
                    SelectorItem(
                        key="reviewer",
                        label="Reviewer",
                        specs=f"Current: {self.session.model_overrides.get('reviewer', 'default')}",
                        badge="[Quality & Syntax Gate]",
                        badge_style="cyan",
                    ),
                    SelectorItem(
                        key="writer",
                        label="Writer",
                        specs=f"Current: {self.session.model_overrides.get('writer', 'default')}",
                        badge="[Docs & Copywriting]",
                        badge_style="bold cyan",
                    ),
                    SelectorItem(
                        key="all",
                        label="All Roles",
                        specs="Bind all roles to one model",
                        badge="[Unified Model]",
                        badge_style="yellow",
                    ),
                ]

                gpu_name = (
                    self.session.profile.primary_gpu.name
                    if self.session.profile.primary_gpu
                    else "CPU"
                )
                role_choice = TUISelector.choose(
                    title="Select Agent Role to Configure",
                    subtitle=f"Compute Tier: {self.session.profile.tier.value.upper()} • Primary Compute: {gpu_name}",
                    items=role_items,
                    default_index=0,
                    cancel_label="Done / Exit (Back to REPL)",
                )

                if not role_choice:
                    break

                target_role = role_choice.lower().strip()
                await self._select_and_bind_model(target_role)

            return

        role_lower = role.lower().strip()
        if role_lower not in ("chat", "planner", "coder", "reviewer", "writer", "all"):
            console.print(
                f"[red]Invalid role '{role}'. Valid roles are: chat, planner, coder, reviewer, writer, all.[/red]"
            )
            return

        if not model_name:
            await self._select_and_bind_model(role_lower)
            return

        self._apply_model_binding(role_lower, model_name)

    async def _select_and_bind_model(self, role_lower: str) -> None:
        """Interactive model picker for a specific role."""
        if not await self.session.client.health_check():
            console.print(
                "[bold yellow][Notice] Ollama daemon is offline. Cannot inspect local models.[/bold yellow]"
            )
            custom_name = Prompt.ask(
                f"Enter model name manually for [bold yellow]{role_lower.upper()}[/bold yellow] (or press Enter to cancel)"
            ).strip()
            if not custom_name:
                return
            model_name = custom_name
        else:
            raw_models = await self.session.client.list_models()
            target_role = "coder" if role_lower == "all" else role_lower
            current_active = self.session.model_overrides.get(target_role, "")

            if raw_models:
                model_items = ModelCatalogHelper.classify_and_build_items(
                    installed_models=raw_models,
                    role=target_role,
                    current_model=current_active,
                )
            else:
                opt = ResourceOptimizer.get_params_for_profile(self.session.profile)
                model_items = [
                    SelectorItem(
                        key=opt.recommended_model,
                        label=opt.recommended_model,
                        specs=f"Recommended for {self.session.profile.tier.value.upper()} (Not yet pulled)",
                        badge="[Recommended]",
                        badge_style="bold green",
                    )
                ]

            gpu_desc = (
                f"{self.session.profile.primary_gpu.name} ({self.session.profile.total_vram_free_gb:.1f} GB free VRAM)"
                if self.session.profile.primary_gpu
                else "CPU Mode"
            )
            model_choice = TUISelector.choose(
                title=f"Select Model for {role_lower.upper()}",
                subtitle=f"Hardware: {gpu_desc} • Tier: {self.session.profile.tier.value.upper()}",
                items=model_items,
                default_index=0,
                allow_custom=True,
                custom_prompt="Enter custom model tag (e.g. qwen2.5:7b or deepseek-r1:7b)",
                cancel_label="Cancel",
            )

            if not model_choice:
                return
            model_name = model_choice

        self._apply_model_binding(role_lower, model_name)

    def _apply_model_binding(self, role_lower: str, model_name: str) -> None:
        """Apply model binding to session overrides and display feedback."""
        if role_lower == "all":
            for r in ("chat", "planner", "coder", "reviewer", "writer"):
                self.session.model_overrides[r] = model_name
                try:
                    ConfigManager.set_user_config_value(f"models.{r}", model_name)
                except Exception:
                    pass
            console.print(
                f"[bold green]All agent roles updated to model '[bold white]{model_name}[/bold white]' (saved to config).[/bold green]"
            )
        else:
            self.session.model_overrides[role_lower] = model_name
            try:
                ConfigManager.set_user_config_value(f"models.{role_lower}", model_name)
            except Exception:
                pass
            console.print(
                f"[bold green]Role '[bold white]{role_lower}[/bold white]' updated to model '[bold white]{model_name}[/bold white]' (saved to config).[/bold green]"
            )

    async def _cmd_mode(self, mode_str: str) -> None:
        """Inspect or switch agent operating mode (auto, chat, planner, coder, reviewer)."""
        mode_str = mode_str.lower().strip()
        # Backward compatibility for safety mode keywords
        if mode_str in ("balanced", "strict", "autonomous"):
            await self._cmd_safety(mode_str)
            return

        if not mode_str:
            current = self.session.agent_mode.value
            mode_items = [
                SelectorItem(
                    key="auto",
                    label="Auto (Intelligent Dynamic Routing)",
                    specs="Automatically classifies requests and switches between chat, planner, and coder",
                    badge="[Default / Recommended]",
                    badge_style="bold green",
                ),
                SelectorItem(
                    key="chat",
                    label="Chat (Conversational Assistant)",
                    specs="Direct single-model dialogue for Q&A, greetings, and conceptual guidance",
                    badge="[Fast / Low Overhead]",
                    badge_style="cyan",
                ),
                SelectorItem(
                    key="planner",
                    label="Planner (Architecture & Decomposition)",
                    specs="Formulates detailed task plans and prompts for user approval before coding",
                    badge="[Approval Gate]",
                    badge_style="magenta",
                ),
                SelectorItem(
                    key="coder",
                    label="Coder (Autonomous Implementation)",
                    specs="Executes code implementation, file edits, and tools",
                    badge="[Development]",
                    badge_style="yellow",
                ),
                SelectorItem(
                    key="reviewer",
                    label="Reviewer (Code Audit & Quality Gate)",
                    specs="Performs code inspection, syntax verification, and quality audit",
                    badge="[Audit]",
                    badge_style="blue",
                ),
                SelectorItem(
                    key="writer",
                    label="Writer (Technical Docs & Copywriting)",
                    specs="Drafts READMEs, documentation, specifications, and articles",
                    badge="[Writing]",
                    badge_style="bold cyan",
                ),
            ]
            default_idx = 0
            for idx, item in enumerate(mode_items):
                if item.key == current:
                    default_idx = idx
                    break

            mode_choice = TUISelector.choose(
                title="Select Agent Operating Mode",
                subtitle=f"Current Active Mode: {current.upper()}",
                items=mode_items,
                default_index=default_idx,
                cancel_label="Cancel (Keep Current)",
            )
            if not mode_choice:
                return
            mode_str = mode_choice

        try:
            new_mode = AgentMode(mode_str)
            self.session.agent_mode = new_mode
            try:
                ConfigManager.set_user_config_value("agent_mode", new_mode.value)
            except Exception:
                pass
            console.print(
                f"[bold green]Agent operating mode switched to: [bold white]{new_mode.value.upper()}[/bold white] (saved to config)[/bold green]"
            )
        except ValueError:
            console.print(
                f"[red]Invalid mode '{mode_str}'. Valid choices: auto, chat, planner, coder, reviewer, writer (or use /safety for tool safety).[/red]"
            )

    async def _cmd_safety(self, mode_str: str) -> None:
        """Inspect or switch human-in-the-loop safety confirmation mode with interactive TUI selector."""
        if not mode_str:
            current = self.session.security_gate.mode.value
            mode_items = [
                SelectorItem(
                    key="balanced",
                    label="Balanced",
                    specs="Prompt for shell commands & dangerous file modifications",
                    badge="[Default / Recommended]",
                    badge_style="bold green",
                ),
                SelectorItem(
                    key="strict",
                    label="Strict",
                    specs="Prompt for confirmation before EVERY tool execution",
                    badge="[Maximum Safety]",
                    badge_style="yellow",
                ),
                SelectorItem(
                    key="autonomous",
                    label="Autonomous",
                    specs="Execute all tools autonomously without prompting (Full Speed)",
                    badge="[High Speed]",
                    badge_style="magenta",
                ),
            ]
            default_idx = 0 if current == "balanced" else (1 if current == "strict" else 2)
            mode_choice = TUISelector.choose(
                title="Select Safety Confirmation Mode",
                subtitle=f"Current Active Mode: {current.upper()}",
                items=mode_items,
                default_index=default_idx,
                cancel_label="Cancel (Keep Current)",
            )
            if not mode_choice:
                return
            mode_str = mode_choice

        try:
            new_mode = SafetyMode(mode_str.lower().strip())
            self.session.security_gate.mode = new_mode
            try:
                ConfigManager.set_user_config_value("safety_mode", new_mode.value)
            except Exception:
                pass
            console.print(
                f"[bold green]Security mode switched to: [bold white]{new_mode.value.upper()}[/bold white] (saved to config)[/bold green]"
            )
        except ValueError:
            console.print(
                f"[red]Invalid safety mode '{mode_str}'. Valid choices: strict, balanced, autonomous.[/red]"
            )

    async def _cmd_workspace(self, arg1: str, arg2: str) -> None:
        """Manage active developer workspace directory."""
        sub = arg1.lower().strip()
        if not sub:
            ws = self.session.workspace_root
            files = list(ws.rglob("*")) if ws.exists() else []
            regular_files = [f for f in files if f.is_file()]
            total_bytes = sum(f.stat().st_size for f in regular_files)
            size_mb = total_bytes / (1024 * 1024)
            is_git = (ws / ".git").is_dir()

            table = Table(title="[Active Workspace Status]", border_style="cyan")
            table.add_column("Property", style="bold white", width=18)
            table.add_column("Value", style="cyan")
            table.add_row("Root Path", str(ws))
            table.add_row("Total Files", f"{len(regular_files):,}")
            table.add_row("Total Disk Size", f"{size_mb:.2f} MB")
            table.add_row(
                "Git Repository", "[bold green]Yes[/bold green]" if is_git else "[dim]No[/dim]"
            )

            console.print(table)
            console.print(
                "[dim]Commands: /workspace <path> (switch), /workspace list (show files), /workspace new <name> (create project)[/dim]\n"
            )
            return

        if sub in ("list", "files", "ls"):
            ws = self.session.workspace_root
            files = [f for f in ws.iterdir() if not f.name.startswith(".")] if ws.exists() else []
            if not files:
                console.print(f"[dim]Workspace {ws} is empty.[/dim]\n")
                return
            table = Table(title=f"[Workspace Files: {ws.name}]", border_style="cyan")
            table.add_column("Name", style="bold white")
            table.add_column("Type", style="yellow")
            table.add_column("Size", style="green")
            table.add_column("Modified", style="dim")
            for f in sorted(files, key=lambda x: (not x.is_dir(), x.name.lower())):
                ftype = "DIR" if f.is_dir() else f.suffix or "FILE"
                fsize = "-" if f.is_dir() else f"{f.stat().st_size:,} B"
                mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                table.add_row(f.name, ftype, fsize, mtime)
            console.print(table)
            console.print()
            return

        if sub in ("new", "create", "init"):
            name = arg2.strip()
            if not name:
                name = Prompt.ask("Enter new workspace folder name").strip()
            if not name:
                return
            target_path = self.session.workspace_root / name
            self.session.switch_workspace(target_path)
            console.print(
                f"[bold green]Created and switched to new workspace: [bold white]{target_path}[/bold white][/bold green]\n"
            )
            return

        # Otherwise, switch to specified path
        target_path = Path(arg1)
        if not target_path.is_absolute():
            target_path = (self.session.workspace_root / target_path).resolve()
        self.session.switch_workspace(target_path)
        console.print(
            f"[bold green]Switched active workspace to: [bold white]{target_path}[/bold white][/bold green]\n"
        )

    async def _cmd_session(self, subcommand: str = "", arg: str = "") -> None:
        """
        Unified interactive session manager.
        Handles resume, delete, prune, and listing with full Graphical TUI interfaces.
        """
        sub = subcommand.lower().strip()

        # Direct resume if argument looks like a session ID
        if sub.startswith("sess_"):
            await self._apply_resume_session(subcommand.strip())
            return

        # Direct subcommands
        if sub in ("resume", "load", "restore"):
            if arg.strip():
                await self._apply_resume_session(arg.strip())
            else:
                await self._interactive_pick_and_resume_session()
            return

        if sub in ("delete", "rm", "remove", "drop"):
            raw_arg = arg.strip()
            force = False
            for f in ("--yes", "-y", "--force", "-f"):
                if f" {f}" in raw_arg or raw_arg == f:
                    force = True
                    raw_arg = raw_arg.replace(f" {f}", "").replace(f, "").strip()

            if raw_arg:
                target_id = raw_arg
                if target_id == self.session.session_id:
                    console.print(
                        "[red]Cannot delete active running session. Switch or exit first.[/red]\n"
                    )
                    return
                if not force and not self._confirm_delete_session(target_id):
                    console.print(f"[dim]Deletion of session '{target_id}' cancelled.[/dim]\n")
                    return
                if self.session.sm.delete_session(target_id):
                    console.print(
                        f"[bold green]Session '{target_id}' deleted successfully.[/bold green]\n"
                    )
                else:
                    console.print(
                        f"[bold red]Session '{target_id}' not found or could not be deleted.[/bold red]\n"
                    )
            else:
                await self._interactive_pick_and_delete_session()
            return

        if sub in ("prune", "clean"):
            count = self.session.sm.prune_empty_sessions()
            console.print(f"[bold green]Pruned {count} empty ghost session(s).[/bold green]\n")
            return

        if sub in ("list", "show", "ls"):
            self._display_sessions_table()
            return

        # If an unknown argument was given
        if sub:
            meta = self.session.sm.get_meta(subcommand.strip())
            if meta:
                await self._apply_resume_session(subcommand.strip())
                return
            console.print(f"[yellow]Unknown session action: '{subcommand}'.[/yellow]")
            console.print("[dim]Usage: /session [resume|delete|prune|list|<session_id>][/dim]\n")
            return

        # No subcommand -> Launch Interactive Graphical Session Management Hub
        while True:
            sessions = self.session.sm.list_sessions(limit=30, filter_empty=True)
            removable_count = len([s for s in sessions if s.session_id != self.session.session_id])

            hub_items = [
                SelectorItem(
                    key="resume",
                    label="Resume Session",
                    specs=f"Restore conversation history & context ({len(sessions)} available)",
                    badge="[Resume]",
                    badge_style="bold green",
                ),
                SelectorItem(
                    key="delete",
                    label="Delete Session",
                    specs=f"Permanently remove a session from disk ({removable_count} removable)",
                    badge="[Delete]",
                    badge_style="bold red",
                ),
                SelectorItem(
                    key="list",
                    label="View Sessions Table",
                    specs="Display overview table of all saved sessions",
                    badge="[Overview]",
                    badge_style="cyan",
                ),
                SelectorItem(
                    key="prune",
                    label="Prune Ghost Sessions",
                    specs="Clean up empty sessions with zero dialogue or steps",
                    badge="[Prune]",
                    badge_style="yellow",
                ),
            ]

            action = TUISelector.choose(
                title="Session Management Hub",
                subtitle=f"Active Session: {self.session.session_id} • Total Saved: {len(sessions)}",
                items=hub_items,
                default_index=0,
                cancel_label="Exit Session Hub (Back to REPL)",
            )

            if not action:
                break

            if action == "resume":
                resumed = await self._interactive_pick_and_resume_session()
                if resumed:
                    break
            elif action == "delete":
                await self._interactive_pick_and_delete_session()
            elif action == "list":
                self._display_sessions_table(sessions)
            elif action == "prune":
                pruned = self.session.sm.prune_empty_sessions()
                console.print(f"[bold green]Pruned {pruned} empty ghost session(s).[/bold green]\n")

    async def _cmd_sessions(self, subcommand: str = "", arg: str = "") -> None:
        """Manage or list historical sessions (alias for /session, defaults to listing table)."""
        if not subcommand:
            self._display_sessions_table()
            return
        await self._cmd_session(subcommand, arg)

    async def _cmd_resume(self, session_id: str = "") -> None:
        """Resume a saved session (shortcut alias for /session resume)."""
        await self._cmd_session("resume", session_id)

    async def _interactive_pick_and_resume_session(self) -> bool:
        """Interactive Graphical TUI selector to choose and restore a prior session."""
        sessions = self.session.sm.list_sessions(limit=25, filter_empty=True)
        if not sessions:
            console.print("[yellow]No prior saved sessions found in Brain store.[/yellow]\n")
            return False

        session_items = [
            SelectorItem(
                key=s.session_id,
                label=s.session_id,
                specs=f"{s.goal[:38]}... • {datetime.fromtimestamp(s.created_at).strftime('%m-%d %H:%M')}",
                badge=f"[{s.status.upper()}] ({s.step_count} steps)",
                badge_style="green" if s.status == "completed" else "yellow",
            )
            for s in sessions
        ]

        chosen_sid = TUISelector.choose(
            title="Select Session to Resume",
            subtitle="Restore conversation history, memory checkpoints, and workspace context",
            items=session_items,
            default_index=0,
            allow_custom=True,
            custom_prompt="Enter custom session ID to resume",
            custom_label="Enter Custom Session ID...",
            custom_specs="Enter custom session ID to restore",
            cancel_label="Cancel",
        )
        if not chosen_sid:
            return False

        return await self._apply_resume_session(chosen_sid)

    def _confirm_delete_session(self, target_id: str) -> bool:
        """Present an explicit confirmation dialog before deleting a session."""
        meta = self.session.sm.get_meta(target_id)
        goal_text = f" ('{meta.goal[:36]}...')" if meta and meta.goal else ""
        try:
            confirm_choice = TUISelector.choose(
                title=f"Are you sure you want to delete session '{target_id}'?",
                subtitle=f"This will permanently erase session data{goal_text} from disk.",
                items=[
                    SelectorItem(
                        key="cancel",
                        label="Cancel (Keep Session)",
                        specs="Do not delete this session",
                        badge="[Keep]",
                        badge_style="bold green",
                    ),
                    SelectorItem(
                        key="confirm",
                        label=f"Delete Session '{target_id}'",
                        specs=f"Permanently remove '{target_id}'",
                        badge="[Delete]",
                        badge_style="bold red",
                    ),
                ],
                default_index=0,
                cancel_label="Cancel",
            )
            return confirm_choice == "confirm"
        except Exception:
            from rich.prompt import Confirm

            return Confirm.ask(
                f"Are you sure you want to delete session '{target_id}'?", default=False
            )

    async def _interactive_pick_and_delete_session(self) -> bool:
        """Interactive Graphical TUI selector to choose and permanently delete a session."""
        sessions = self.session.sm.list_sessions(limit=25, filter_empty=True)
        removable = [s for s in sessions if s.session_id != self.session.session_id]
        if not removable:
            console.print(
                "[yellow]No historical sessions available to delete (active session protected).[/yellow]\n"
            )
            return False

        del_items = [
            SelectorItem(
                key=s.session_id,
                label=s.session_id,
                specs=f"{s.goal[:38]}... • {datetime.fromtimestamp(s.created_at).strftime('%m-%d %H:%M')}",
                badge=f"[{s.status.upper()}] ({s.step_count} steps)",
                badge_style="red" if s.status == "failed" else "yellow",
            )
            for s in removable
        ]

        chosen = TUISelector.choose(
            title="Select Session to Delete",
            subtitle="Choose a session to permanently remove from disk",
            items=del_items,
            default_index=0,
            allow_custom=True,
            custom_prompt="Enter session ID to delete",
            custom_label="Enter Custom Session ID to Delete...",
            custom_specs="Specify exact session ID to remove",
            cancel_label="Cancel",
        )
        if not chosen:
            return False

        if chosen == self.session.session_id:
            console.print(
                "[red]Cannot delete active running session. Switch or exit first.[/red]\n"
            )
            return False

        if not self._confirm_delete_session(chosen):
            console.print(f"[dim]Deletion of session '{chosen}' cancelled.[/dim]\n")
            return False

        if self.session.sm.delete_session(chosen):
            console.print(
                f"[bold green]Deleted session: [bold white]{chosen}[/bold white][/bold green]\n"
            )
            return True
        else:
            console.print(
                f"[bold red]Failed to delete session: [bold white]{chosen}[/bold white][/bold red]\n"
            )
            return False

    def _display_sessions_table(self, sessions: list[Any] | None = None) -> None:
        """Render a formatted Rich Table of saved sessions."""
        if sessions is None:
            sessions = self.session.sm.list_sessions(limit=30, filter_empty=True)
        if not sessions:
            console.print("[dim]No historical sessions found.[/dim]\n")
            return

        table = Table(title="Historical Agent Sessions", border_style="cyan")
        table.add_column("Session ID", style="bold white")
        table.add_column("Created At", style="dim")
        table.add_column("Status", style="yellow")
        table.add_column("Steps", style="green", justify="right")
        table.add_column("Hardware Tier", style="magenta")
        table.add_column("Objective", style="white")

        for s in sessions:
            status_color = (
                "green"
                if s.status == "completed"
                else ("red" if s.status == "failed" else "yellow")
            )
            if isinstance(s.created_at, (int, float)):
                time_str = datetime.fromtimestamp(s.created_at).strftime("%Y-%m-%d %H:%M:%S")
            else:
                time_str = str(s.created_at)[:19].replace("T", " ")
            goal_short = (s.goal[:40] + "...") if len(s.goal) > 40 else s.goal
            is_current = (
                " [bold cyan](Active)[/bold cyan]"
                if s.session_id == self.session.session_id
                else ""
            )
            table.add_row(
                f"{s.session_id}{is_current}",
                time_str,
                f"[{status_color}]{s.status}[/{status_color}]",
                str(s.step_count),
                s.tier.value,
                goal_short,
            )

        console.print(table)
        console.print(
            "[dim]Tip: Use /session to launch interactive graphical session manager.[/dim]\n"
        )

    async def _apply_resume_session(self, session_id: str) -> bool:
        """Apply session restoration from Brain store into active REPL."""
        meta = self.session.sm.get_meta(session_id)
        if not meta:
            console.print(f"[red]Session '{session_id}' not found in Brain store.[/red]\n")
            return False

        session_dir = self.session.brain.get_session_dir(session_id)
        ckpt_store = CheckpointStore(session_dir / "checkpoints.db")
        target_state = ckpt_store.get_latest_checkpoint(session_id)
        if not target_state:
            target_state = AgentState(
                session_id=session_id,
                user_goal=meta.user_goal,
                workspace_dir=str(self.session.workspace_root),
                status=meta.status,
            )

        # Clean up prior empty session if it had no turns
        if self.session.session_id != session_id:
            old_sid = self.session.session_id
            if not self.session.history and getattr(self.session.meta, "step_count", 0) == 0:
                self.session.brain.clean_session(old_sid)

        # Restore session state into active REPL
        self.session.session_id = session_id
        self.session.meta = meta
        self.session.session_dir = session_dir
        self.session.checkpoint_store = ckpt_store
        self.session.artifact_store = ArtifactStore(session_dir)
        self.session.episodic_memory = EpisodicMemory(session_dir / "episodic.jsonl")
        self.session.current_state = target_state

        if target_state.messages:
            self.session.history = list(target_state.messages)
        if target_state.scratchpad:
            for k, v in target_state.scratchpad.items():
                self.session.working_memory.set(k, v)

        # Restore workspace artifacts
        if target_state.artifacts:
            for fname, content in target_state.artifacts.items():
                dest = self.session.workspace_root / fname
                try:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(content, encoding="utf-8")
                except Exception as e:
                    console.print(f"[dim yellow]Warning writing {fname}: {e}[/dim yellow]")

        console.print(
            Panel(
                f"[bold green]Successfully resumed session:[/bold green] [bold white]{session_id}[/bold white]\n"
                f"[bold cyan]Goal:[/bold cyan] {target_state.user_goal}\n"
                f"[dim]Step: {target_state.current_step} • Status: {target_state.status} • "
                f"Artifacts: {len(target_state.artifacts)} files • History: {len(self.session.history)} msgs[/dim]",
                title="Session Resumed",
                border_style="green",
            )
        )
        return True

    def _cmd_add(self, filepath: str) -> None:
        """Pin a file into active context memory."""
        if not filepath:
            console.print("[yellow]Usage: /add <relative_path_to_file>[/yellow]")
            return

        p = (self.session.workspace_root / filepath).resolve()
        try:
            p.relative_to(self.session.workspace_root.resolve())
        except ValueError:
            console.print(f"[red]Error: {filepath} is outside workspace.[/red]")
            return

        if not p.exists() or not p.is_file():
            console.print(f"[red]File not found: {filepath}[/red]")
            return

        if p.suffix.lower() in BINARY_EXTENSIONS:
            console.print(f"[yellow]Skipping binary file: {filepath}[/yellow]")
            return

        rel_path = p.relative_to(self.session.workspace_root.resolve()).as_posix()
        try:
            content = p.read_text(encoding="utf-8", errors="replace")
            self.session.pinned_files[rel_path] = content
            console.print(
                f"[green]Pinned '{rel_path}' ({len(content)} chars) into active context.[/green]"
            )
        except Exception as e:
            console.print(f"[red]Failed to read file: {e}[/red]")

    def _cmd_drop(self, filepath: str) -> None:
        """Unpin a file from active context."""
        rel_path = filepath.strip().lstrip("@")
        if rel_path in self.session.pinned_files:
            del self.session.pinned_files[rel_path]
            console.print(f"[green]Unpinned '{rel_path}'.[/green]")
        else:
            console.print(f"[yellow]'{rel_path}' was not in pinned files list.[/yellow]")

    def _cmd_context(self) -> None:
        """Display token budget and pinned files breakdown."""
        # Calculate tokens
        pinned_tokens = sum(
            ContextManager.estimate_tokens(c) for c in self.session.pinned_files.values()
        )
        history_tokens = ContextManager.estimate_messages_tokens(self.session.history)
        total_used = pinned_tokens + history_tokens
        max_tokens = self.session.current_num_ctx

        TUIRenderer.render_context_budget_bar(
            used_tokens=total_used,
            max_tokens=max_tokens,
            pinned_files_tokens=pinned_tokens,
        )

        if self.session.pinned_files:
            console.print("[bold cyan]Pinned Files in Context:[/bold cyan]")
            for path, text in self.session.pinned_files.items():
                tok = ContextManager.estimate_tokens(text)
                console.print(f"  • [white]{path}[/white] [dim](~{tok:,} tokens)[/dim]")
        else:
            console.print("[dim](No files currently pinned. Use /add <file> or @file)[/dim]")

    def _cmd_diff(self) -> None:
        """Display git diff of workspace."""
        if not (self.session.workspace_root / ".git").exists():
            console.print(
                "[dim]Workspace is not a git repository (run 'git init' to track diffs).[/dim]"
            )
            return
        try:
            res = subprocess.run(
                ["git", "diff", "--color=always"],
                cwd=self.session.workspace_root,
                capture_output=True,
                text=True,
            )
            if res.stdout.strip():
                console.print(
                    Panel(res.stdout, title="Git Working Tree Diff", border_style="yellow")
                )
            else:
                console.print("[dim]No uncommitted git changes detected in workspace.[/dim]")
        except Exception as e:
            console.print(f"[yellow]Could not retrieve git diff: {e}[/yellow]")

    def _cmd_undo(self) -> None:
        """Revert the most recent step via CheckpointStore."""
        if not self.session.checkpoint_store:
            console.print("[yellow]Checkpoint store is not active for this session.[/yellow]")
            return

        ckpts = self.session.checkpoint_store.list_checkpoints(self.session.session_id)
        if len(ckpts) < 2:
            console.print("[yellow]No prior checkpoint available to undo.[/yellow]")
            return

        target_ckpt_id = ckpts[-2]["checkpoint_id"]
        target_state = self.session.checkpoint_store.get_checkpoint_by_id(target_ckpt_id)
        if not target_state:
            console.print("[red]Failed to restore checkpoint state.[/red]")
            return

        # Delete orphaned files
        current_artifacts = (
            set(self.session.current_state.artifacts.keys())
            if self.session.current_state and self.session.current_state.artifacts
            else set()
        )
        target_artifacts = set(target_state.artifacts.keys()) if target_state.artifacts else set()
        for fname in current_artifacts - target_artifacts:
            dest = self.session.workspace_root / fname
            try:
                if dest.exists():
                    dest.unlink()
                    console.print(f"[dim yellow]Deleted orphaned file: {fname}[/dim yellow]")
            except Exception as e:
                console.print(
                    f"[dim yellow]Failed to delete orphaned file {fname}: {e}[/dim yellow]"
                )

        # Restore state artifacts
        self.session.current_state = target_state
        if target_state.artifacts:
            for fname, content in target_state.artifacts.items():
                dest = self.session.workspace_root / fname
                try:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(content, encoding="utf-8")
                except Exception:
                    pass

        console.print(
            f"[bold green]Successfully rolled back to Step {target_state.current_step} ({target_state.active_agent}).[/bold green]"
        )

    def _cmd_commit(self, message: str) -> None:
        """Stage and commit changes."""
        if not (self.session.workspace_root / ".git").exists():
            console.print(
                "[yellow]Notice: Workspace is not a git repository. Run 'git init' first.[/yellow]"
            )
            return
        msg = message.strip() or "feat(agent): update implementation via Aglibol Agent"
        try:
            subprocess.run(["git", "add", "-A"], cwd=self.session.workspace_root, check=True)
            res = subprocess.run(
                ["git", "commit", "-m", msg],
                cwd=self.session.workspace_root,
                capture_output=True,
                text=True,
            )
            if res.returncode != 0:
                console.print(
                    Panel(
                        res.stdout or res.stderr, title="Git Commit Warning", border_style="yellow"
                    )
                )
            else:
                console.print(
                    Panel(res.stdout or res.stderr, title="Git Commit Result", border_style="green")
                )
        except Exception as e:
            console.print(f"[red]Failed to commit: {e}[/red]")

    def _cmd_compact(self) -> None:
        """Manually trigger context compression."""
        old_count = len(self.session.history)
        self.session.history = ContextManager.compress_messages(
            self.session.history,
            max_context_tokens=self.session.current_num_ctx,
            headroom_ratio=0.75,
        )
        new_count = len(self.session.history)
        console.print(
            f"[bold green]Context compacted: messages reduced from {old_count} to {new_count}.[/bold green]"
        )

    def _cmd_workflow(self, workflow_name: str) -> None:
        """Inspect or switch workflow."""
        if not workflow_name:
            console.print(f"Active workflow: [bold white]{self.session.workflow_name}[/bold white]")
            console.print("[dim]Use /workflow <name|path> to switch.[/dim]")
            return
        self.session.workflow_name = workflow_name
        try:
            ConfigManager.set_user_config_value("workflow.name", workflow_name)
        except Exception:
            pass
        console.print(
            f"[bold green]Workflow switched to '{workflow_name}' (saved to config).[/bold green]"
        )

    def _cmd_init(self) -> None:
        """Initialize local project guidelines."""
        cfg_path = self.session.workspace_root / ".aglibol.yaml"
        agents_md = self.session.workspace_root / "AGENTS.md"

        if not cfg_path.exists():
            cfg_path.write_text(
                "# Aglibol Agent Project Configuration\nmodels:\n  planner: qwen2.5:7b\n  coder: qwen2.5-coder:7b\n  reviewer: qwen2.5:7b\n  writer: qwen2.5:7b\n",
                encoding="utf-8",
            )
            console.print("[green]Created .aglibol.yaml[/green]")

        if not agents_md.exists():
            agents_md.write_text(
                "# Agent Project Instructions\n\n- Build tool: python -m pytest\n- Coding style: PEP 8 with type hints\n",
                encoding="utf-8",
            )
            console.print("[green]Created AGENTS.md[/green]")

    async def _cmd_doctor(self) -> None:
        """Run system and dependencies diagnostics in interactive REPL."""
        from aglibol.cli.commands.doctor import run_diagnostics

        console.print("[cyan]Running Aglibol Doctor diagnostics...[/cyan]")
        results = await run_diagnostics(auto_fix=False)

        table = Table(title="System & Dependency Health Check", border_style="bright_blue")
        table.add_column("Category", style="bold white", width=12)
        table.add_column("Check Item", style="cyan", width=22)
        table.add_column("Status", width=10)
        table.add_column("Details", style="dim white")

        status_styles = {
            "PASS": "[bold green]PASS[/bold green]",
            "WARN": "[bold yellow]WARN[/bold yellow]",
            "FAIL": "[bold red]FAIL[/bold red]",
            "INFO": "[dim blue]INFO[/dim blue]",
        }
        for r in results:
            table.add_row(r.category, r.name, status_styles.get(r.status, r.status), r.details)
        console.print(table)

    async def _cmd_update(self) -> None:
        """Check for updates on GitHub and prompt user."""
        from aglibol.utils.updater import UpdateManager

        updater = UpdateManager()
        console.print("[cyan]Checking GitHub for updates...[/cyan]")
        info = updater.check_for_updates(force=True, timeout=4.0)

        if not info.is_newer:
            console.print(
                f"[bold green]You are on the latest version of Aglibol Agent (v{info.current_version}).[/bold green]"
            )
            return

        panel_content = (
            f"[bold yellow]A new version of Aglibol Agent is available![/bold yellow]\n\n"
            f"  • Current Version: [dim]v{info.current_version}[/dim]\n"
            f"  • Latest Version:  [bold green]v{info.latest_version}[/bold green]\n"
            f"  • Install Method:  [cyan]{updater.detect_install_method().upper()}[/cyan]\n"
        )
        console.print(Panel(panel_content, border_style="yellow"))
        ans = Prompt.ask("Would you like to upgrade now?", choices=["y", "n"], default="y")
        if ans.lower() == "y":
            updater.perform_upgrade()
