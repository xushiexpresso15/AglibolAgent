"""Smart prompt_toolkit auto-completer for Slash Commands, multi-level parameters, and @file mentions."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document

if TYPE_CHECKING:
    from aglibol.cli.interactive.repl import InteractiveSession

SLASH_COMMANDS = {
    "/help": "Show available commands and keyboard shortcuts",
    "/status": "Display live hardware specs, VRAM residency, and Ollama status",
    "/doctor": "Run system diagnostics, check Ollama, VRAM, and tools",
    "/update": "Check for updates on GitHub and self-upgrade",
    "/models": "List installed Ollama models with compatibility metrics",
    "/model": "Inspect or switch active model for Chat, Planner, Coder, Reviewer, or Writer",
    "/mode": "Switch agent operating mode (auto, chat, planner, coder, reviewer, writer)",
    "/safety": "Switch tool confirmation safety mode (strict, balanced, autonomous)",
    "/workspace": "Manage, list files, or switch active project workspace",
    "/session": "Session Hub: interactive resume, graphical delete, list, or prune",
    "/resume": "Restore state from a previous session (alias for /session resume)",
    "/sessions": "Manage historical agent sessions (alias for /session)",
    "/add": "Pin a file or folder into active agent context",
    "/drop": "Unpin a file from active agent context",
    "/context": "Inspect active token usage breakdown and pinned files",
    "/diff": "Show git/workspace diff of generated code changes",
    "/undo": "Rollback the most recent step or file changes via checkpoint",
    "/commit": "Create a git commit with auto-generated message",
    "/compact": "Compress conversation context and observation masking",
    "/workflow": "Switch active workflow definition (e.g. default, fast)",
    "/init": "Initialize local project config and AGENTS.md guidelines",
    "/clear": "Clear screen while preserving conversation history",
    "/exit": "Safely unload VRAM models and exit interactive session",
    "/quit": "Alias for /exit",
}

ROLES_METADATA = {
    "chat": "Conversational assistant for Q&A and concepts",
    "planner": "Planning & task decomposition agent",
    "coder": "Code implementation & tool execution agent",
    "reviewer": "Syntax, quality & security reviewer agent",
    "writer": "Technical documentation, articles & copywriting agent",
}

AGENT_MODES_METADATA = {
    "auto": "Dynamic intent routing between chat, planner, coder, and writer (Default)",
    "chat": "Conversational assistant for Q&A and concepts",
    "planner": "Architecture planning with user approval gate",
    "coder": "Autonomous code implementation and file edits",
    "reviewer": "Code inspection and syntax quality gate",
    "writer": "Technical documentation, articles & copywriting",
}

SAFETY_MODES_METADATA = {
    "balanced": "Ask before shell commands & dangerous file changes (Default)",
    "strict": "Ask before EVERY tool execution (Maximum security)",
    "autonomous": "Execute all tools autonomously without prompting",
}

WORKSPACE_COMMANDS_METADATA = {
    "list": "List all files in current workspace",
    "new": "Create and switch to a new workspace directory",
}

SESSION_COMMANDS_METADATA = {
    "resume": "Interactive TUI or direct session resumption (/session resume [id])",
    "delete": "Interactive TUI or direct session deletion (/session delete [id])",
    "list": "List all historical sessions stored in Brain (/session list)",
    "prune": "Purge all empty ghost sessions that contain no dialogue (/session prune)",
}


class AgentInputCompleter(Completer):
    """Provides dynamic multi-level completions for Slash Commands (/), parameters, and Workspace Files (@)."""

    def __init__(
        self,
        session: InteractiveSession | None = None,
        workspace_root: Path | None = None,
    ) -> None:
        self.session = session
        self.workspace_root = workspace_root or (session.workspace_root if session else Path.cwd())

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        text_before_cursor = document.text_before_cursor
        word_before_cursor = document.get_word_before_cursor(WORD=True)

        # --- 1. Slash Command & Parameter Multi-Level Auto-completion ---
        stripped = text_before_cursor.lstrip()
        if stripped.startswith("/"):
            # Check if we are completing the command itself or its arguments
            if " " not in stripped:
                # 1A. Top-level Slash command completion
                query = stripped.lower()
                for cmd, desc in SLASH_COMMANDS.items():
                    if cmd.startswith(query):
                        yield Completion(
                            text=cmd,
                            start_position=-len(stripped),
                            display=cmd,
                            display_meta=desc,
                        )
                return

            # 1B. Parameter-level completion
            parts = stripped.split()
            cmd = parts[0].lower()
            ends_with_space = stripped.endswith(" ")

            # Calculate argument index (1-based: arg 1, arg 2...)
            if ends_with_space:
                arg_idx = len(parts)
                current_prefix = ""
            else:
                arg_idx = len(parts) - 1
                current_prefix = parts[-1].lower()

            prefix_len = len(current_prefix)

            # --- /model [role] [model_name] ---
            if cmd == "/model":
                if arg_idx == 1:
                    for role, desc in ROLES_METADATA.items():
                        if role.startswith(current_prefix):
                            yield Completion(
                                text=role,
                                start_position=-prefix_len,
                                display=role,
                                display_meta=desc,
                            )
                    return
                elif arg_idx == 2:
                    # Suggest installed models
                    models = self._get_installed_models()
                    for m in models:
                        if m.lower().startswith(current_prefix):
                            yield Completion(
                                text=m,
                                start_position=-prefix_len,
                                display=m,
                                display_meta="Installed Ollama Model",
                            )
                    return

            # --- /mode [auto|chat|planner|coder|reviewer] ---
            elif cmd == "/mode":
                if arg_idx == 1:
                    all_modes = {**AGENT_MODES_METADATA, **SAFETY_MODES_METADATA}
                    for mode, desc in all_modes.items():
                        if mode.startswith(current_prefix):
                            yield Completion(
                                text=mode,
                                start_position=-prefix_len,
                                display=mode,
                                display_meta=desc,
                            )
                    return

            # --- /safety [balanced|strict|autonomous] ---
            elif cmd == "/safety":
                if arg_idx == 1:
                    for mode, desc in SAFETY_MODES_METADATA.items():
                        if mode.startswith(current_prefix):
                            yield Completion(
                                text=mode,
                                start_position=-prefix_len,
                                display=mode,
                                display_meta=desc,
                            )
                    return

            # --- /workspace [list|new|<path>] ---
            elif cmd == "/workspace":
                if arg_idx == 1:
                    for subcmd, desc in WORKSPACE_COMMANDS_METADATA.items():
                        if subcmd.startswith(current_prefix):
                            yield Completion(
                                text=subcmd,
                                start_position=-prefix_len,
                                display=subcmd,
                                display_meta=desc,
                            )
                    for rel_path, is_dir in self._find_workspace_files(current_prefix):
                        if is_dir:
                            yield Completion(
                                text=rel_path + "/",
                                start_position=-prefix_len,
                                display=rel_path + "/",
                                display_meta="Directory",
                            )
                    return

            # --- /resume [session_id] ---
            elif cmd == "/resume":
                if arg_idx == 1:
                    sessions = self._get_recent_sessions()
                    for s_id, s_goal in sessions:
                        if s_id.lower().startswith(current_prefix):
                            display_goal = (s_goal[:45] + "...") if len(s_goal) > 45 else s_goal
                            yield Completion(
                                text=s_id,
                                start_position=-prefix_len,
                                display=s_id,
                                display_meta=display_goal,
                            )
                    return

            # --- /sessions or /session [list|delete|prune] ---
            elif cmd in ("/sessions", "/session"):
                if arg_idx == 1:
                    for subcmd, desc in SESSION_COMMANDS_METADATA.items():
                        if subcmd.startswith(current_prefix):
                            yield Completion(
                                text=subcmd,
                                start_position=-prefix_len,
                                display=subcmd,
                                display_meta=desc,
                            )
                    sessions = self._get_recent_sessions()
                    for s_id, s_goal in sessions:
                        if s_id.lower().startswith(current_prefix):
                            display_goal = (s_goal[:45] + "...") if len(s_goal) > 45 else s_goal
                            yield Completion(
                                text=s_id,
                                start_position=-prefix_len,
                                display=s_id,
                                display_meta=display_goal,
                            )
                    return
                elif (
                    arg_idx == 2
                    and len(parts) > 1
                    and parts[1].lower() in ("delete", "rm", "remove", "resume", "load", "restore")
                ):
                    sessions = self._get_recent_sessions()
                    for s_id, s_goal in sessions:
                        if s_id.lower().startswith(current_prefix):
                            display_goal = (s_goal[:45] + "...") if len(s_goal) > 45 else s_goal
                            yield Completion(
                                text=s_id,
                                start_position=-prefix_len,
                                display=s_id,
                                display_meta=display_goal,
                            )
                    return

            # --- /drop [pinned_file] ---
            elif cmd == "/drop":
                if arg_idx == 1 and self.session:
                    for pinned in self.session.pinned_files.keys():
                        if pinned.lower().startswith(current_prefix):
                            yield Completion(
                                text=pinned,
                                start_position=-prefix_len,
                                display=pinned,
                                display_meta="Pinned File",
                            )
                    return

            # --- /add [workspace_file] ---
            elif cmd == "/add":
                if arg_idx == 1:
                    for rel_path, is_dir in self._find_workspace_files(current_prefix):
                        completion_text = rel_path + ("/" if is_dir else "")
                        display_meta = "Directory" if is_dir else "File"
                        yield Completion(
                            text=completion_text,
                            start_position=-prefix_len,
                            display=rel_path,
                            display_meta=display_meta,
                        )
                    return

            # --- /workflow [default|...] ---
            elif cmd == "/workflow":
                if arg_idx == 1:
                    workflows = ["default", "fast", "research", "single_turn"]
                    for wf in workflows:
                        if wf.startswith(current_prefix):
                            yield Completion(
                                text=wf,
                                start_position=-prefix_len,
                                display=wf,
                                display_meta="Workflow Template",
                            )
                    return

        # --- 2. File Mention Auto-completion (@path/to/file) Anywhere ---
        if "@" in word_before_cursor:
            at_idx = word_before_cursor.rfind("@")
            file_query = word_before_cursor[at_idx + 1 :]
            start_pos = -(len(file_query) + 1)

            # Search workspace files
            for rel_path, is_dir in self._find_workspace_files(file_query):
                completion_text = f"@{rel_path}" + ("/" if is_dir else " ")
                display_meta = "Directory" if is_dir else "File"
                yield Completion(
                    text=completion_text,
                    start_position=start_pos,
                    display=f"@{rel_path}",
                    display_meta=display_meta,
                )

    def _get_installed_models(self) -> list[str]:
        """Fetch list of installed models from session cache or fallback defaults."""
        if self.session and getattr(self.session, "installed_models_cache", None):
            return self.session.installed_models_cache
        return [
            "qwen2.5-coder:7b",
            "qwen2.5:7b",
            "glm4:9b",
            "ornith-1.5:9b",
            "gemma2:9b",
            "llama3.1:8b",
        ]

    def _get_recent_sessions(self) -> list[tuple[str, str]]:
        """Retrieve recent session IDs and their goals from session manager."""
        if self.session and getattr(self.session, "sm", None):
            try:
                sessions = self.session.sm.list_sessions(limit=10)
                return [(s.session_id, s.goal) for s in sessions]
            except Exception:
                pass
        try:
            from aglibol.core.config import ConfigManager
            from aglibol.storage.brain import Brain
            from aglibol.storage.session import SessionManager

            cfg = ConfigManager.load()
            brain = Brain(cfg.storage.brain_dir)
            sm = SessionManager(brain.sessions_dir)
            return [(s.session_id, s.goal) for s in sm.list_sessions(limit=10)]
        except Exception:
            return []

    def _find_workspace_files(
        self, query: str, max_depth: int = 4, max_scanned: int = 3000
    ) -> list[tuple[str, bool]]:
        """Find matching files and directories in workspace, ignoring heavy build/git folders."""
        matches: list[tuple[str, bool]] = []
        ignored_dirs = {
            ".git",
            "__pycache__",
            ".pytest_cache",
            ".venv",
            "venv",
            "node_modules",
            ".idea",
            ".vscode",
            "dist",
            "build",
            ".aglibol",
        }

        q = query.lower().replace("\\", "/")
        try:
            root_depth = len(self.workspace_root.parts)
            scanned = 0
            for root, dirs, files in os.walk(self.workspace_root):
                scanned += len(dirs) + len(files)
                current_depth = len(Path(root).parts) - root_depth

                if current_depth > max_depth or scanned > max_scanned:
                    dirs[:] = []
                    continue

                # Prune ignored directories in-place
                dirs[:] = [d for d in dirs if d not in ignored_dirs and not d.startswith(".")]

                rel_root = Path(root).relative_to(self.workspace_root).as_posix()
                if rel_root == ".":
                    rel_root = ""

                # Suggest directories
                for d in dirs:
                    rel_d = f"{rel_root}/{d}" if rel_root else d
                    if q in rel_d.lower():
                        matches.append((rel_d, True))
                        if len(matches) >= 15:
                            return matches

                # Suggest files
                for f in files:
                    rel_f = f"{rel_root}/{f}" if rel_root else f
                    if q in rel_f.lower():
                        matches.append((rel_f, False))
                        if len(matches) >= 15:
                            return matches
        except Exception:
            pass

        return matches
