"""Rich terminal presentation and live streaming renderer for the interactive CLI."""

from __future__ import annotations

import re
import sys
import time
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

console = Console()


class LiveThoughtBox:
    """Manages an animated live 2-line thinking box with spinner and live scrolling."""

    def __init__(self, agent_name: str = "Assistant") -> None:
        self.agent_name = agent_name
        self.lines: list[str] = []
        self._current_line = ""
        self._live: Live | None = None
        self._spinner = Spinner("dots", style="bold cyan")
        self._status = "Thinking and formulating actions..."
        self._is_finished: bool = False
        self._final_summary: str = ""

    def start(self, initial_text: str = "Thinking and formulating actions...") -> None:
        """Start the live animated display."""
        self._status = initial_text
        self._is_finished = False
        self._final_summary = ""
        try:
            self._live = Live(
                get_renderable=self._build_panel,
                console=console,
                refresh_per_second=8,
                transient=False,
            )
            self._live.start()
        except Exception:
            self._live = None

    def update_status(self, status: str) -> None:
        """Update the active in-progress status label."""
        self._status = status

    def update_token(self, token: str) -> None:
        """Append streaming thought token to live buffer."""
        for ch in token:
            if ch == "\n":
                if self._current_line.strip():
                    self.lines.append(self._current_line.strip())
                self._current_line = ""
            else:
                self._current_line += ch

    def finish(self, summary: str = "") -> None:
        """Finalize and close live display."""
        if self._current_line.strip():
            self.lines.append(self._current_line.strip())
        self._is_finished = True
        self._final_summary = summary
        if self._live:
            try:
                self._live.refresh()
                self._live.stop()
            except Exception:
                pass
            finally:
                self._live = None

    def _build_panel(self, status: str = "", final: bool | None = None, summary: str = "") -> Panel:
        is_final = self._is_finished if final is None else final
        if not is_final:
            spinner_frame = self._spinner.render(time.time())
            title = Text.assemble(
                (f"[{self.agent_name.capitalize()} Thinking ", "bold cyan"),
                spinner_frame,
                ("]", "bold cyan"),
            )
        else:
            title = Text(f"[{self.agent_name.capitalize()} Thinking ✓]", style="dim cyan")

        border_style = "dim cyan" if is_final else "dim blue"

        visible_lines = list(self.lines)
        if self._current_line.strip() and not is_final:
            visible_lines.append(self._current_line.strip())

        active_status = status or self._status
        active_summary = summary or self._final_summary

        if active_summary and is_final:
            line1 = active_summary[:90]
            line2 = "[dim](Thinking completed)[/dim]"
        elif not visible_lines:
            line1 = f"{active_status or 'Analyzing requirements and planning...'}"[:90]
            line2 = "[dim](Processing step...)[/dim]"
        elif len(visible_lines) == 1:
            line1 = visible_lines[0][:90]
            line2 = "[dim](Formulating plan and next actions...)[/dim]"
        elif len(visible_lines) == 2:
            line1 = visible_lines[0][:90]
            line2 = visible_lines[1][:90]
        else:
            earlier = len(visible_lines) - 2
            line1 = f"[dim]({earlier} earlier thoughts...)[/dim] {visible_lines[-2]}"[:90]
            line2 = visible_lines[-1][:90]

        return Panel(
            f"{line1}\n{line2}",
            title=title,
            border_style=border_style,
            padding=(0, 1),
            height=4,
        )


class ToolExecutionEntry:
    """Tracks a single tool execution status and details."""

    def __init__(self, tool_name: str, args_str: str) -> None:
        self.tool_name = tool_name
        self.args_str = args_str
        self.status = "running"
        self.success: bool | None = None
        self.output: str = ""
        self.error: str | None = None


class LiveToolContainer:
    """
    Consolidated single-dialog container for tool executions.
    All consecutive tools in a step share a single dialog box.
    Each tool occupies exactly one line with a live animation while running,
    which turns into a checkmark (✓) upon completion.
    """

    def __init__(self) -> None:
        self.entries: list[ToolExecutionEntry] = []
        self._live: Live | None = None
        self._spinner = Spinner("dots", style="bold cyan")
        self._is_finished = False

    def start(self) -> None:
        """Start the live container display."""
        if self._live is not None:
            return
        try:
            self._live = Live(
                get_renderable=self._build_panel,
                console=console,
                refresh_per_second=8,
                transient=False,
            )
            self._live.start()
        except Exception:
            self._live = None

    def add_tool_start(self, tool_name: str, args: dict[str, Any] | str) -> None:
        """Add a newly started tool as a running line."""
        if isinstance(args, dict):
            parts = [f"{k}={v}" for k, v in args.items() if v is not None]
            args_str = ", ".join(parts)
        else:
            args_str = str(args or "").strip()
        if len(args_str) > 75:
            args_str = args_str[:72] + "..."

        self.entries.append(ToolExecutionEntry(tool_name=tool_name, args_str=args_str))
        if self._live is None:
            self.start()

    def update_tool_finish(
        self,
        tool_name: str,
        success: bool,
        output: str = "",
        error: str | None = None,
    ) -> None:
        """Mark a tool as finished (success checkmark or failure cross)."""
        target: ToolExecutionEntry | None = None
        for entry in reversed(self.entries):
            if entry.tool_name == tool_name and entry.status == "running":
                target = entry
                break
        if not target and self.entries:
            target = self.entries[-1]

        if target:
            target.status = "done"
            target.success = success
            target.output = output
            target.error = error

    def finish(self) -> None:
        """Finalize and close the live tool container."""
        self._is_finished = True
        if self._live:
            try:
                self._live.refresh()
                self._live.stop()
            except Exception:
                pass
            finally:
                self._live = None

    def _build_panel(self) -> Panel:
        all_done = bool(self.entries) and all(e.status == "done" for e in self.entries)
        title_suffix = " ✓" if (self._is_finished or all_done) else ""
        title = f"[Tools Execution{title_suffix}]"
        border_style = "green" if (self._is_finished or all_done) else "cyan"

        lines: list[Text] = []
        spinner_frame = self._spinner.render(time.time())

        for entry in self.entries:
            line = Text()
            if entry.status == "running":
                line.append_text(spinner_frame)
                line.append(f" {entry.tool_name}", style="bold cyan")
                if entry.args_str:
                    line.append(f"({entry.args_str})", style="dim")
                line.append("  [running]", style="bold yellow")
            else:
                if entry.success is False:
                    line.append("✗ ", style="bold red")
                    line.append(f"{entry.tool_name}", style="bold white")
                    if entry.args_str:
                        line.append(f"({entry.args_str})", style="dim")
                    err_hint = f" [{entry.error or 'Failed'}]"
                    line.append(err_hint[:35], style="bold red")
                else:
                    line.append("✓ ", style="bold green")
                    line.append(f"{entry.tool_name}", style="bold white")
                    if entry.args_str:
                        line.append(f"({entry.args_str})", style="dim")
                    line.append("  [Done]", style="dim green")
            lines.append(line)

        if not lines:
            body: Any = Text("Initializing tool execution...", style="dim")
        else:
            body = Text("\n").join(lines)

        return Panel(
            body,
            title=title,
            border_style=border_style,
            padding=(0, 1),
        )


class LiveActivityView:
    """
    Unified real-time activity and progress display for an agent turn.
    Manages a single live-updating container displaying:
    - Active agent role and phase with animated spinner
    - Streaming thought / reasoning summary (compact 1-2 lines)
    - Live tool executions feed (compact 1 line per tool, spinner -> checkmark)
    - Status notices (e.g. model switching or critique loop status)
    """

    def __init__(self) -> None:
        self.current_agent: str = "assistant"
        self.current_phase: str = "Processing request..."
        self.current_thought: str = ""
        self.current_thought_topic: str = ""
        self.is_thinking: bool = True
        self._rolling_window: str = ""
        self._thought_buffer: list[str] = []
        self._current_thought_line: str = ""
        self.status_notice: str = ""
        self.tool_entries: list[ToolExecutionEntry] = []
        self._live: Live | None = None
        self._spinner = Spinner("dots", style="bold cyan")
        self._is_finished: bool = False
        self._final_summary: str = ""
        self.review_decision: str | None = None

    def start(self, agent_name: str = "assistant", initial_phase: str = "Working...") -> None:
        """Start the single unified live activity view."""
        self.current_agent = agent_name
        self.current_phase = initial_phase
        self._is_finished = False
        if self._live is not None:
            return
        try:
            self._live = Live(
                get_renderable=self._build_panel,
                console=console,
                refresh_per_second=8,
                transient=False,
            )
            self._live.start()
        except Exception:
            self._live = None

    def set_agent(self, agent_name: str, phase: str = "") -> None:
        """Transition active agent and phase in-place without creating a new container."""
        self.current_agent = agent_name
        if phase:
            self.current_phase = phase
        self._thought_buffer.clear()
        self._current_thought_line = ""
        self.current_thought = ""
        self.current_thought_topic = ""
        self._rolling_window = ""
        self.is_thinking = True

    def set_phase(self, phase: str) -> None:
        """Update active phase description."""
        self.current_phase = phase

    def set_notice(self, notice: str) -> None:
        """Set a transient status notice (e.g. model loaded or review feedback)."""
        self.status_notice = notice

    def set_task_progress(self, current: int, total: int, task_title: str) -> None:
        """Update active subtask progress in the activity view."""
        self.status_notice = f"Task [{current}/{total}]: {task_title}"

    def update_thought(
        self,
        token_or_text: str,
        is_thinking: bool = True,
        agent_name: str | None = None,
    ) -> None:
        """Append streaming thought or drafting tokens and maintain dynamic rolling ticker across all modes."""
        if agent_name and agent_name != self.current_agent:
            self.current_agent = agent_name

        self.is_thinking = is_thinking

        # Strip think tags from display stream
        clean_chunk = re.sub(r"</?(?:think|thinking)>", "", token_or_text)
        if not clean_chunk:
            return

        for ch in clean_chunk:
            if ch == "\n":
                line = self._current_thought_line.strip()
                if line:
                    self._thought_buffer.append(line)
                    if re.match(
                        r"^(\d+[\.\)]|\#+|Step \d+|Task \d+|Phase \d+|[-*])\s*", line, re.IGNORECASE
                    ) or (len(line) < 50 and line.endswith(":")):
                        self.current_thought_topic = line
                self._current_thought_line = ""
            else:
                self._current_thought_line += ch

        # Append to rolling window for dynamic real-time ticker
        self._rolling_window += clean_chunk
        if len(self._rolling_window) > 400:
            self._rolling_window = self._rolling_window[-300:]

        active_line = self._current_thought_line.strip()
        if active_line:
            self.current_thought = active_line
        elif self._thought_buffer:
            self.current_thought = self._thought_buffer[-1]

    def add_tool_start(self, tool_name: str, args: dict[str, Any] | str) -> None:
        """Record and display a newly started tool execution line."""
        if isinstance(args, dict):
            parts = [f"{k}={v}" for k, v in args.items() if v is not None]
            args_str = ", ".join(parts)
        else:
            args_str = str(args or "").strip()
        if len(args_str) > 75:
            args_str = args_str[:72] + "..."

        self.tool_entries.append(ToolExecutionEntry(tool_name=tool_name, args_str=args_str))
        if self._live is None:
            self.start(self.current_agent, self.current_phase)

    def update_tool_finish(
        self,
        tool_name: str,
        success: bool,
        output: str = "",
        error: str | None = None,
    ) -> None:
        """Update status of a finished tool."""
        target: ToolExecutionEntry | None = None
        for entry in reversed(self.tool_entries):
            if entry.tool_name == tool_name and entry.status == "running":
                target = entry
                break
        if not target and self.tool_entries:
            target = self.tool_entries[-1]

        if target:
            target.status = "done"
            target.success = success
            target.output = output
            target.error = error

    def set_review_outcome(self, decision: str) -> None:
        """Update review decision outcome."""
        self.review_decision = decision.lower()
        if self.review_decision == "approved":
            self.status_notice = "Review: Approved ✓"
        else:
            self.status_notice = f"Review: {decision.upper()} - Revisions needed"

    def finish(self, summary: str = "") -> None:
        """Finalize the unified live activity view into a clean compact completion card."""
        self._is_finished = True
        self._final_summary = summary
        if self._live:
            try:
                self._live.refresh()
                self._live.stop()
            except Exception:
                pass
            finally:
                self._live = None

    def _build_panel(self) -> Panel:
        if not self._is_finished:
            spinner_frame = self._spinner.render(time.time())
            title = Text.assemble(
                ("Aglibol ", "bold cyan"),
                ("• ", "dim"),
                (f"{self.current_agent.capitalize()} ", "bold white"),
                spinner_frame,
                (f" {self.current_phase}", "dim cyan"),
            )
            border_style = "cyan"
        else:
            title = Text.assemble(
                ("Aglibol ", "bold cyan"),
                ("• ", "dim"),
                ("Execution Complete ✓", "bold green"),
            )
            border_style = "green" if self.review_decision != "rejected" else "yellow"

        lines: list[Text] = []

        if not self._is_finished:
            # Determine unified dynamic action label based on agent and mode
            agent_lower = self.current_agent.lower()
            if self.is_thinking:
                action_label = "Thinking"
            else:
                if agent_lower == "coder":
                    action_label = "Implementing"
                elif agent_lower == "planner":
                    action_label = "Formulating Plan"
                elif agent_lower == "reviewer":
                    action_label = "Auditing"
                elif agent_lower == "writer":
                    action_label = "Drafting"
                else:
                    action_label = "Generating"

            # 1. Milestone / Topic Header (if detected)
            if self.current_thought_topic:
                top_text = self.current_thought_topic
                if len(top_text) > 75:
                    top_text = top_text[:72] + "..."
                top_line = Text()
                top_line.append("• ", style="dim cyan")
                top_line.append(top_text, style="dim white")
                lines.append(top_line)

            # 2. Dynamic rolling ticker showing the latest words
            # Clean JSON syntax framing noise for structured modes (Planner/Reviewer)
            raw_text = (
                self._rolling_window.strip()
                if self._rolling_window.strip()
                else (self.current_thought or self.current_phase)
            )
            cleaned_text = re.sub(
                r'["\'](?:summary|feedback|description|tasks|title|reasons)["\']\s*:\s*["\']?',
                "",
                raw_text,
            )
            cleaned_text = re.sub(r"[{}[\]]", "", cleaned_text)
            clean_stream = " ".join(cleaned_text.split())

            max_w = 78
            if len(clean_stream) > max_w:
                rolling_thought = "..." + clean_stream[-(max_w - 3) :]
            else:
                rolling_thought = clean_stream

            stream_line = Text()
            stream_line.append("› ", style="bold cyan")
            stream_line.append(f"{action_label}: ", style="bold cyan")
            stream_line.append(rolling_thought, style="italic cyan")
            lines.append(stream_line)

        if self.status_notice:
            notice_line = Text()
            notice_line.append("ℹ ", style="dim yellow")
            notice_line.append(self.status_notice, style="dim yellow")
            lines.append(notice_line)

        if len(self.tool_entries) > 5:
            skipped = len(self.tool_entries) - 5
            lines.append(Text(f"  ... ({skipped} earlier tool calls completed)", style="dim"))
            visible_tools = self.tool_entries[-5:]
        else:
            visible_tools = self.tool_entries

        for entry in visible_tools:
            tline = Text()
            if entry.status == "running":
                tline.append("  ", style="default")
                tline.append_text(self._spinner.render(time.time()))
                tline.append(f" {entry.tool_name}", style="bold cyan")
                if entry.args_str:
                    tline.append(f"({entry.args_str})", style="dim")
                tline.append("  [running]", style="bold yellow")
            else:
                if entry.success is False:
                    tline.append("  ✗ ", style="bold red")
                    tline.append(f"{entry.tool_name}", style="bold white")
                    if entry.args_str:
                        tline.append(f"({entry.args_str})", style="dim")
                    err_hint = f" [{entry.error or 'Failed'}]"
                    tline.append(err_hint[:35], style="bold red")
                else:
                    tline.append("  ✓ ", style="bold green")
                    tline.append(f"{entry.tool_name}", style="bold white")
                    if entry.args_str:
                        tline.append(f"({entry.args_str})", style="dim")
                    tline.append("  [Done]", style="dim green")
            lines.append(tline)

        if self._is_finished and not lines:
            lines.append(Text("✓ Turn completed successfully.", style="dim green"))

        body = Text("\n").join(lines) if lines else Text("Initializing...", style="dim")
        return Panel(
            body,
            title=title,
            border_style=border_style,
            padding=(0, 1),
        )


class TUIRenderer:
    """Renders formatted interactive responses, diffs, tool cards, and live progress."""

    @staticmethod
    def create_live_activity_view() -> LiveActivityView:
        """Factory method for unified animated live activity view."""
        return LiveActivityView()

    @staticmethod
    def render_code_artifact(filename: str, code: str, title: str | None = None) -> None:
        """Render a generated code artifact with syntax highlighting and line numbers."""
        from pathlib import Path

        if not code.strip():
            return
        ext = Path(filename).suffix.lstrip(".").lower() or "python"
        lexer = ext
        if ext in ("py", "pyw"):
            lexer = "python"
        elif ext in ("js", "ts", "jsx", "tsx"):
            lexer = "javascript"
        elif ext in ("sh", "bash", "zsh"):
            lexer = "bash"
        elif ext in ("json", "yml", "yaml", "toml", "md", "html", "css", "sql"):
            lexer = ext

        syntax = Syntax(
            code.strip(),
            lexer,
            theme="monokai",
            line_numbers=True,
            word_wrap=True,
        )
        box_title = title or f"[Artifact: {filename}]"
        console.print()
        console.print(Panel(syntax, title=box_title, border_style="cyan", padding=(0, 1)))

    @staticmethod
    def create_live_thought_box(agent_name: str = "Assistant") -> LiveThoughtBox:
        """Factory method for animated live thought boxes."""
        return LiveThoughtBox(agent_name)

    @staticmethod
    def create_live_tool_container() -> LiveToolContainer:
        """Factory method for consolidated animated live tool container."""
        return LiveToolContainer()

    @staticmethod
    def render_welcome_banner(session_id: str, tier: str, gpu_name: str, workspace: str) -> None:
        """Display welcoming banner upon entering interactive mode."""
        banner_content = Text()
        banner_content.append("Aglibol Agent ", style="bold cyan")
        banner_content.append("• Autonomous Multi-Agent Developer CLI\n", style="bold white")
        banner_content.append(
            f"Session: {session_id}  |  Tier: {tier.upper()} ({gpu_name})\n", style="dim"
        )
        banner_content.append(f"Workspace: {workspace}\n\n", style="dim")
        banner_content.append("Type your objective, or use ", style="white")
        banner_content.append("/help", style="bold yellow")
        banner_content.append(" for commands, ", style="white")
        banner_content.append("@file", style="bold cyan")
        banner_content.append(" to attach context.\n", style="white")
        banner_content.append("Tip: Use ", style="dim")
        banner_content.append("/session", style="bold green")
        banner_content.append(" to resume or manage conversation history.", style="dim")

        console.print(Panel(banner_content, border_style="bright_blue", padding=(1, 2)))

    @staticmethod
    def render_thought_box(content: str, agent_name: str = "Assistant") -> None:
        """Render intermediate agent thinking in a compact 2-line box."""
        if not content or not content.strip():
            return

        import re

        m = re.search(r"<(?:think|thinking)>([\s\S]*?)</(?:think|thinking)>", content)
        if m:
            thought = m.group(1).strip()
        elif content.strip().startswith(("{", "[", "def ", "class ", "import ", "from ", "#!")) or (
            content.strip().endswith("}") and "{" in content
        ):
            return
        else:
            thought = content.strip()

        thought = re.sub(r"</?(?:think|thinking)>", "", thought).strip()
        if not thought:
            return

        lines = [line.strip() for line in thought.splitlines() if line.strip()]
        if not lines:
            return

        if len(lines) > 2:
            earlier_count = len(lines) - 2
            line1 = f"[dim]({earlier_count} earlier thoughts...)[/dim] {lines[-2]}"
            if len(line1) > 90:
                line1 = line1[:87] + "..."
            line2 = lines[-1]
            if len(line2) > 90:
                line2 = line2[:87] + "..."
            display_text = f"{line1}\n{line2}"
        elif len(lines) == 2:
            display_text = f"{lines[0][:90]}\n{lines[1][:90]}"
        else:
            display_text = f"{lines[0][:90]}\n[dim](Formulating plan and next actions...)[/dim]"

        console.print(
            Panel(
                display_text,
                title=f"[{agent_name.capitalize()} Thinking]",
                border_style="dim blue",
                padding=(0, 1),
                height=4,
            )
        )

    @staticmethod
    def render_thought_card(content: str, agent_name: str = "coder") -> None:
        """Backward-compatible alias for render_thought_box."""
        TUIRenderer.render_thought_box(content, agent_name)

    @staticmethod
    def render_tool_box(
        tool_name: str,
        details: str | dict[str, Any],
        status: str = "running",
        success: bool | None = None,
    ) -> None:
        """Render tool execution in a compact 2-line box."""
        if isinstance(details, dict):
            parts = [f"{k}={v}" for k, v in details.items() if v]
            arg_str = ", ".join(parts)
        else:
            arg_str = str(details).strip()

        if len(arg_str) > 75:
            arg_str = arg_str[:72] + "..."

        line1 = (
            f"[bold cyan]Action:[/bold cyan] {tool_name}({arg_str})"
            if arg_str
            else f"[bold cyan]Action:[/bold cyan] {tool_name}"
        )

        if success is True:
            status_badge = "[bold green][OK][/bold green]"
            border_style = "green"
            line2 = f"[bold]Status:[/bold] {status_badge} {status}"
        elif success is False:
            status_badge = "[bold red][FAILED][/bold red]"
            border_style = "red"
            line2 = f"[bold]Status:[/bold] {status_badge} {status}"
        else:
            status_badge = f"[bold yellow][{status.upper()}][/bold yellow]"
            border_style = "cyan"
            line2 = f"[bold]Status:[/bold] {status_badge} In progress..."

        if len(line2) > 90:
            line2 = line2[:87] + "..."

        display_text = f"{line1}\n{line2}"

        console.print(
            Panel(
                display_text,
                title=f"[Tool: {tool_name}]",
                border_style=border_style,
                padding=(0, 1),
                height=4,
            )
        )

    @staticmethod
    def render_tool_start(tool_name: str, args: dict[str, Any]) -> None:
        """Render compact 2-line box when a tool starts."""
        TUIRenderer.render_tool_box(tool_name, details=args, status="running")

    @staticmethod
    def render_tool_finish(
        tool_name: str, success: bool, output: str, error: str | None = None
    ) -> None:
        """Render compact 2-line box when a tool finishes."""
        status_msg = (error or output or "Done").strip()
        if len(status_msg) > 60:
            status_msg = status_msg[:57] + "..."
        TUIRenderer.render_tool_box(tool_name, details="", status=status_msg, success=success)

    @staticmethod
    def confirm_mode_switch(
        current_mode: str,
        current_model: str,
        target_mode: str,
        target_model: str,
        reason: str = "Objective requires specialized capabilities",
        auto_approve: bool = False,
    ) -> bool:
        """Display mode switch confirmation UI and prompt user for approval."""
        import sys

        table = Table.grid(padding=(0, 1))
        table.add_column(style="bold white", width=14)
        table.add_column()

        table.add_row(
            "Current Mode:",
            f"[cyan]{current_mode.upper()}[/cyan] (Model: [dim]{current_model}[/dim])",
        )
        table.add_row(
            "Target Mode:",
            f"[bold yellow]{target_mode.upper()}[/bold yellow] (Model: [bold magenta]{target_model}[/bold magenta])",
        )
        table.add_row("Reason:", f"[dim italic]{reason}[/dim italic]")

        console.print()
        console.print(
            Panel(
                table,
                title="[Mode Switch Confirmation]",
                border_style="yellow",
                padding=(0, 2),
            )
        )

        if auto_approve or not sys.stdin.isatty():
            console.print(
                "  [dim yellow]Auto-approved mode switch (non-interactive environment)[/dim yellow]\n"
            )
            return True

        from rich.prompt import Confirm

        try:
            return Confirm.ask(
                f"  Switch to [bold yellow]{target_mode.upper()}[/bold yellow] mode?",
                default=True,
                console=console,
            )
        except Exception:
            return True

    @staticmethod
    def render_diff(filename: str, old_code: str, new_code: str) -> None:
        """Render colored unified diff between old and new file content."""
        import difflib

        diff_lines = list(
            difflib.unified_diff(
                old_code.splitlines(keepends=True),
                new_code.splitlines(keepends=True),
                fromfile=f"a/{filename}",
                tofile=f"b/{filename}",
                n=3,
            )
        )
        if not diff_lines:
            console.print(f"[dim]No changes detected in {filename}.[/dim]")
            return

        diff_text = "".join(diff_lines)
        syntax = Syntax(diff_text, "diff", theme="monokai", line_numbers=False)
        console.print(Panel(syntax, title=f"[Diff] {filename}", border_style="yellow"))

    @staticmethod
    def render_markdown(text: str) -> None:
        """Render standard GitHub flavored markdown with syntax highlighting."""
        console.print(Markdown(text))

    @staticmethod
    def render_assistant_response(content: str, title: str = "Assistant") -> None:
        """Render direct conversational response from assistant."""
        if not content.strip():
            return
        console.print()
        console.print(
            Panel(
                Markdown(content.strip()),
                title=f"[{title}]",
                border_style="cyan",
                padding=(1, 2),
            )
        )
        console.print()

    @staticmethod
    def render_plan_card(plan: Any) -> None:
        """Render structured TaskPlan card with approval instructions."""
        from rich.table import Table

        table = Table(
            title="[Implementation Plan Tasks]",
            border_style="cyan",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("#", style="dim", width=4)
        table.add_column("Task ID", style="bold cyan", width=12)
        table.add_column("Title", style="bold white")
        table.add_column("Assigned", style="yellow", width=10)
        table.add_column("Description", style="dim")

        for idx, task in enumerate(getattr(plan, "tasks", []), start=1):
            table.add_row(
                str(idx),
                str(task.id),
                str(task.title),
                str(task.assigned_agent),
                str(task.description)[:80] + ("..." if len(task.description) > 80 else ""),
            )

        summary_text = getattr(plan, "summary", "") or "(No summary provided)"
        panel_content = Text()
        panel_content.append(f"Goal: {getattr(plan, 'goal', '')}\n", style="bold white")
        panel_content.append(f"Summary: {summary_text}\n\n", style="italic")

        console.print(
            Panel(
                panel_content,
                title="[Planner Architecture Plan]",
                border_style="magenta",
                padding=(1, 2),
            )
        )
        console.print(table)
        console.print(
            Panel(
                "[bold cyan]Action Required:[/bold cyan] Do you approve this plan to proceed with execution?\n"
                "• Enter [bold green]'yes'[/bold green] or [bold green]'approve'[/bold green] to automatically switch to [bold yellow]Coder[/bold yellow] mode and build.\n"
                "• Enter [bold red]'no'[/bold red] to cancel, or provide feedback/adjustments to revise the plan.",
                title="[Plan Approval Gate]",
                border_style="yellow",
                padding=(0, 2),
            )
        )

    @staticmethod
    def render_context_budget_bar(
        used_tokens: int,
        max_tokens: int,
        pinned_files_tokens: int = 0,
    ) -> None:
        """Render horizontal visual bar chart of token budget allocation."""
        ratio = min(1.0, used_tokens / max(1, max_tokens))
        percent = int(ratio * 100)

        # 40-char width bar
        bar_width = 40
        filled = int(bar_width * ratio)
        empty = bar_width - filled

        color = "green" if percent < 65 else ("yellow" if percent < 85 else "red")
        bar_str = f"[{color}]" + "━" * filled + "[/]" + "[dim]" + "─" * empty + "[/dim]"

        table = Table.grid(padding=(0, 1))
        table.add_column(style="bold white")
        table.add_column()
        table.add_column(style="dim")

        table.add_row(
            "Context Window:",
            f"{bar_str} [{color}]{percent}%[/]",
            f"({used_tokens:,} / {max_tokens:,} tokens)",
        )
        if pinned_files_tokens:
            table.add_row("Pinned Files:", f"[cyan]~{pinned_files_tokens:,} tokens[/cyan]")

        console.print(Panel(table, title="[Context Budget Allocation]", border_style=color))
