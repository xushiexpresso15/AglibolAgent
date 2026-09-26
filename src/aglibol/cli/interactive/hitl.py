"""Human-in-the-Loop (HITL) approval gate for tool execution safety."""

from __future__ import annotations

from enum import Enum
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()


class SafetyMode(str, Enum):
    STRICT = "strict"  # Ask for all tools
    BALANCED = "balanced"  # Ask only for shell execution and destructive file ops
    AUTONOMOUS = "autonomous"  # Never ask; fully autonomous execution


class ApprovalDecision(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    EDITED = "edited"


class SecurityGate:
    """Controls and intercepts tool execution requests requiring human confirmation."""

    def __init__(self, mode: SafetyMode = SafetyMode.BALANCED) -> None:
        self.mode = mode
        self.always_allowed_tools: set[str] = set()

    def should_prompt(self, tool_name: str, args: dict[str, Any]) -> bool:
        """Determine if tool call requires user confirmation under active safety policy."""
        if self.mode == SafetyMode.AUTONOMOUS:
            return False

        if tool_name in self.always_allowed_tools:
            return False

        if self.mode == SafetyMode.STRICT:
            return True

        # Balanced mode: prompt for shell commands and file deletions/overwrites
        if tool_name == "execute_shell":
            return True

        if tool_name in ("write_file", "search_replace"):
            # If path looks like a critical config, prompt
            path = str(args.get("path", ""))
            if any(p in path.lower() for p in [".git", "config", "env", "settings"]):
                return True
            from pathlib import Path

            if ".." in path or Path(path).name.startswith(".") or Path(path).is_absolute():
                return True

        return False

    async def request_approval(
        self, tool_name: str, args: dict[str, Any]
    ) -> tuple[ApprovalDecision, dict[str, Any]]:
        """
        Prompt the user interactively in the terminal for approval.
        Returns (decision, possibly_edited_arguments).
        """
        if not self.should_prompt(tool_name, args):
            return ApprovalDecision.APPROVED, args

        # Render handsome tool inspection panel
        if tool_name == "execute_shell":
            cmd = args.get("command", "")
            title = "[bold yellow][Confirmation Required] Shell Command[/bold yellow]"
            content = f"[bold cyan]Command:[/bold cyan] [white]{cmd}[/white]"
        else:
            title = f"[bold yellow][Confirmation Required] Tool '{tool_name}'[/bold yellow]"
            items = [f"[bold]{k}:[/bold] {v}" for k, v in args.items()]
            content = "\n".join(items)

        console.print("\n" + "=" * 60)
        console.print(Panel(content, title=title, border_style="yellow"))

        while True:
            choice = Prompt.ask(
                "Action",
                choices=["y", "n", "a", "e"],
                default="y",
                show_choices=False,
                description="[bold white][Y]es[/bold white] / [red][N]o[/red] / [green][A]lways allow in session[/green] / [cyan][E]dit[/cyan]",
            ).lower()

            if choice == "y":
                return ApprovalDecision.APPROVED, args
            elif choice == "a":
                self.always_allowed_tools.add(tool_name)
                console.print(f"[green]Tool '{tool_name}' is now allowed for this session.[/green]")
                return ApprovalDecision.APPROVED, args
            elif choice == "e":
                if tool_name == "execute_shell":
                    new_cmd = Prompt.ask("Enter edited command", default=args.get("command", ""))
                    new_args = dict(args)
                    new_args["command"] = new_cmd
                    return ApprovalDecision.EDITED, new_args
                console.print(
                    "[yellow]Editing is not supported for this tool. Please choose another option.[/yellow]"
                )
                continue
            else:
                console.print("[red]Tool execution rejected by user.[/red]")
                return ApprovalDecision.REJECTED, args
