"""Command: aglibol brain"""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table

from aglibol.cli.display import Display
from aglibol.core.config import ConfigManager
from aglibol.storage.brain import Brain
from aglibol.storage.session import SessionManager

console = Console()


@click.group("brain")
def brain_cmd() -> None:
    """Manage local Brain disk persistence, sessions, and artifacts."""
    pass


@brain_cmd.command("list")
@click.option(
    "--all", "show_all", is_flag=True, default=False, help="Include empty ghost sessions."
)
def list_sessions(show_all: bool) -> None:
    """List all persisted agent sessions stored on disk."""
    Display.print_banner()

    cfg = ConfigManager.load()
    brain = Brain(cfg.storage.brain_dir)
    sm = SessionManager(brain.sessions_dir)

    sessions = sm.list_all_sessions(filter_empty=not show_all)
    if not sessions:
        console.print("[yellow]No persisted sessions found in Brain.[/yellow]")
        return

    table = Table(title="Persisted Sessions", border_style="cyan")
    table.add_column("Session ID", style="bold cyan")
    table.add_column("Status", style="yellow")
    table.add_column("Steps", style="green", justify="right")
    table.add_column("Tier", style="magenta")
    table.add_column("Goal", style="white")

    for s in sessions:
        status_color = "green" if s.status == "completed" else "yellow"
        table.add_row(
            s.session_id,
            f"[{status_color}]{s.status}[/{status_color}]",
            str(s.step_count),
            s.tier.value,
            s.user_goal[:50] + ("..." if len(s.user_goal) > 50 else ""),
        )

    console.print(table)


@brain_cmd.command("delete")
@click.argument("session_id", type=str)
def delete_session_cmd(session_id: str) -> None:
    """Permanently delete a specific session by session ID."""
    Display.print_banner()
    cfg = ConfigManager.load()
    brain = Brain(cfg.storage.brain_dir)
    sm = SessionManager(brain.sessions_dir)
    if sm.delete_session(session_id):
        Display.print_success(f"Session '{session_id}' successfully deleted.")
    else:
        Display.print_error(f"Session '{session_id}' not found or could not be deleted.")


@brain_cmd.command("prune")
def prune_sessions_cmd() -> None:
    """Purge all empty ghost sessions that contain no dialogue or steps."""
    Display.print_banner()
    cfg = ConfigManager.load()
    brain = Brain(cfg.storage.brain_dir)
    sm = SessionManager(brain.sessions_dir)
    count = sm.prune_empty_sessions()
    Display.print_success(f"Successfully pruned {count} empty ghost session(s).")


@brain_cmd.command("clean")
@click.option("--days", type=int, default=None, help="Only clean sessions older than N days.")
@click.confirmation_option(prompt="Are you sure you want to clean Brain storage?")
def clean_brain(days: int | None) -> None:
    """Purge persisted Brain sessions and logs to free disk space."""
    cfg = ConfigManager.load()
    brain = Brain(cfg.storage.brain_dir)
    if days is not None:
        count = brain.clean_older_than(days)
        Display.print_success(f"Cleaned {count} session(s) older than {days} day(s).")
    else:
        brain.clean_all()
        Display.print_success("Brain storage successfully cleaned.")
