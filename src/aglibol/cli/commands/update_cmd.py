"""Command: aglibol update"""

from __future__ import annotations

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

from aglibol.utils.updater import UpdateManager

console = Console()


@click.command("update")
@click.option("--check-only", is_flag=True, help="Only check for updates without upgrading.")
@click.option("--force", is_flag=True, help="Bypass cached check and force query GitHub.")
def update_cmd(check_only: bool, force: bool) -> None:
    """Check for new versions on GitHub and upgrade Aglibol Agent."""
    updater = UpdateManager()
    console.print("[cyan]Checking GitHub for updates...[/cyan]")

    info = updater.check_for_updates(force=force or True, timeout=5.0)

    if not info.is_newer:
        console.print(
            Panel(
                f"[bold green]You are on the latest version of Aglibol Agent![/bold green]\n"
                f"Current Version: [cyan]v{info.current_version}[/cyan]\n"
                f"Source: [dim]{info.source}[/dim]",
                title="Aglibol Agent Update",
                border_style="green",
            )
        )
        return

    # Newer version available
    panel_content = (
        f"[bold yellow]A new version of Aglibol Agent is available![/bold yellow]\n\n"
        f"  • Current Version: [dim]v{info.current_version}[/dim]\n"
        f"  • Latest Version:  [bold green]v{info.latest_version}[/bold green]\n"
        f"  • Install Method:  [cyan]{updater.detect_install_method().upper()}[/cyan]\n"
    )
    if info.release_notes:
        panel_content += f"\n[dim]Release Notes:\n{info.release_notes.strip()[:300]}[/dim]"

    console.print(Panel(panel_content, title="Update Available", border_style="yellow"))

    if check_only:
        console.print("[dim]Run 'aglibol update' without --check-only to upgrade.[/dim]")
        return

    if Confirm.ask("Would you like to upgrade Aglibol Agent now?", default=True):
        updater.perform_upgrade()
