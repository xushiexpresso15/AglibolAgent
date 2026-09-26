import asyncio
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import click

from aglibol import __version__
from aglibol.cli.commands.brain import brain_cmd
from aglibol.cli.commands.config_cmd import config_cmd
from aglibol.cli.commands.doctor import doctor_cmd
from aglibol.cli.commands.models import models_cmd
from aglibol.cli.commands.resume import resume_cmd
from aglibol.cli.commands.run import run_cmd
from aglibol.cli.commands.status import status_cmd
from aglibol.cli.commands.update_cmd import update_cmd
from aglibol.cli.interactive.hitl import SafetyMode
from aglibol.cli.interactive.repl import InteractiveSession
from aglibol.utils.updater import UpdateManager


def _launch_interactive_repl(workspace: str | None = None, safety: str | None = None) -> None:
    # Check for GitHub updates on startup non-blockingly
    try:
        UpdateManager().check_and_prompt_on_startup()
    except Exception:
        pass

    mode = SafetyMode(safety.lower()) if safety else None
    session = InteractiveSession(
        workspace_root=Path(workspace) if workspace else None,
        safety_mode=mode,
    )
    try:
        asyncio.run(session.start())
    except Exception as e:
        import sys

        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@click.group(invoke_without_command=True)
# NOTE: duplicated with chat_cmd
@click.option(
    "--workspace",
    type=click.Path(),
    default=None,
    help="Working directory for interactive session.",
)
@click.option(
    "--safety",
    type=click.Choice(["strict", "balanced", "autonomous"], case_sensitive=False),
    default=None,
    help="Security confirmation level for tool execution.",
)
@click.version_option(version=__version__, prog_name="aglibol")
@click.pass_context
def cli(ctx: click.Context, workspace: str | None, safety: str | None) -> None:
    """Aglibol Agent: An open-source, Ollama-native multi-agent framework for personal devices."""
    if ctx.invoked_subcommand is None:
        _launch_interactive_repl(workspace=workspace, safety=safety)


@click.command("chat")
@click.option(
    "--workspace",
    type=click.Path(),
    default=None,
    help="Working directory for interactive session.",
)
@click.option(
    "--safety",
    type=click.Choice(["strict", "balanced", "autonomous"], case_sensitive=False),
    default=None,
    help="Security confirmation level for tool execution.",
)
def chat_cmd(workspace: str | None, safety: str | None) -> None:
    """Launch the Grok Build / Claude Code-style interactive developer REPL."""
    _launch_interactive_repl(workspace=workspace, safety=safety)


cli.add_command(run_cmd)
cli.add_command(chat_cmd)
cli.add_command(resume_cmd)
cli.add_command(config_cmd)
cli.add_command(models_cmd)
cli.add_command(status_cmd)
cli.add_command(brain_cmd)
cli.add_command(doctor_cmd)
cli.add_command(update_cmd)

if __name__ == "__main__":
    cli()
