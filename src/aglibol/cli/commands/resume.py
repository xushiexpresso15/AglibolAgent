"""Command: aglibol resume"""

from __future__ import annotations

import asyncio

import click

from aglibol.cli.commands.run import execute_run_pipeline
from aglibol.cli.display import Display
from aglibol.core.config import ConfigManager
from aglibol.storage.brain import Brain
from aglibol.storage.session import SessionManager


@click.command("resume")
@click.argument("session_id", type=str)
@click.option(
    "--config", "config_path", type=click.Path(exists=True), help="Custom YAML config path."
)
@click.option(
    "--workspace", type=click.Path(), default=None, help="Working directory for file generation."
)
def resume_cmd(session_id: str, config_path: str | None, workspace: str | None) -> None:
    """Resume an interrupted or previous agent session by session ID."""
    Display.print_banner()

    cfg = ConfigManager.load(config_path)
    brain = Brain(cfg.storage.brain_dir)
    sm = SessionManager(brain.sessions_dir)

    meta = sm.get_meta(session_id)
    if not meta:
        Display.print_error(f"Session '{session_id}' not found in Brain persistence.")
        return

    asyncio.run(
        execute_run_pipeline(
            goal=meta.user_goal,
            config_path=config_path,
            resume_session_id=session_id,
            workspace=workspace,
        )
    )
