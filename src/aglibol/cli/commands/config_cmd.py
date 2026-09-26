"""Command: aglibol config"""

from __future__ import annotations

from typing import Any

import click
from rich.console import Console
from rich.panel import Panel

from aglibol.cli.display import Display
from aglibol.core.config import ConfigManager
from aglibol.core.hardware import HardwareProfiler
from aglibol.core.optimizer import ResourceOptimizer

console = Console()


@click.group("config", invoke_without_command=True)
@click.option(
    "--auto", is_flag=True, help="Auto-detect hardware and display optimal configuration."
)
@click.option("--show", is_flag=True, help="Display currently loaded configuration settings.")
@click.pass_context
def config_cmd(ctx: click.Context, auto: bool, show: bool) -> None:
    """Inspect and manage Aglibol Agent configuration."""
    if ctx.invoked_subcommand is not None:
        return

    Display.print_banner()

    if show:
        cfg = ConfigManager.load()
        console.print(
            Panel(cfg.model_dump_json(indent=2), title="Current Configuration", border_style="blue")
        )
    else:
        profile = HardwareProfiler.detect()
        Display.print_hardware_profile(profile)
        params = ResourceOptimizer.get_params_for_profile(profile, role="coder")
        Display.print_optimization_summary(params)


@config_cmd.command("set")
@click.argument("key", type=str)
@click.argument("value", type=str)
def set_config(key: str, value: str) -> None:
    """Set a persistent configuration key (e.g. 'models.coder' 'qwen2.5-coder:7b')."""
    Display.print_banner()

    parsed_value: Any = value
    if value.lower() == "true":
        parsed_value = True
    elif value.lower() == "false":
        parsed_value = False
    else:
        try:
            parsed_value = int(value)
        except ValueError:
            try:
                parsed_value = float(value)
            except ValueError:
                pass

    user_file = ConfigManager.set_user_config_value(key, parsed_value)
    Display.print_success(
        f"Saved [bold cyan]{key}[/bold cyan] = [bold green]{parsed_value}[/bold green] to {user_file}"
    )
