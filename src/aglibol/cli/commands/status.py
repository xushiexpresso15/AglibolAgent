"""Command: aglibol status"""

from __future__ import annotations

import asyncio

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from aglibol.cli.display import Display
from aglibol.core.config import ConfigManager
from aglibol.core.hardware import HardwareProfiler
from aglibol.ollama.client import OllamaClient
from aglibol.storage.brain import Brain

console = Console()


@click.command("status")
def status_cmd() -> None:
    """Inspect active Ollama status, resident VRAM models, and disk storage."""
    Display.print_banner()

    cfg = ConfigManager.load()
    client = OllamaClient(host=cfg.ollama.host)
    profile = HardwareProfiler.detect()
    brain = Brain(cfg.storage.brain_dir)

    async def _status() -> None:
        is_alive = await client.health_check()
        status_color = "green" if is_alive else "red"
        status_text = "Running" if is_alive else "Offline (Unable to connect)"

        console.print(
            Panel(
                f"[bold]Ollama Endpoint:[/bold] {cfg.ollama.host}  [{status_color}]● {status_text}[/{status_color}]\n"
                f"[bold]Host GPU:[/bold] {profile.primary_gpu.name if profile.primary_gpu else 'None'}  "
                f"[bold]Brain Disk Usage:[/bold] {brain.get_disk_usage_mb():.2f} MB ({brain.root})",
                title="System Telemetry",
                border_style="blue",
            )
        )

        if not is_alive:
            return

        # Check resident loaded models
        loaded = await client.ps()
        if not loaded:
            console.print(
                "[green]No models currently resident in VRAM (GPU memory is 100% free).[/green]"
            )
        else:
            table = Table(title="Currently Loaded Models in VRAM", border_style="yellow")
            table.add_column("Model Name", style="bold white")
            table.add_column("VRAM Size", style="yellow")
            table.add_column("Expires At", style="dim")

            for lm in loaded:
                table.add_row(lm.name, f"{lm.vram_gb:.2f} GB", lm.expires_at)

            console.print(table)

    asyncio.run(_status())
