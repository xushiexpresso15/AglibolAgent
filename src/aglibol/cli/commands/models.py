"""Command: aglibol models"""

from __future__ import annotations

import asyncio

import click
from rich.console import Console
from rich.table import Table

from aglibol.cli.display import Display
from aglibol.cli.interactive.selector import ModelCatalogHelper, TUISelector
from aglibol.core.config import ConfigManager
from aglibol.core.hardware import HardwareProfiler
from aglibol.core.optimizer import ResourceOptimizer
from aglibol.ollama.client import OllamaClient

console = Console()


def _run_models_list() -> None:
    Display.print_banner()

    cfg = ConfigManager.load()
    client = OllamaClient(host=cfg.ollama.host)
    profile = HardwareProfiler.detect()

    async def _list() -> None:
        is_alive = await client.health_check()
        if not is_alive:
            Display.print_error(
                f"Could not connect to Ollama daemon at {cfg.ollama.host}. Please run 'ollama serve'."
            )
            return

        models = await client.list_models()
        if not models:
            console.print("[yellow]No models currently installed in local Ollama.[/yellow]")
            console.print("Pull recommended models using e.g.: [cyan]ollama pull qwen2.5:7b[/cyan]")
            return

        effective_vram = (
            profile.total_vram_free_gb
            if profile.total_vram_free_gb > 0
            else (profile.primary_gpu.vram_free_gb if profile.primary_gpu else 0.0)
        )

        table = Table(
            title=f"Installed Local Ollama Models (Compute Tier: {profile.tier.value.upper()})",
            border_style="cyan",
            title_style="bold cyan",
        )
        table.add_column("Model Name", style="bold white")
        table.add_column("Parameters", style="green")
        table.add_column("Quantization", style="yellow")
        table.add_column("Disk Size", style="magenta")
        table.add_column("Hardware Offload Feasibility", style="bold")

        for m in models:
            size_gb = m.size_gb
            param_b = ResourceOptimizer.estimate_model_parameters(m.name)
            layers, ratio = ResourceOptimizer.calculate_gpu_layers(effective_vram, param_b=param_b)

            if not profile.has_discrete_gpu:
                if size_gb <= profile.ram_available_gb * 0.7:
                    status = "[green][OK] Runs in RAM (CPU)[/green]"
                else:
                    status = "[red][TIGHT] Exceeds available RAM[/red]"
            else:
                if layers == -1:
                    status = "[green][FULL] 100% GPU Offload[/green]"
                elif layers > 0:
                    status = f"[yellow][PARTIAL] {int(ratio * 100)}% GPU ({layers} layers), rest CPU[/yellow]"
                else:
                    if size_gb <= profile.ram_available_gb * 0.7:
                        status = "[cyan][CPU] Insufficient VRAM, runs in RAM[/cyan]"
                    else:
                        status = "[red][EXCEEDED] Exceeds VRAM & RAM[/red]"

            table.add_row(
                m.name,
                m.parameter_size or f"~{param_b:.0f}B",
                m.quantization_level or "Unknown",
                f"{size_gb:.1f} GB",
                status,
            )

        console.print(table)
        console.print("\n[dim]To bind a model to an agent role permanently:[/dim]")
        console.print("  [cyan]aglibol models bind coder <model_name>[/cyan]")
        console.print("  [cyan]aglibol models select[/cyan] (interactive selection wizard)\n")

    asyncio.run(_list())


@click.group("models", invoke_without_command=True)
@click.pass_context
def models_cmd(ctx: click.Context) -> None:
    """List installed Ollama models, assess hardware compatibility, or bind models to roles."""
    if ctx.invoked_subcommand is not None:
        return
    _run_models_list()


@models_cmd.command("list")
def list_models() -> None:
    """List installed Ollama models and assess hardware offload feasibility."""
    _run_models_list()


@models_cmd.command("ls")
def ls_models() -> None:
    """Alias for 'list'."""
    _run_models_list()


@models_cmd.command("bind")
@click.argument(
    "role",
    type=click.Choice(
        ["chat", "planner", "coder", "reviewer", "writer", "all"], case_sensitive=False
    ),
)
@click.argument("model_name", type=str)
def bind_model(role: str, model_name: str) -> None:
    """Bind an agent role ('chat', 'planner', 'coder', 'reviewer', 'writer', or 'all') to a specific model."""
    Display.print_banner()

    target_role = role.lower()
    if target_role == "all":
        roles = ["chat", "planner", "coder", "reviewer", "writer"]
    else:
        roles = [target_role]

    saved_file = None
    for r in roles:
        saved_file = ConfigManager.set_user_config_value(f"models.{r}", model_name)

    roles_str = ", ".join(roles)
    Display.print_success(
        f"Bound role(s) [bold cyan]{roles_str}[/bold cyan] to [bold green]{model_name}[/bold green]\n"
        f"Saved to: [dim]{saved_file}[/dim]"
    )


@models_cmd.command("select")
def select_models() -> None:
    """Interactive wizard with graphical TUI to choose and bind models for agent roles."""
    Display.print_banner()

    cfg = ConfigManager.load()
    client = OllamaClient(host=cfg.ollama.host)
    profile = HardwareProfiler.detect()

    async def _interactive() -> None:
        is_alive = await client.health_check()
        if not is_alive:
            Display.print_error(
                f"Could not connect to Ollama daemon at {cfg.ollama.host}. Please run 'ollama serve'."
            )
            return

        models = await client.list_models()
        if not models:
            console.print("[yellow]No models currently installed in local Ollama.[/yellow]")
            return

        roles = [
            ("chat", getattr(cfg.models, "chat", "qwen2.5:7b")),
            ("planner", cfg.models.planner),
            ("coder", cfg.models.coder),
            ("reviewer", cfg.models.reviewer),
            ("writer", getattr(cfg.models, "writer", "qwen2.5:7b")),
        ]

        gpu_desc = (
            f"{profile.primary_gpu.name} ({profile.total_vram_free_gb:.1f} GB free VRAM)"
            if profile.primary_gpu
            else "CPU Mode"
        )

        for role_name, current_val in roles:
            model_items = ModelCatalogHelper.classify_and_build_items(
                installed_models=models,
                role=role_name,
                current_model=current_val,
            )

            chosen = TUISelector.choose(
                title=f"Configure Model for {role_name.upper()}",
                subtitle=f"Current: {current_val} • Hardware: {gpu_desc} ({profile.tier.value.upper()})",
                items=model_items,
                default_index=0,
                allow_custom=True,
                cancel_label=f"Keep Current ({current_val})",
            )

            if chosen:
                ConfigManager.set_user_config_value(f"models.{role_name}", chosen)
                console.print(f"  [green]{role_name}[/green] -> [bold cyan]{chosen}[/bold cyan]")
            else:
                console.print(f"  - Kept current: [dim]{current_val}[/dim]")

        Display.print_success("Model configuration updated successfully!")

    asyncio.run(_interactive())
