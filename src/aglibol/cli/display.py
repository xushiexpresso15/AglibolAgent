"""Rich terminal presentation and display utilities with cross-platform encoding safety."""

from __future__ import annotations

import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from aglibol.core.types import HardwareProfile, OptimizedParams, TaskPlan

# Ensure stdout/stderr do not crash on non-UTF-8 Windows codepages (e.g., cp950, cp936)
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

console = Console(force_terminal=True, legacy_windows=False)


class Display:
    """Renders formatted CLI outputs using Rich."""

    @staticmethod
    def print_banner() -> None:
        """Display Aglibol Agent branding banner."""
        banner_text = Text()
        banner_text.append("[*] Aglibol Agent ", style="bold cyan")
        banner_text.append(
            "— Personal Multi-Agent AI Framework (Ollama-Native)\n", style="bold white"
        )
        banner_text.append(
            "Continuous Dynamic Resource Adaptation • From 0GB CPU to 80GB+ GPUs • Zero OOM",
            style="dim",
        )
        console.print(Panel(banner_text, border_style="cyan", padding=(1, 2)))

    @staticmethod
    def print_hardware_profile(profile: HardwareProfile) -> None:
        """Display detected hardware and continuous capacity table."""
        table = Table(
            title="Detected System & Hardware Specifications",
            border_style="bright_blue",
            title_style="bold cyan",
        )
        table.add_column("Component", style="bold white")
        table.add_column("Detected Specifications", style="green")
        table.add_column("Status / Utilization", style="yellow")

        # OS & CPU
        table.add_row("Operating System", f"{profile.os_name} {profile.os_version}", "Normal")
        table.add_row(
            "Processor (CPU)",
            f"{profile.cpu_name} ({profile.cpu_cores_physical}C / {profile.cpu_cores_logical}T)",
            "Active",
        )

        # RAM
        table.add_row(
            "System Memory (RAM)",
            f"{profile.ram_available_gb:.1f} GB available / {profile.ram_total_gb:.1f} GB total",
            f"{profile.ram_used_percent}% used",
        )

        # GPU / Graphics
        if profile.has_discrete_gpu and profile.primary_gpu:
            for idx, g in enumerate(profile.gpus):
                table.add_row(
                    f"GPU #{idx + 1} ({g.vendor})",
                    f"{g.name}",
                    f"VRAM: {g.vram_free_gb:.1f} GB free / {g.vram_total_gb:.1f} GB total",
                )
            if profile.gpu_count > 1:
                table.add_row(
                    "Multi-GPU Total",
                    f"{profile.gpu_count} GPUs detected",
                    f"Total VRAM: {profile.total_vram_free_gb:.1f} GB free / {profile.total_vram_gb:.1f} GB total",
                )
        elif profile.is_unified_memory:
            table.add_row(
                "Graphics Architecture", "Apple Silicon Unified Memory", "Shared RAM/VRAM"
            )
        else:
            table.add_row(
                "Graphics Architecture",
                "No Dedicated GPU (Pure CPU Mode)",
                "Inference on System RAM",
            )

        # Storage
        for d in profile.disks[:2]:
            table.add_row(
                "Storage Disk",
                f"{d.mountpoint} ({d.free_gb:.1f} GB free / {d.total_gb:.1f} GB)",
                f"{d.used_percent}% used",
            )

        # Assigned Tier
        table.add_row(
            "Hardware Compute Tier",
            f"[bold magenta]{profile.tier.value.upper()}[/bold magenta]",
            "[bold green]DYNAMIC ADAPTATION[/bold green]",
        )

        console.print(table)

    @staticmethod
    def print_optimization_summary(params: OptimizedParams) -> None:
        """Display detailed dynamic optimization deductions."""
        lines = [
            f"[bold]Target Inference Model:[/bold] [cyan]{params.recommended_model}[/cyan]",
            f"[bold]Dynamic Context Window:[/bold] [green]{params.num_ctx:,}[/green] tokens",
            "[bold]Layer Offload Strategy:[/bold] "
            + (
                "[green]100% GPU Offload[/green]"
                if params.num_gpu == -1
                else (
                    "[yellow]100% CPU Execution[/yellow]"
                    if params.num_gpu == 0
                    else f"[cyan]Partial Offload ({params.num_gpu} layers on GPU)[/cyan]"
                )
            ),
            "[bold]Memory Residency Policy:[/bold] "
            + (
                f"[magenta]High-Capacity Mode (Up to {params.max_loaded_models} resident models)[/magenta]"
                if params.concurrency_allowed
                else "[yellow]Constrained Mode (Strict Sequential Swapping)[/yellow]"
            ),
            f"[bold]Flash Attention:[/bold] {'Enabled' if params.flash_attention else 'Disabled'}",
            f"[bold]Compute Backend:[/bold] [bold white]{params.target_backend.upper()}[/bold white]",
            "",
            f"[dim]Mathematical Deduction: {params.reasoning}[/dim]",
        ]
        console.print(
            Panel("\n".join(lines), title="Dynamic Mathematical Optimization", border_style="green")
        )

    @staticmethod
    def print_plan(plan: TaskPlan) -> None:
        """Display the task decomposition plan table."""
        table = Table(
            title=f"Execution Plan ({plan.plan_id})", border_style="green", title_style="bold green"
        )
        table.add_column("Step ID", style="cyan", no_wrap=True)
        table.add_column("Title", style="bold white")
        table.add_column("Assigned Agent", style="magenta")
        table.add_column("Status", style="yellow")

        for t in plan.tasks:
            status_style = "green" if t.status == "completed" else "yellow"
            table.add_row(
                t.id,
                t.title,
                t.assigned_agent,
                f"[{status_style}]{t.status.value}[/{status_style}]",
            )

        console.print(table)

    @staticmethod
    def print_step(step: int, agent: str, title: str) -> None:
        """Display step header."""
        console.print(
            f"\n[bold cyan]--- Step {step}:[/bold cyan] [bold yellow]{agent.upper()}[/bold yellow] [dim]({title})[/dim]"
        )

    @staticmethod
    def print_success(message: str) -> None:
        """Display success message."""
        console.print(Panel(f"[bold green][OK][/bold green] {message}", border_style="green"))

    @staticmethod
    def print_error(message: str) -> None:
        """Display error message."""
        console.print(Panel(f"[bold red][ERROR][/bold red] {message}", border_style="red"))
