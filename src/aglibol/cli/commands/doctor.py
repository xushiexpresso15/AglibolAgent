"""Command: aglibol doctor — System diagnostics, dependency checking, and onboarding onboarding."""

from __future__ import annotations

import asyncio
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import click
import httpx
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

from aglibol import __version__
from aglibol.core.config import ConfigManager
from aglibol.core.hardware import HardwareProfiler
from aglibol.ollama.client import OllamaClient
from aglibol.storage.brain import Brain

console = Console()


class DiagnosticResult:
    def __init__(
        self,
        category: str,
        name: str,
        status: str,
        details: str,
        fix_available: bool = False,
        fix_action: Any = None,
    ):
        self.category = category
        self.name = name
        self.status = status  # "PASS", "WARN", "FAIL"
        self.details = details
        self.fix_available = fix_available
        self.fix_action = fix_action


async def run_diagnostics(auto_fix: bool = False) -> list[DiagnosticResult]:
    results: list[DiagnosticResult] = []
    cfg = ConfigManager.load()
    profile = HardwareProfiler.detect()

    # 1. Python Environment
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info >= (3, 11):  # noqa: UP036
        results.append(
            DiagnosticResult(
                "Runtime", "Python Version", "PASS", f"Python {py_ver} ({sys.executable})"
            )
        )
    else:
        results.append(
            DiagnosticResult(
                "Runtime", "Python Version", "FAIL", f"Python {py_ver} is too old. Requires >= 3.11"
            )
        )

    in_venv = sys.prefix != sys.base_prefix
    results.append(
        DiagnosticResult(
            "Runtime",
            "Virtual Environment",
            "PASS" if in_venv else "WARN",
            "Active venv" if in_venv else "Global environment (Virtual environment recommended)",
        )
    )

    # 2. Hardware & VRAM
    results.append(
        DiagnosticResult(
            "Hardware",
            "Platform",
            "PASS",
            f"{platform.system()} {platform.release()} ({platform.machine()})",
        )
    )
    ram_gb = profile.ram_total_gb
    results.append(
        DiagnosticResult(
            "Hardware",
            "System RAM",
            "PASS" if ram_gb >= 8 else "WARN",
            f"{profile.ram_available_gb:.1f} GB free / {ram_gb:.1f} GB total",
        )
    )

    if profile.has_discrete_gpu and profile.primary_gpu:
        results.append(
            DiagnosticResult(
                "Hardware",
                "GPU Compute",
                "PASS",
                f"{profile.primary_gpu.name} ({profile.primary_gpu.vram_total_gb:.1f} GB VRAM, {profile.tier.value.upper()})",
            )
        )
    elif profile.is_unified_memory:
        results.append(
            DiagnosticResult(
                "Hardware",
                "GPU Compute",
                "PASS",
                f"Apple Silicon Unified Memory ({profile.tier.value.upper()})",
            )
        )
    else:
        results.append(
            DiagnosticResult(
                "Hardware",
                "GPU Compute",
                "WARN",
                f"No discrete GPU detected. Inference runs in CPU mode ({profile.tier.value.upper()})",
            )
        )

    # 3. Ollama Executable & Service
    ollama_bin = shutil.which("ollama")
    if ollama_bin:
        results.append(
            DiagnosticResult("Ollama", "Ollama Binary", "PASS", f"Found at {ollama_bin}")
        )
    else:
        install_hint = (
            "winget install Ollama.Ollama"
            if sys.platform == "win32"
            else "curl -fsSL https://ollama.com/install.sh | sh"
        )
        results.append(
            DiagnosticResult(
                "Ollama",
                "Ollama Binary",
                "FAIL",
                f"Ollama is not installed. Install via: {install_hint}",
            )
        )

    # Check Ollama Daemon Connectivity
    client = OllamaClient(host=cfg.ollama.host)
    is_alive = await client.health_check()

    async def _start_ollama_daemon() -> bool:
        console.print("[cyan]→ Starting Ollama service in the background...[/cyan]")
        return await client.start_daemon(timeout_seconds=10.0)

    has_ollama_installed = bool(ollama_bin) or (
        sys.platform == "win32"
        and (
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama app.exe"
        ).exists()
    )

    if is_alive:
        results.append(
            DiagnosticResult(
                "Ollama", "Daemon Connection", "PASS", f"Connected to {cfg.ollama.host}"
            )
        )
    else:
        results.append(
            DiagnosticResult(
                "Ollama",
                "Daemon Connection",
                "FAIL",
                f"Cannot connect to {cfg.ollama.host}. Is Ollama running?",
                fix_available=has_ollama_installed,
                fix_action=_start_ollama_daemon,
            )
        )

    # Check Installed Models
    models = []
    if is_alive:
        try:
            models = await client.list_models()
        except Exception:
            pass

    from aglibol.core.optimizer import ResourceOptimizer

    opt_params = ResourceOptimizer.get_params_for_profile(profile)
    recommended_model = opt_params.recommended_model

    async def _pull_recommended_model() -> bool:
        console.print(
            f"[cyan]→ Pulling recommended model '{recommended_model}' (this may take a few minutes)...[/cyan]"
        )
        try:
            res = subprocess.run(["ollama", "pull", recommended_model], text=True)
            return res.returncode == 0
        except Exception:
            return False

    if models:
        model_names = [m.name for m in models]
        results.append(
            DiagnosticResult(
                "Ollama",
                "Local Models",
                "PASS",
                f"{len(models)} installed: {', '.join(model_names[:4])}{'...' if len(models) > 4 else ''}",
            )
        )
    elif is_alive:
        results.append(
            DiagnosticResult(
                "Ollama",
                "Local Models",
                "WARN",
                f"No models installed. Recommended for {profile.tier.value}: '{recommended_model}'",
                fix_available=True,
                fix_action=_pull_recommended_model,
            )
        )

    # 4. Git Version Control
    git_bin = shutil.which("git")
    if git_bin:
        results.append(DiagnosticResult("Tools", "Git Executable", "PASS", f"Found at {git_bin}"))
    else:
        results.append(
            DiagnosticResult(
                "Tools",
                "Git Executable",
                "WARN",
                "Git not found. /diff and /commit commands will be disabled.",
            )
        )

    is_git_repo = (Path.cwd() / ".git").exists()
    results.append(
        DiagnosticResult(
            "Workspace",
            "Git Repository",
            "PASS" if is_git_repo else "INFO",
            "Working directory is a Git repository"
            if is_git_repo
            else "Working directory is not a Git repo (run 'git init' to enable full tracking)",
        )
    )

    # 5. Storage Directory Permissions
    brain = Brain(cfg.storage.brain_dir)
    can_write = False
    try:
        test_file = brain.root / ".perm_check"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink()
        can_write = True
    except Exception:
        pass

    results.append(
        DiagnosticResult(
            "Storage",
            "Brain Persistence",
            "PASS" if can_write else "FAIL",
            f"Writable: {brain.root} ({brain.get_disk_usage_mb():.2f} MB used)",
        )
    )

    # 6. Network Access to GitHub
    can_reach_github = False
    try:
        with httpx.Client(timeout=2.0) as http_client:
            r = http_client.get(
                "https://api.github.com", headers={"User-Agent": f"AglibolAgent/{__version__}"}
            )
            can_reach_github = r.status_code == 200
    except Exception:
        pass

    results.append(
        DiagnosticResult(
            "Network",
            "GitHub Connectivity",
            "PASS" if can_reach_github else "WARN",
            "Connected to GitHub API"
            if can_reach_github
            else "Cannot reach GitHub (Auto-updates will use offline cache)",
        )
    )

    return results


async def _async_doctor_cmd(fix: bool) -> None:
    results = await run_diagnostics(auto_fix=fix)

    table = Table(title="System & Dependency Health Check", border_style="bright_blue")
    table.add_column("Category", style="bold white", width=12)
    table.add_column("Check Item", style="cyan", width=22)
    table.add_column("Status", width=10)
    table.add_column("Details", style="dim white")

    status_styles = {
        "PASS": "[bold green]PASS[/bold green]",
        "WARN": "[bold yellow]WARN[/bold yellow]",
        "FAIL": "[bold red]FAIL[/bold red]",
        "INFO": "[dim blue]INFO[/dim blue]",
    }

    fail_count = 0
    warn_count = 0

    for r in results:
        if r.status == "FAIL":
            fail_count += 1
        elif r.status == "WARN":
            warn_count += 1
        table.add_row(r.category, r.name, status_styles.get(r.status, r.status), r.details)

    console.print(table)

    # Summary
    if fail_count == 0 and warn_count == 0:
        console.print(
            "\n[bold green]All system checks passed! Your environment is in optimal condition.[/bold green]\n"
        )
        return

    console.print(
        f"\n[bold yellow]Found {fail_count} critical issue(s) and {warn_count} warning(s).[/bold yellow]"
    )

    # Auto-fix handling
    fixable = [r for r in results if r.fix_available and r.fix_action]
    if fixable:
        for r in fixable:
            console.print(f"\n[cyan]Auto-fix available for: {r.name}[/cyan]")
            should_run = fix or Confirm.ask(f"Apply fix for '{r.name}' now?", default=True)
            if should_run:
                if asyncio.iscoroutinefunction(r.fix_action):
                    success = await r.fix_action()
                else:
                    success = r.fix_action()
                if success:
                    console.print(f"[bold green]Successfully resolved '{r.name}'![/bold green]")
                else:
                    console.print(
                        f"[bold red]Could not automatically fix '{r.name}'. Please see details above.[/bold red]"
                    )


@click.command("doctor")
@click.option("--fix", is_flag=True, help="Automatically attempt to fix detected issues.")
def doctor_cmd(fix: bool) -> None:
    """Run comprehensive system diagnostics, verify Ollama, and check dependencies."""
    console.print(
        Panel(
            f"[bold cyan]Aglibol Agent Doctor v{__version__}[/bold cyan]\n"
            "Diagnosing local system requirements, Ollama service, GPU resources, and tools...",
            border_style="cyan",
        )
    )
    asyncio.run(_async_doctor_cmd(fix=fix))
