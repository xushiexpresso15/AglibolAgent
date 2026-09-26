"""Automated update detection and self-upgrade manager for Aglibol Agent."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

from aglibol import __version__

console = Console()

GITHUB_REPO = "xushiexpresso15/AglibolAgent"
GITHUB_RAW_VERSION_URL = (
    f"https://raw.githubusercontent.com/{GITHUB_REPO}/main/src/aglibol/__init__.py"
)
GITHUB_API_RELEASE_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
CACHE_TTL_SECONDS = 12 * 3600  # 12 hours


class VersionInfo(NamedTuple):
    current_version: str
    latest_version: str
    is_newer: bool
    source: str
    release_notes: str | None = None


def parse_version_tuple(ver_str: str) -> tuple[int, ...]:
    """Parse a semver string like '0.1.0' or 'v0.2.1' into integer tuple."""
    cleaned = re.sub(r"^[vV]", "", ver_str.strip())
    # Extract only numeric parts for standard comparison
    parts = re.findall(r"\d+", cleaned)
    return tuple(int(p) for p in parts) if parts else (0, 0, 0)


class UpdateManager:
    """Manages update checks, caching, and self-upgrades."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or Path(os.path.expanduser("~/.aglibol"))
        self.cache_file = self.cache_dir / "update_cache.json"

    def _read_cache(self) -> dict | None:
        if not self.cache_file.exists():
            return None
        try:
            with open(self.cache_file, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _write_cache(self, data: dict) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception:
            pass

    def check_for_updates(self, force: bool = False, timeout: float = 2.0) -> VersionInfo:
        """Check for updates using cached results or fetching from GitHub with short timeout."""
        now = time.time()
        cached = self._read_cache()

        if not force and cached:
            last_checked = cached.get("timestamp", 0)
            if now - last_checked < CACHE_TTL_SECONDS:
                latest = cached.get("latest_version", __version__)
                is_newer = parse_version_tuple(latest) > parse_version_tuple(__version__)
                return VersionInfo(
                    current_version=__version__,
                    latest_version=latest,
                    is_newer=is_newer,
                    source="cache",
                    release_notes=cached.get("release_notes"),
                )

        # Query remote GitHub
        latest_ver = __version__
        release_notes = None
        source = "github"

        try:
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                # 1. Try GitHub Releases API
                headers = {"User-Agent": f"AglibolAgent/{__version__}"}
                resp = client.get(GITHUB_API_RELEASE_URL, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    tag_name = data.get("tag_name", "")
                    latest_ver = re.sub(r"^[vV]", "", tag_name) or __version__
                    release_notes = data.get("body", "")
                else:
                    # 2. Fallback to raw __init__.py on main branch
                    raw_resp = client.get(GITHUB_RAW_VERSION_URL, headers=headers)
                    if raw_resp.status_code == 200:
                        m = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', raw_resp.text)
                        if m:
                            latest_ver = m.group(1)
        except Exception:
            # On network error / offline, fallback to current version or existing cache
            if cached:
                latest_ver = cached.get("latest_version", __version__)
                source = "cache_offline"

        is_newer = parse_version_tuple(latest_ver) > parse_version_tuple(__version__)

        # Save to cache
        self._write_cache(
            {
                "timestamp": now,
                "latest_version": latest_ver,
                "release_notes": release_notes,
            }
        )

        return VersionInfo(
            current_version=__version__,
            latest_version=latest_ver,
            is_newer=is_newer,
            source=source,
            release_notes=release_notes,
        )

    def detect_install_method(self) -> str:
        """Detect how Aglibol Agent is installed (git, uv, or pip)."""
        # Check if running from a git clone
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        if (repo_root / ".git").exists() and (repo_root / "pyproject.toml").exists():
            return "git"

        # Check if installed via uv tool
        if shutil.which("uv"):
            try:
                res = subprocess.run(
                    ["uv", "tool", "list"], capture_output=True, text=True, timeout=3
                )
                if "aglibol-agent" in res.stdout:
                    return "uv"
            except Exception:
                pass

        return "pip"

    def perform_upgrade(self) -> bool:
        """Perform self-upgrade based on the detected installation method."""
        method = self.detect_install_method()
        console.print(f"\n[cyan]→ Upgrading Aglibol Agent via {method.upper()}...[/cyan]")

        try:
            if method == "git":
                repo_root = Path(__file__).resolve().parent.parent.parent.parent
                console.print(f"[dim]Pulling latest changes in {repo_root}...[/dim]")
                res_pull = subprocess.run(
                    ["git", "pull"], cwd=repo_root, capture_output=True, text=True
                )
                if res_pull.returncode != 0:
                    console.print(f"[bold red]Git pull failed:[/bold red] {res_pull.stderr}")
                    return False
                console.print("[dim]Reinstalling in editable mode...[/dim]")
                res_inst = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-e", "."],
                    cwd=repo_root,
                    capture_output=True,
                    text=True,
                )
                if res_inst.returncode != 0:
                    console.print(f"[bold red]Pip install failed:[/bold red] {res_inst.stderr}")
                    return False

            elif method == "uv":
                res = subprocess.run(
                    ["uv", "tool", "upgrade", "aglibol-agent"],
                    capture_output=True,
                    text=True,
                )
                if res.returncode != 0:
                    console.print(f"[bold red]UV upgrade failed:[/bold red] {res.stderr}")
                    return False

            else:  # pip
                res = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--upgrade", "aglibol-agent"],
                    capture_output=True,
                    text=True,
                )
                if res.returncode != 0:
                    console.print(f"[bold red]Pip upgrade failed:[/bold red] {res.stderr}")
                    return False

            # Invalidate cache
            if self.cache_file.exists():
                self.cache_file.unlink(missing_ok=True)

            console.print(
                "[bold green]Aglibol Agent successfully upgraded! Please restart your terminal/session.[/bold green]\n"
            )
            return True

        except Exception as e:
            console.print(f"[bold red]Upgrade failed with error: {e}[/bold red]")
            return False

    def check_and_prompt_on_startup(self) -> bool:
        """Checks for updates non-blockingly; prompts user if a new version is available."""
        try:
            info = self.check_for_updates(force=False, timeout=1.5)
            if info.is_newer:
                panel_msg = (
                    f"[bold yellow]A new version of Aglibol Agent is available![/bold yellow]\n\n"
                    f"Current: [dim]v{info.current_version}[/dim]  →  Latest: [bold green]v{info.latest_version}[/bold green]\n"
                    f"[dim]GitHub Repository: https://github.com/{GITHUB_REPO}[/dim]"
                )
                console.print(Panel(panel_msg, border_style="yellow", padding=(1, 2)))
                if Confirm.ask("Would you like to update Aglibol Agent now?", default=False):
                    return self.perform_upgrade()
        except Exception:
            pass
        return False
