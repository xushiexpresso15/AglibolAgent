"""Graphical and interactive TUI selector for roles and models."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

console = Console()


@dataclass
class SelectorItem:
    """Represents a single selectable option in the TUI menu."""

    key: str
    label: str
    specs: str = ""
    badge: str = ""
    badge_style: str = "bold green"
    is_recommended: bool = False
    is_disabled: bool = False


def _read_single_key() -> str:
    """Read a single keypress across Windows and Unix platforms."""
    if sys.platform == "win32":
        import msvcrt

        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            ch2 = msvcrt.getwch()
            if ch2 == "H":
                return "up"
            elif ch2 == "P":
                return "down"
            elif ch2 == "K":
                return "left"
            elif ch2 == "M":
                return "right"
            return "special"
        elif ch == "\r":
            return "enter"
        elif ch in ("\x1b", "\x03"):
            return "escape"
        return ch
    else:
        import select
        import termios
        import tty

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                r, _, _ = select.select([sys.stdin], [], [], 0.05)
                if r:
                    ch2 = sys.stdin.read(1)
                    if ch2 == "[":
                        ch3 = sys.stdin.read(1)
                        if ch3 == "A":
                            return "up"
                        elif ch3 == "B":
                            return "down"
                        elif ch3 == "C":
                            return "right"
                        elif ch3 == "D":
                            return "left"
                return "escape"
            elif ch in ("\r", "\n"):
                return "enter"
            elif ch == "\x03":
                return "escape"
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


class TUISelector:
    """Interactive Graphical TUI Selector using arrow keys, numeric shortcuts, and Rich rendering."""

    @classmethod
    def choose(
        cls,
        title: str,
        items: Sequence[SelectorItem],
        subtitle: str = "",
        default_index: int = 0,
        allow_custom: bool = False,
        custom_prompt: str = "Enter custom model tag (e.g. qwen2.5:7b)",
        custom_key: str = "__custom__",
        custom_label: str = "Enter Custom Model Tag...",
        custom_specs: str = "Manually enter custom input",
        cancel_label: str = "Cancel",
    ) -> str | None:
        """
        Display an interactive TUI selector with keyboard navigation.
        Returns the chosen item key, custom input string, or None if cancelled.
        """
        all_items: list[SelectorItem] = list(items)

        if allow_custom:
            all_items.append(
                SelectorItem(
                    key=custom_key,
                    label=custom_label,
                    specs=custom_specs,
                    badge="[Custom]",
                    badge_style="cyan",
                )
            )

        all_items.append(
            SelectorItem(
                key="__cancel__",
                label=cancel_label,
                specs="Return without changes",
                badge="[Exit]",
                badge_style="dim",
            )
        )

        if not all_items:
            return None

        # Clamp default_index
        selected_index = max(0, min(default_index, len(all_items) - 1))

        # Check if environment supports interactive TTY
        is_interactive = sys.stdin.isatty() and sys.stdout.isatty()

        if not is_interactive:
            return cls._fallback_prompt(
                title=title,
                items=all_items,
                subtitle=subtitle,
                default_index=selected_index,
                custom_prompt=custom_prompt,
                custom_key=custom_key,
            )

        def _build_panel(current_idx: int) -> Panel:
            table = Table.grid(padding=(0, 1), expand=True)
            table.add_column("Cursor", width=8, no_wrap=True)
            table.add_column("Label", width=26, no_wrap=True)
            table.add_column("Specs", ratio=1)
            table.add_column("Badge", width=22, justify="right", no_wrap=True)

            for idx, item in enumerate(all_items):
                is_selected = idx == current_idx
                num_label = f"[{idx + 1}]" if idx < len(all_items) - 1 else "[0]"

                if is_selected:
                    cursor_str = f"[bold cyan]> {num_label}[/bold cyan]"
                    label_str = f"[bold cyan]{item.label}[/bold cyan]"
                    specs_str = f"[white]{item.specs}[/white]"
                else:
                    cursor_str = f"  [dim]{num_label}[/dim]"
                    label_str = (
                        f"[white]{item.label}[/white]"
                        if not item.is_disabled
                        else f"[dim]{item.label}[/dim]"
                    )
                    specs_str = f"[dim]{item.specs}[/dim]"

                badge_str = (
                    f"[{item.badge_style}]{item.badge}[/{item.badge_style}]" if item.badge else ""
                )
                table.add_row(cursor_str, label_str, specs_str, badge_str)

            instructions = (
                "[dim]Use [bold white]↑/↓[/bold white] or [bold white]j/k[/bold white] to move  •  "
                "[bold white][1-9,0][/bold white] quick jump  •  "
                "[bold white]Enter[/bold white] confirm  •  "
                "[bold white]Esc[/bold white] cancel[/dim]"
            )

            full_subtitle = f"{subtitle}\n{instructions}" if subtitle else instructions

            return Panel(
                table,
                title=f"[bold cyan]{title}[/bold cyan]",
                subtitle=full_subtitle,
                border_style="cyan",
                padding=(1, 2),
            )

        try:
            with Live(
                _build_panel(selected_index), console=console, auto_refresh=False, transient=True
            ) as live:
                while True:
                    key = _read_single_key()

                    if key in ("up", "k"):
                        selected_index = (selected_index - 1) % len(all_items)
                        live.update(_build_panel(selected_index))
                        live.refresh()
                    elif key in ("down", "j"):
                        selected_index = (selected_index + 1) % len(all_items)
                        live.update(_build_panel(selected_index))
                        live.refresh()
                    elif key in ("escape", "q"):
                        return None
                    elif key == "enter":
                        chosen = all_items[selected_index]
                        break
                    elif key.isdigit():
                        num = int(key)
                        if num == 0:
                            # 0 corresponds to cancel (last item)
                            selected_index = len(all_items) - 1
                        elif 1 <= num < len(all_items):
                            selected_index = num - 1
                        live.update(_build_panel(selected_index))
                        live.refresh()

            if chosen.key == "__cancel__":
                return None
            elif chosen.key == custom_key:
                val = Prompt.ask(f"[bold cyan]{custom_prompt}[/bold cyan]").strip()
                return val if val else None
            else:
                return chosen.key

        except (KeyboardInterrupt, EOFError):
            return None
        except Exception:
            # Safe fallback if any terminal error occurs
            return cls._fallback_prompt(
                title=title,
                items=all_items,
                subtitle=subtitle,
                default_index=selected_index,
                custom_prompt=custom_prompt,
                custom_key=custom_key,
            )

    @classmethod
    def _fallback_prompt(
        cls,
        title: str,
        items: list[SelectorItem],
        subtitle: str,
        default_index: int,
        custom_prompt: str,
        custom_key: str,
    ) -> str | None:
        """Non-interactive or fallback prompt accepting numbers, keys, or aliases."""
        table = Table(title=title, border_style="cyan")
        table.add_column("No.", style="cyan", width=5)
        table.add_column("Option", style="bold white", width=24)
        table.add_column("Details", style="dim", ratio=1)
        table.add_column("Tag", style="green", width=20)

        for idx, it in enumerate(items, 1):
            num_str = f"[{idx}]" if idx < len(items) else "[0]"
            badge_str = f"[{it.badge_style}]{it.badge}[/{it.badge_style}]" if it.badge else ""
            table.add_row(num_str, it.label, it.specs, badge_str)

        console.print(table)
        if subtitle:
            console.print(f"[dim]{subtitle}[/dim]")

        default_num = str(default_index + 1) if default_index < len(items) - 1 else "1"
        try:
            choice = Prompt.ask(
                "Select option by number or name (or press Enter for default)",
                default=default_num,
            ).strip()
        except (KeyboardInterrupt, EOFError, OSError):
            return None

        if not choice or choice == "0" or choice.lower() in ("cancel", "exit", "q"):
            return None

        # 1. Match by number
        if choice.isdigit():
            num = int(choice)
            if 1 <= num <= len(items):
                selected = items[num - 1]
                if selected.key == "__cancel__":
                    return None
                elif selected.key == custom_key:
                    try:
                        return Prompt.ask(custom_prompt).strip() or None
                    except (KeyboardInterrupt, EOFError, OSError):
                        return None
                return selected.key

        # 2. Match by key or label (case-insensitive)
        choice_lower = choice.lower()
        for it in items:
            if choice_lower in (it.key.lower(), it.label.lower()):
                if it.key == "__cancel__":
                    return None
                elif it.key == custom_key:
                    try:
                        return Prompt.ask(custom_prompt).strip() or None
                    except (KeyboardInterrupt, EOFError, OSError):
                        return None
                return it.key

        return choice


class ModelCatalogHelper:
    """Classifies, tags, and ranks local Ollama models for optimal agent role assignment."""

    EMBEDDING_KEYWORDS = {"embed", "bge", "minilm", "e5", "bert", "text-embedding"}
    VISION_KEYWORDS = {"ocr", "vision", "vl", "clip", "llava"}
    CODER_KEYWORDS = {"coder", "code", "dev", "deepseek-coder", "starcoder"}
    REASONING_KEYWORDS = {"r1", "reason", "qwq", "deepseek-r1"}

    @classmethod
    def is_embedding_model(cls, model_name: str) -> bool:
        lower = model_name.lower()
        return any(k in lower for k in cls.EMBEDDING_KEYWORDS)

    @classmethod
    def is_vision_model(cls, model_name: str) -> bool:
        lower = model_name.lower()
        return any(k in lower for k in cls.VISION_KEYWORDS)

    @classmethod
    def classify_and_build_items(
        cls,
        installed_models: list[Any],
        role: str,
        current_model: str = "",
    ) -> list[SelectorItem]:
        """
        Convert list of ModelInfo into ranked, tagged SelectorItem entries for a specific role.
        """
        role_lower = role.lower()

        # Score and categorize each installed model
        scored_models = []
        for m in installed_models:
            name = getattr(m, "name", str(m))
            size_gb = getattr(m, "size_gb", 0.0)
            params = getattr(m, "parameter_size", "")
            quant = getattr(m, "quantization_level", "")

            name_lower = name.lower()
            is_embed = cls.is_embedding_model(name)
            is_vision = cls.is_vision_model(name)

            specs_parts = []
            if params:
                specs_parts.append(params)
            if quant:
                specs_parts.append(quant)
            if size_gb > 0:
                specs_parts.append(f"{size_gb:.1f} GB")
            specs = " • ".join(specs_parts) if specs_parts else "Local model"

            score = 50
            badge = ""
            badge_style = "dim"

            if is_embed:
                score = 0
                badge = "[Embedding Only]"
                badge_style = "dim yellow"
            elif is_vision:
                score = 20
                badge = "[Vision/OCR Only]"
                badge_style = "dim cyan"
            else:
                if role_lower == "coder":
                    if any(k in name_lower for k in cls.CODER_KEYWORDS):
                        score = 100
                        badge = "[BEST FOR CODER]"
                        badge_style = "bold green"
                    elif "qwen" in name_lower or "deepseek" in name_lower or "llama" in name_lower:
                        score = 70
                        badge = "[General LLM]"
                        badge_style = "green"
                elif role_lower == "planner":
                    if any(k in name_lower for k in cls.REASONING_KEYWORDS):
                        score = 100
                        badge = "[BEST FOR PLANNER]"
                        badge_style = "bold magenta"
                    elif "qwen" in name_lower or "llama" in name_lower:
                        score = 80
                        badge = "[Recommended]"
                        badge_style = "bold green"
                elif role_lower == "reviewer":
                    if any(k in name_lower for k in cls.CODER_KEYWORDS):
                        score = 90
                        badge = "[Code Analysis]"
                        badge_style = "bold green"
                    elif "qwen" in name_lower or "llama" in name_lower:
                        score = 80
                        badge = "[Recommended]"
                        badge_style = "green"
                elif role_lower == "writer":
                    if any(
                        k in name_lower
                        for k in ("writer", "chat", "instruct", "qwen", "llama", "mistral", "gemma")
                    ):
                        score = 90
                        badge = "[Writing & Content]"
                        badge_style = "bold cyan"
                    elif any(k in name_lower for k in cls.CODER_KEYWORDS):
                        score = 70
                        badge = "[General LLM]"
                        badge_style = "green"
                elif role_lower == "chat":
                    if any(
                        k in name_lower
                        for k in ("chat", "instruct", "qwen", "llama", "mistral", "gemma", "phi")
                    ):
                        score = 90
                        badge = "[Conversational & Q&A]"
                        badge_style = "bold cyan"
                    elif any(k in name_lower for k in cls.CODER_KEYWORDS):
                        score = 70
                        badge = "[General LLM]"
                        badge_style = "green"

            # If this is currently the active model
            if current_model and (
                name == current_model or name.split(":")[0] == current_model.split(":")[0]
            ):
                score += 15
                if not badge:
                    badge = "[Current Active]"
                    badge_style = "cyan"
                else:
                    badge = f"{badge} (Active)"

            scored_models.append(
                (
                    score,
                    SelectorItem(
                        key=name,
                        label=name,
                        specs=specs,
                        badge=badge,
                        badge_style=badge_style,
                        is_recommended=(score >= 80),
                        is_disabled=is_embed,
                    ),
                )
            )

        # Sort: highest score first (embedding models naturally go to the bottom)
        scored_models.sort(key=lambda x: x[0], reverse=True)
        return [it for _, it in scored_models]

    @classmethod
    def resolve_best_model_for_role(
        cls,
        requested_model: str,
        role: str,
        installed_models: list[str] | list[Any],
    ) -> tuple[str, bool]:
        """
        Check if requested_model is installed on host Ollama.
        If not, dynamically auto-resolve to the best installed generative model.
        Returns (selected_model_name, was_auto_fallback).
        """
        if not installed_models:
            return requested_model, False

        names = [getattr(m, "name", str(m)) for m in installed_models]

        # 1. Exact match
        if requested_model in names:
            return requested_model, False

        # Filter out embedding-only models
        generative = [n for n in names if not cls.is_embedding_model(n)]
        candidates = generative if generative else names

        # 2. Base name match (e.g. requested 'qwen2.5:7b', installed 'qwen2.5-coder:7b' or 'qwen2.5:latest')
        req_base = requested_model.split(":")[0].lower()
        for inst in candidates:
            inst_base = inst.split(":")[0].lower()
            if inst_base == req_base:
                return inst, True

        # 3. Partial base match (e.g. 'qwen2.5' prefix matches 'qwen2.5-coder:7b')
        for inst in candidates:
            inst_base = inst.split(":")[0].lower()
            if req_base in inst_base or inst_base in req_base:
                return inst, True

        # 4. Role-specific keyword matching
        role_lower = role.lower()
        if role_lower == "coder":
            for inst in candidates:
                if any(k in inst.lower() for k in cls.CODER_KEYWORDS):
                    return inst, True

        if role_lower in ("chat", "planner", "reviewer", "writer"):
            for inst in candidates:
                if any(k in inst.lower() for k in cls.REASONING_KEYWORDS):
                    return inst, True

        # 5. Fallback to any general LLM in candidates
        for inst in candidates:
            if not cls.is_vision_model(inst):
                return inst, True

        # Final fallback
        return candidates[0], True
