"""Command: aglibol run"""

from __future__ import annotations

import asyncio
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel

from aglibol.cli.display import Display
from aglibol.cli.interactive.selector import ModelCatalogHelper, TUISelector
from aglibol.core.config import ConfigManager
from aglibol.core.events import AgentEvent, EventBus
from aglibol.core.hardware import HardwareProfiler
from aglibol.core.scheduler import ModelScheduler
from aglibol.core.types import AgentState
from aglibol.core.workflow import WorkflowEngine
from aglibol.memory.episodic import EpisodicMemory
from aglibol.memory.working import WorkingMemory
from aglibol.ollama.client import OllamaClient
from aglibol.storage.artifact_store import ArtifactStore
from aglibol.storage.brain import Brain
from aglibol.storage.checkpoint import CheckpointStore
from aglibol.storage.session import SessionManager
from aglibol.tools.file_ops import register_file_tools
from aglibol.tools.registry import ToolRegistry
from aglibol.tools.search_replace import register_search_replace_tools
from aglibol.tools.shell import register_shell_tools

console = Console()


def resolve_installed_model_for_role(
    requested_model: str,
    role: str,
    installed_models: list[str],
) -> tuple[str, bool]:
    """
    Check if requested_model is installed on host Ollama.
    If not, dynamically auto-resolve to the best installed model using ModelCatalogHelper.
    Returns (selected_model_name, was_auto_fallback).
    """
    return ModelCatalogHelper.resolve_best_model_for_role(
        requested_model=requested_model,
        role=role,
        installed_models=installed_models,
    )


async def execute_run_pipeline(
    goal: str,
    config_path: str | None = None,
    unified_model: str | None = None,
    planner_model: str | None = None,
    coder_model: str | None = None,
    reviewer_model: str | None = None,
    resume_session_id: str | None = None,
    workspace: str | None = None,
    workflow_path: str | None = None,
    interactive_select: bool = False,
) -> None:
    """Core asynchronous execution pipeline for running Aglibol Agent workflows."""
    cfg = ConfigManager.load(config_path)
    profile = HardwareProfiler.detect()

    client = OllamaClient(host=cfg.ollama.host)
    try:
        is_alive = await client.health_check()
        if not is_alive:
            Display.print_error(
                f"Could not connect to Ollama daemon at {cfg.ollama.host}. Please start it with 'ollama serve'."
            )
            return

        # 1. Fetch live installed models from local Ollama daemon
        installed_model_infos = await client.list_models()
        installed_model_names = [m.name for m in installed_model_infos]

        # 2. Interactive model selection if requested via --select
        if interactive_select:
            if not installed_model_infos:
                console.print(
                    "[yellow]No models currently installed in local Ollama to select from.[/yellow]"
                )
            else:
                p_items = ModelCatalogHelper.classify_and_build_items(
                    installed_model_infos, "planner", cfg.models.planner
                )
                p_choice = TUISelector.choose(
                    title="Choose Model for PLANNER",
                    subtitle=f"Tier: {profile.tier.value.upper()}",
                    items=p_items,
                    default_index=0,
                    cancel_label=f"Default ({cfg.models.planner})",
                )
                planner_model = p_choice or cfg.models.planner

                c_items = ModelCatalogHelper.classify_and_build_items(
                    installed_model_infos, "coder", cfg.models.coder
                )
                c_choice = TUISelector.choose(
                    title="Choose Model for CODER",
                    subtitle=f"Tier: {profile.tier.value.upper()}",
                    items=c_items,
                    default_index=0,
                    cancel_label=f"Default ({cfg.models.coder})",
                )
                coder_model = c_choice or cfg.models.coder

                r_items = ModelCatalogHelper.classify_and_build_items(
                    installed_model_infos, "reviewer", cfg.models.reviewer
                )
                r_choice = TUISelector.choose(
                    title="Choose Model for REVIEWER",
                    subtitle=f"Tier: {profile.tier.value.upper()}",
                    items=r_items,
                    default_index=0,
                    cancel_label=f"Default ({cfg.models.reviewer})",
                )
                reviewer_model = r_choice or cfg.models.reviewer

        # 3. Model resolution: --model > flag > config > tier default > smart auto-resolution
        p_model = unified_model or planner_model or cfg.models.planner
        c_model = unified_model or coder_model or cfg.models.coder
        r_model = unified_model or reviewer_model or cfg.models.reviewer

        # Smart auto-resolution against installed models
        if installed_model_names:
            p_model, p_auto = resolve_installed_model_for_role(
                p_model, "planner", installed_model_names
            )
            c_model, c_auto = resolve_installed_model_for_role(
                c_model, "coder", installed_model_names
            )
            r_model, r_auto = resolve_installed_model_for_role(
                r_model, "reviewer", installed_model_names
            )

            if p_auto:
                console.print(
                    f"  [dim yellow]ℹ Auto-selected installed model '{p_model}' for Planner[/dim yellow]"
                )
            if c_auto:
                console.print(
                    f"  [dim yellow]ℹ Auto-selected installed model '{c_model}' for Coder[/dim yellow]"
                )
            if r_auto:
                console.print(
                    f"  [dim yellow]ℹ Auto-selected installed model '{r_model}' for Reviewer[/dim yellow]"
                )

        workspace_dir = Path(workspace).resolve() if workspace else Path.cwd().resolve()
        workspace_dir.mkdir(parents=True, exist_ok=True)

        console.print(f"[bold cyan][Goal][/bold cyan] {goal}")
        console.print(
            f"[dim]Hardware Tier: {profile.tier.value.upper()} | "
            f"Primary GPU: {profile.primary_gpu.name if profile.primary_gpu else 'CPU'}[/dim]"
        )
        console.print(f"[dim]Workspace Directory: {workspace_dir}[/dim]")
        console.print(
            f"[dim]Assigned Models: Planner={p_model}, Coder={c_model}, Reviewer={r_model}[/dim]\n"
        )

        # 4. Initialize EventBus & Live Telemetry UI
        event_bus = EventBus()

        async def _on_event(ev: AgentEvent) -> None:
            if ev.event_type == "MODEL_LOAD_STARTED":
                num_ctx = ev.data.get("num_ctx")
                ctx_desc = f" (context: {num_ctx:,} tokens)" if num_ctx else ""
                console.print(
                    f"  [yellow][Model Load][/yellow] [bold]{ev.data.get('model')}[/bold]{ctx_desc}"
                )
            elif ev.event_type == "MODEL_UNLOADED":
                console.print(f"  [dim][Model Evicted] {ev.data.get('model')} (freed VRAM)[/dim]")
            elif ev.event_type == "PLAN_CREATED":
                console.print(
                    f"  [green][Plan Created] Task plan created with {ev.data.get('task_count')} tasks[/green]"
                )
            elif ev.event_type == "TOOL_EXEC_STARTED":
                console.print(f"  [cyan][Calling Tool] {ev.data.get('tool')}[/cyan]")
            elif ev.event_type == "TOOL_EXEC_FINISHED":
                status = "[green]success[/green]" if ev.data.get("success") else "[red]failed[/red]"
                console.print(f"  [dim][Tool Finished] {status}[/dim]")
            elif ev.event_type == "CODE_GENERATED":
                console.print(
                    f"  [bold green][Code Generated] ({ev.data.get('length')} chars)[/bold green]"
                )
            elif ev.event_type == "REVIEW_COMPLETED":
                dec = ev.data.get("decision", "").upper()
                color = "green" if dec == "APPROVED" else "yellow"
                console.print(f"  [{color}][Review Outcome] {dec}[/{color}]")

        event_bus.subscribe("*", _on_event)

        # 5. Brain & Storage Hierarchy Setup
        brain = Brain(cfg.storage.brain_dir)
        sm = SessionManager(brain.sessions_dir)

        if resume_session_id:
            meta = sm.get_meta(resume_session_id)
            if not meta:
                Display.print_error(f"Session {resume_session_id} not found.")
                return
            session_id = resume_session_id
            session_dir = brain.get_session_dir(session_id)
            ckpt_store = CheckpointStore(session_dir / "checkpoints.db")
            initial_state = ckpt_store.get_latest_checkpoint(session_id)
            if not initial_state:
                Display.print_error(f"No valid checkpoints found for session {session_id}.")
                return
            initial_state.workspace_dir = str(workspace_dir)
            console.print(
                f"[green]Resuming session {session_id} from step {initial_state.current_step}...[/green]"
            )
        else:
            meta = sm.create_session(goal=goal, tier=profile.tier)
            session_id = meta.session_id
            session_dir = brain.get_session_dir(session_id)
            ckpt_store = CheckpointStore(session_dir / "checkpoints.db")
            initial_state = AgentState(
                session_id=session_id,
                user_goal=goal,
                workspace_dir=str(workspace_dir),
            )

        artifact_store = ArtifactStore(session_dir)
        working_memory = WorkingMemory()
        episodic_memory = EpisodicMemory(session_dir / "episodic.jsonl")

        # 7. Tool Registry Setup with Workspace Sandboxing
        tool_registry = ToolRegistry()
        register_file_tools(tool_registry, workspace_root=workspace_dir)
        register_shell_tools(tool_registry, workspace_root=workspace_dir)
        register_search_replace_tools(tool_registry, workspace_root=workspace_dir)

        # 8. Model Scheduler
        scheduler = ModelScheduler(client=client, profile=profile, event_bus=event_bus)

        # 9. Workflow Engine Setup (Dynamic YAML loading with fallback)
        target_wf_path = workflow_path or "default.yaml"
        model_overrides = {
            "planner": p_model,
            "coder": c_model,
            "reviewer": r_model,
        }

        engine = WorkflowEngine.build_from_yaml(
            yaml_path=target_wf_path,
            client=client,
            scheduler=scheduler,
            tool_registry=tool_registry,
            checkpoint_store=ckpt_store,
            artifact_store=artifact_store,
            event_bus=event_bus,
            working_memory=working_memory,
            episodic_memory=episodic_memory,
            model_overrides=model_overrides,
            max_total_steps=15,
        )

        # 10. Run Workflow Graph
        final_state = await engine.run(initial_state)

        # 11. Persist artifacts to workspace directory as well
        if final_state.artifacts:
            for fname, content in final_state.artifacts.items():
                dest = workspace_dir / fname
                try:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(content, encoding="utf-8")
                except Exception as e:
                    console.print(
                        f"[dim yellow]Warning: could not write {fname} to workspace: {e}[/dim yellow]"
                    )

        # 12. Update Session Status
        meta.status = (
            "completed"
            if (final_state.is_completed and final_state.status != "failed")
            else "failed"
        )
        meta.step_count = final_state.current_step
        sm.save_meta(meta)

        # 13. Report Outcome
        console.print("\n" + "=" * 60)
        if final_state.is_completed and final_state.status != "failed":
            Display.print_success("Task workflow completed successfully!")
        else:
            Display.print_error(
                f"Workflow ended with status [{final_state.status}]: {final_state.error or 'Incomplete'}"
            )

        if final_state.artifacts:
            console.print(
                Panel(
                    "\n".join(
                        [
                            f" • {k} ({len(v)} bytes) -> {workspace_dir / k}"
                            for k, v in final_state.artifacts.items()
                        ]
                    ),
                    title="Generated Files & Artifacts",
                    border_style="green",
                )
            )

        console.print(f"[dim]Session artifacts and state persisted to: {session_dir}[/dim]\n")

    finally:
        # Ensure httpx connections are properly cleaned up
        await client.close()


@click.command("run")
@click.argument("goal", type=str)
@click.option(
    "--config", "config_path", type=click.Path(exists=True), help="Custom YAML config path."
)
@click.option(
    "--workspace", type=click.Path(), default=None, help="Working directory for file generation."
)
@click.option(
    "--workflow",
    "workflow_path",
    type=click.Path(),
    default=None,
    help="Custom workflow YAML path.",
)
@click.option(
    "--model", "unified_model", type=str, default=None, help="Unified model override for ALL roles."
)
@click.option(
    "--select",
    "interactive_select",
    is_flag=True,
    help="Interactively select models from installed Ollama models.",
)
@click.option("--planner-model", type=str, help="Override Planner model.")
@click.option("--coder-model", type=str, help="Override Coder model.")
@click.option("--reviewer-model", type=str, help="Override Reviewer model.")
@click.option(
    "--resume", "resume_session_id", type=str, help="Resume an interrupted session by ID."
)
def run_cmd(
    goal: str,
    config_path: str | None,
    workspace: str | None,
    workflow_path: str | None,
    unified_model: str | None,
    interactive_select: bool,
    planner_model: str | None,
    coder_model: str | None,
    reviewer_model: str | None,
    resume_session_id: str | None,
) -> None:
    """Execute a task using the Aglibol Agent multi-agent workflow."""
    Display.print_banner()
    asyncio.run(
        execute_run_pipeline(
            goal=goal,
            config_path=config_path,
            unified_model=unified_model,
            planner_model=planner_model,
            coder_model=coder_model,
            reviewer_model=reviewer_model,
            resume_session_id=resume_session_id,
            workspace=workspace,
            workflow_path=workflow_path,
            interactive_select=interactive_select,
        )
    )
