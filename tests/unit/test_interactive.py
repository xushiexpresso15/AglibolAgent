"""Unit tests for Phase D: Interactive REPL, Completer, HITL SecurityGate, and Slash Commands."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from prompt_toolkit.document import Document

from aglibol.cli.interactive.commands import SlashCommandHandler
from aglibol.cli.interactive.completer import AgentInputCompleter
from aglibol.cli.interactive.hitl import SafetyMode, SecurityGate
from aglibol.cli.interactive.repl import InteractiveSession
from aglibol.cli.interactive.tui_renderer import TUIRenderer

# --- 1. Smart Completer Tests ---


def test_completer_slash_commands():
    completer = AgentInputCompleter()

    # Document with just "/"
    doc = Document(text="/", cursor_position=1)
    completions = list(completer.get_completions(doc, None))
    comp_texts = [c.text for c in completions]

    assert "/help" in comp_texts
    assert "/status" in comp_texts
    assert "/model" in comp_texts
    assert "/undo" in comp_texts
    assert "/diff" in comp_texts
    assert "/context" in comp_texts

    # Document with "/st"
    doc_st = Document(text="/st", cursor_position=3)
    completions_st = list(completer.get_completions(doc_st, None))
    comp_texts_st = [c.text for c in completions_st]

    assert "/status" in comp_texts_st
    assert "/help" not in comp_texts_st


def test_completer_at_file_mentions():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "calculator.py").write_text("def add(a, b): return a + b\n")
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_calc.py").write_text("def test(): pass\n")

        completer = AgentInputCompleter(workspace_root=tmp_path)

        # Document with "@calc"
        doc = Document(text="Please inspect @calc", cursor_position=len("Please inspect @calc"))
        completions = list(completer.get_completions(doc, None))
        comp_texts = [c.display_text for c in completions]

        assert any("calculator.py" in d for d in comp_texts)


# --- 2. Human-in-the-Loop Security Gate Tests ---


def test_hitl_security_gate_policies():
    # Autonomous mode: never prompts
    gate_auto = SecurityGate(mode=SafetyMode.AUTONOMOUS)
    assert gate_auto.should_prompt("execute_shell", {"command": "dir"}) is False
    assert gate_auto.should_prompt("write_file", {"path": "main.py"}) is False

    # Strict mode: always prompts
    gate_strict = SecurityGate(mode=SafetyMode.STRICT)
    assert gate_strict.should_prompt("read_file", {"path": "main.py"}) is True
    assert gate_strict.should_prompt("execute_shell", {"command": "dir"}) is True

    # Balanced mode: prompts for shell and critical configs, but allows read_file
    gate_balanced = SecurityGate(mode=SafetyMode.BALANCED)
    assert gate_balanced.should_prompt("execute_shell", {"command": "dir"}) is True
    assert gate_balanced.should_prompt("read_file", {"path": "main.py"}) is False
    assert gate_balanced.should_prompt("write_file", {"path": "settings.json"}) is True

    # Always allowed session whitelist
    gate_balanced.always_allowed_tools.add("execute_shell")
    assert gate_balanced.should_prompt("execute_shell", {"command": "dir"}) is False


# --- 3. Slash Command Handler Tests ---


@pytest.mark.asyncio
async def test_slash_command_handler():
    with tempfile.TemporaryDirectory() as tmpdir:
        session = InteractiveSession(workspace_root=Path(tmpdir))
        handler = SlashCommandHandler(session)

        # 1. Non-slash prompt returns False
        handled = await handler.handle("build an app")
        assert handled is False

        # 2. /help returns True
        assert await handler.handle("/help") is True

        # 3. /model changes assignment
        assert await handler.handle("/model coder qwen2.5-coder:14b") is True
        assert session.model_overrides["coder"] == "qwen2.5-coder:14b"

        # 4. /add pins file into context
        test_file = Path(tmpdir) / "sample.py"
        test_file.write_text("x = 42\n")
        assert await handler.handle("/add sample.py") is True
        assert "sample.py" in session.pinned_files

        # 5. /context
        assert await handler.handle("/context") is True

        # 6. /drop unpins file
        assert await handler.handle("/drop sample.py") is True
        assert "sample.py" not in session.pinned_files

        # 7. /compact
        assert await handler.handle("/compact") is True

        # 8. /init creates .aglibol.yaml and AGENTS.md
        assert await handler.handle("/init") is True
        assert (Path(tmpdir) / ".aglibol.yaml").exists()
        assert (Path(tmpdir) / "AGENTS.md").exists()

        # 9. /exit sets should_exit flag
        assert await handler.handle("/exit") is True
        assert session.should_exit is True


# --- 4. TUI Renderer Visual Components Tests ---


def test_tui_renderer_diff_and_budget_bar():
    # Test diff rendering
    old_code = "def foo():\n    return 1\n"
    new_code = "def foo():\n    return 2\n"
    # Should run without error
    TUIRenderer.render_diff("app.py", old_code, new_code)

    # Test context budget bar rendering
    TUIRenderer.render_context_budget_bar(
        used_tokens=1500, max_tokens=2048, pinned_files_tokens=300
    )


# --- 5. Multi-Level Parameter Completion Tests ---


def test_completer_multi_level_parameters():
    with tempfile.TemporaryDirectory() as tmpdir:
        session = InteractiveSession(workspace_root=Path(tmpdir))
        session.installed_models_cache = ["qwen2.5-coder:7b", "llama3.1:8b"]
        session.pinned_files = {"src/app.py": "print('hello')", "README.md": "# Readme"}

        completer = AgentInputCompleter(session=session, workspace_root=Path(tmpdir))

        # 1. /model <space> -> suggests roles
        doc_model_space = Document(text="/model ")
        comps = [c.text for c in completer.get_completions(doc_model_space, None)]
        assert "planner" in comps
        assert "coder" in comps
        assert "reviewer" in comps

        # 2. /model coder <space> -> suggests installed models
        doc_coder_space = Document(text="/model coder ")
        comps_models = [c.text for c in completer.get_completions(doc_coder_space, None)]
        assert "qwen2.5-coder:7b" in comps_models
        assert "llama3.1:8b" in comps_models

        # 3. /mode <space> -> suggests safety modes
        doc_mode_space = Document(text="/mode ")
        comps_mode = [c.text for c in completer.get_completions(doc_mode_space, None)]
        assert "balanced" in comps_mode
        assert "strict" in comps_mode
        assert "autonomous" in comps_mode

        # 4. /mode a -> filters to auto and autonomous
        doc_mode_a = Document(text="/mode a")
        comps_mode_a = [c.text for c in completer.get_completions(doc_mode_a, None)]
        assert "auto" in comps_mode_a
        assert "autonomous" in comps_mode_a

        # 5. /safety <space> -> suggests safety modes
        doc_safety_space = Document(text="/safety ")
        comps_safety = [c.text for c in completer.get_completions(doc_safety_space, None)]
        assert "balanced" in comps_safety
        assert "strict" in comps_safety
        assert "autonomous" in comps_safety

        # 6. /drop <space> -> suggests pinned files
        doc_drop = Document(text="/drop ")
        comps_drop = [c.text for c in completer.get_completions(doc_drop, None)]
        assert "src/app.py" in comps_drop
        assert "README.md" in comps_drop

        # 7. /workflow <space> -> suggests workflow templates
        doc_wf = Document(text="/workflow ")
        comps_wf = [c.text for c in completer.get_completions(doc_wf, None)]
        assert "default" in comps_wf
        assert "fast" in comps_wf


# --- 6. /mode, /resume, /sessions Slash Command Execution Tests ---


@pytest.mark.asyncio
async def test_slash_command_mode_switch():
    with tempfile.TemporaryDirectory() as tmpdir:
        session = InteractiveSession(workspace_root=Path(tmpdir))
        handler = SlashCommandHandler(session)

        # Default is balanced safety and auto agent mode
        assert session.security_gate.mode == SafetyMode.BALANCED
        assert session.agent_mode.value == "auto"

        # Switch agent mode to planner
        assert await handler.handle("/mode planner") is True
        assert session.agent_mode.value == "planner"

        # Switch agent mode to coder
        assert await handler.handle("/mode coder") is True
        assert session.agent_mode.value == "coder"

        # Switch safety mode via /safety
        assert await handler.handle("/safety strict") is True
        assert session.security_gate.mode == SafetyMode.STRICT

        # Backward compatible: /mode autonomous sets safety
        assert await handler.handle("/mode autonomous") is True
        assert session.security_gate.mode == SafetyMode.AUTONOMOUS

        # Invalid mode doesn't crash
        assert await handler.handle("/mode invalid_mode") is True


@pytest.mark.asyncio
async def test_slash_command_resume_and_sessions():
    with tempfile.TemporaryDirectory() as tmpdir:
        session = InteractiveSession(workspace_root=Path(tmpdir))
        handler = SlashCommandHandler(session)

        # 1. /sessions command lists historical sessions without error
        assert await handler.handle("/sessions") is True

        # 2. Create a prior session and record a checkpoint
        from aglibol.core.types import AgentState, ChatMessage, HardwareTier
        from aglibol.storage.checkpoint import CheckpointStore

        past_meta = session.sm.create_session("Build CLI agent", HardwareTier.TIER_2_6GB)
        past_dir = session.brain.get_session_dir(past_meta.session_id)
        ckpt_store = CheckpointStore(past_dir / "checkpoints.db")

        test_state = AgentState(
            session_id=past_meta.session_id,
            user_goal="Build CLI agent",
            current_step=4,
            active_agent="coder",
            messages=[ChatMessage(role="user", content="Build CLI agent")],
            artifacts={"cli.py": "print('cli running')"},
            scratchpad={"key": "saved_value"},
        )
        ckpt_store.save_checkpoint(test_state)

        # 3. Test /resume <session_id> completion
        completer = AgentInputCompleter(session=session)
        doc_res = Document(text="/resume ")
        res_comps = [c.text for c in completer.get_completions(doc_res, None)]
        assert past_meta.session_id in res_comps

        # 4. Execute /resume <session_id>
        assert await handler.handle(f"/resume {past_meta.session_id}") is True
        assert session.session_id == past_meta.session_id
        assert session.current_state.user_goal == "Build CLI agent"
        assert len(session.history) == 1
        assert session.working_memory.get("key") == "saved_value"

        # Check restored artifact written to workspace
        restored_file = Path(tmpdir) / "cli.py"
        assert restored_file.exists()
        assert restored_file.read_text(encoding="utf-8") == "print('cli running')"


def test_repl_bottom_toolbar():
    with tempfile.TemporaryDirectory() as tmpdir:
        session = InteractiveSession(workspace_root=Path(tmpdir), safety_mode=SafetyMode.BALANCED)
        toolbar_html = session.get_bottom_toolbar()
        raw_html = toolbar_html.formatted_text

        # Combine text fragments
        text = "".join(t[1] for t in raw_html)
        assert "Model:" in text
        assert "Safety:" in text
        assert "BALANCED" in text
        assert "Context:" in text


@pytest.mark.asyncio
async def test_binary_file_rejection():
    with tempfile.TemporaryDirectory() as tmpdir:
        session = InteractiveSession(workspace_root=Path(tmpdir))
        handler = SlashCommandHandler(session)
        png_path = Path(tmpdir) / "image.png"
        png_path.write_bytes(b"\x89PNG\r\n\x1a\n")
        await handler.handle("/add image.png")
        assert "image.png" not in session.pinned_files


def test_at_mention_invalid_path_warns():
    with tempfile.TemporaryDirectory() as tmpdir:
        session = InteractiveSession(workspace_root=Path(tmpdir))
        session._extract_and_pin_at_mentions("@../../../etc/passwd")
        assert "../../../etc/passwd" not in session.pinned_files


def test_artifact_store_path_traversal():
    from aglibol.storage.artifact_store import ArtifactStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = ArtifactStore(Path(tmpdir))
        with pytest.raises(ValueError):
            store.save_artifact("../../../evil.txt", "data")


def test_brain_clean_session_path_traversal():
    from aglibol.storage.brain import Brain

    with tempfile.TemporaryDirectory() as tmpdir:
        brain = Brain(Path(tmpdir))
        assert brain.clean_session("..") is False


def test_session_manager_path_traversal():
    from aglibol.storage.session import SessionManager

    with tempfile.TemporaryDirectory() as tmpdir:
        sm = SessionManager(Path(tmpdir))
        assert sm.get_meta("../../../etc") is None


def test_hitl_edit_non_shell_loops():
    gate = SecurityGate(mode=SafetyMode.STRICT)
    assert gate.should_prompt("write_file", {"path": "test.py"}) is True


def test_balanced_mode_blocks_hidden_files():
    gate = SecurityGate(mode=SafetyMode.BALANCED)
    assert gate.should_prompt("write_file", {"path": ".bashrc"}) is True
    assert gate.should_prompt("write_file", {"path": "normal.py"}) is False


def test_live_tool_container_lifecycle():
    from aglibol.cli.interactive.tui_renderer import LiveToolContainer

    container = LiveToolContainer()

    # Initially empty
    assert len(container.entries) == 0

    # Add tool 1
    container.add_tool_start("write_file", {"path": "src/app.py", "content": "print(1)"})
    assert len(container.entries) == 1
    assert container.entries[0].tool_name == "write_file"
    assert container.entries[0].status == "running"

    panel_running = container._build_panel()
    assert "[Tools Execution]" in panel_running.title

    # Finish tool 1
    container.update_tool_finish("write_file", success=True, output="Saved 15 bytes")
    assert container.entries[0].status == "done"
    assert container.entries[0].success is True

    # Add tool 2 (fails)
    container.add_tool_start("execute_shell", {"command": "pytest -k missing"})
    assert len(container.entries) == 2
    assert container.entries[1].status == "running"

    # Finish tool 2 with error
    container.update_tool_finish("execute_shell", success=False, error="Exit code 1")
    assert container.entries[1].status == "done"
    assert container.entries[1].success is False
    assert container.entries[1].error == "Exit code 1"

    container.finish()
    panel_finished = container._build_panel()
    assert "[Tools Execution ✓]" in panel_finished.title


def test_live_thought_box_lifecycle():
    from aglibol.cli.interactive.tui_renderer import LiveThoughtBox

    box = LiveThoughtBox(agent_name="coder")
    box.update_token("Analyzing objective...\n")
    box.update_token("Step 1: implement helper\n")
    assert len(box.lines) == 2
    assert box.lines[0] == "Analyzing objective..."
    assert box.lines[1] == "Step 1: implement helper"

    panel = box._build_panel(final=False)
    assert "Coder Thinking" in str(panel.title)

    box.finish(summary="Completed plan")
    assert box._is_finished is True
    panel_fin = box._build_panel()
    assert "Coder Thinking ✓" in str(panel_fin.title)


@pytest.mark.asyncio
async def test_repl_tool_events_consolidated_container():
    from aglibol.core.events import AgentEvent

    with tempfile.TemporaryDirectory() as tmpdir:
        session = InteractiveSession(workspace_root=Path(tmpdir))

        assert session.active_tool_container is None

        # Emit tool start event
        await session.event_bus.emit(
            AgentEvent(
                event_type="TOOL_EXEC_STARTED",
                session_id=session.session_id,
                step=1,
                agent="coder",
                data={"tool": "write_file", "args": {"path": "main.py"}},
            )
        )
        assert session.active_tool_container is not None
        assert len(session.active_tool_container.entries) == 1
        assert session.active_tool_container.entries[0].status == "running"

        # Emit tool finish event
        await session.event_bus.emit(
            AgentEvent(
                event_type="TOOL_EXEC_FINISHED",
                session_id=session.session_id,
                step=1,
                agent="coder",
                data={"tool": "write_file", "success": True, "output": "ok"},
            )
        )
        assert session.active_tool_container.entries[0].status == "done"
        assert session.active_tool_container.entries[0].success is True

        # Clean up
        if session.active_tool_container:
            session.active_tool_container.finish()
            session.active_tool_container = None


@pytest.mark.asyncio
async def test_execute_coder_turn_critique_loop():
    from unittest.mock import patch

    from aglibol.core.types import AgentState

    with tempfile.TemporaryDirectory() as tmpdir:
        session = InteractiveSession(workspace_root=Path(tmpdir))
        state = AgentState(
            session_id=session.session_id,
            user_goal="Implement calculator",
            workspace_dir=tmpdir,
        )

        coder_runs = 0
        reviewer_runs = 0

        async def mock_coder_exec(*args, **kwargs):
            nonlocal coder_runs
            coder_runs += 1
            st = kwargs.get("state") or args[0]
            st.artifacts["calc.py"] = f"# version {coder_runs}"
            return st

        async def mock_reviewer_exec(*args, **kwargs):
            nonlocal reviewer_runs
            reviewer_runs += 1
            st = kwargs.get("state") or args[0]
            if reviewer_runs < 2:
                st.review_status = "rejected"
                st.review_feedback = "Syntax error on line 5"
            else:
                st.review_status = "approved"
                st.review_feedback = "Looks great"
                st.is_completed = True
            return st

        with (
            patch("aglibol.cli.interactive.repl.CoderAgent.execute", side_effect=mock_coder_exec),
            patch(
                "aglibol.cli.interactive.repl.ReviewerAgent.execute", side_effect=mock_reviewer_exec
            ),
        ):
            await session._execute_coder_turn(state)

        # Coder ran twice (initial + 1 fix cycle), Reviewer ran twice (rejected + approved)
        assert coder_runs == 2
        assert reviewer_runs == 2
        assert state.review_status == "approved"
        assert (Path(tmpdir) / "calc.py").exists()
        assert (Path(tmpdir) / "calc.py").read_text(encoding="utf-8") == "# version 2"


def test_live_activity_view_lifecycle():
    from aglibol.cli.interactive.tui_renderer import LiveActivityView

    view = LiveActivityView()

    assert view.current_agent == "assistant"
    view.start("coder", "Generating code solution and tests...")
    assert view.current_agent == "coder"

    view.update_thought("Analyzing prime algorithm logic...\n")
    assert "prime algorithm" in view.current_thought

    view.add_tool_start("list_dir", {"path": "workspace"})
    assert len(view.tool_entries) == 1
    assert view.tool_entries[0].status == "running"

    view.update_tool_finish("list_dir", True, "Listed 2 files")
    assert view.tool_entries[0].status == "done"
    assert view.tool_entries[0].success is True

    # Test topic milestone detection and dynamic rolling ticker
    view.update_thought("1. Architecture Plan:\n")
    assert view.current_thought_topic == "1. Architecture Plan:"

    long_thought = "We need to create a modular layout containing header, navigation bar, hero banner with call to action button, and footer."
    view.update_thought(long_thought)
    panel = view._build_panel()
    panel_text = panel.renderable.plain
    assert "..." in panel_text
    assert "footer" in panel_text
    assert "1. Architecture Plan:" in panel_text

    # Transition agent in-place to reviewer
    view.set_agent("reviewer", "Auditing code syntax, correctness, and security...")
    assert view.current_agent == "reviewer"

    panel_running = view._build_panel()
    assert "Reviewer" in str(panel_running.title)

    # Set review outcome and finish
    view.set_review_outcome("approved")
    assert view.review_decision == "approved"

    view.finish()
    assert view._is_finished is True
    panel_fin = view._build_panel()
    assert "Execution Complete ✓" in str(panel_fin.title)


def test_render_code_artifact():
    from aglibol.cli.interactive.tui_renderer import TUIRenderer

    # Test rendering code for python and bash
    py_code = "def is_prime(n):\n    return n > 1\n"
    TUIRenderer.render_code_artifact("is_prime.py", py_code)

    sh_code = "#!/bin/bash\necho hello\n"
    TUIRenderer.render_code_artifact("script.sh", sh_code)


@pytest.mark.asyncio
async def test_execute_coder_turn_renders_artifacts_with_syntax():
    from unittest.mock import patch

    from aglibol.core.types import AgentState

    with tempfile.TemporaryDirectory() as tmpdir:
        session = InteractiveSession(workspace_root=Path(tmpdir))
        state = AgentState(
            session_id=session.session_id,
            user_goal="Write is_prime.py",
            workspace_dir=tmpdir,
        )

        rendered_files = []

        async def mock_coder_exec(*args, **kwargs):
            st = kwargs.get("state") or args[0]
            st.artifacts["is_prime.py"] = "def is_prime(n): return n > 1"
            return st

        async def mock_reviewer_exec(*args, **kwargs):
            st = kwargs.get("state") or args[0]
            st.review_status = "approved"
            st.is_completed = True
            return st

        def mock_render_artifact(filename, code, title=None):
            rendered_files.append(filename)

        with (
            patch("aglibol.cli.interactive.repl.CoderAgent.execute", side_effect=mock_coder_exec),
            patch(
                "aglibol.cli.interactive.repl.ReviewerAgent.execute", side_effect=mock_reviewer_exec
            ),
            patch(
                "aglibol.cli.interactive.tui_renderer.TUIRenderer.render_code_artifact",
                side_effect=mock_render_artifact,
            ),
        ):
            await session._execute_coder_turn(state)

        assert "is_prime.py" in rendered_files
        assert (Path(tmpdir) / "is_prime.py").exists()
        assert (Path(tmpdir) / "is_prime.py").read_text(
            encoding="utf-8"
        ) == "def is_prime(n): return n > 1"


def test_unified_thinking_all_modes():
    """Verify that LiveActivityView standardizes dynamic thinking and drafting telemetry across all modes."""
    from aglibol.cli.interactive.tui_renderer import LiveActivityView

    view = LiveActivityView()

    # 1. Coder mode: Thinking phase
    view.set_agent("coder", "Analyzing requirements...")
    view.update_thought("Examining file layout and preparing HTML structure...", is_thinking=True)
    panel = view._build_panel()
    assert "Thinking:" in panel.renderable.plain
    assert "HTML structure" in panel.renderable.plain

    # 2. Coder mode: Implementing phase
    view.update_thought(
        "<!DOCTYPE html><html><body><h1>Title</h1></body></html>", is_thinking=False
    )
    panel = view._build_panel()
    assert "Implementing:" in panel.renderable.plain
    assert "Title" in panel.renderable.plain

    # 3. Writer mode: Thinking phase
    view.set_agent("writer", "Outlining prose...")
    view.update_thought(
        "Structuring narrative arc: introduction, development, resolution", is_thinking=True
    )
    panel = view._build_panel()
    assert "Thinking:" in panel.renderable.plain
    assert "narrative arc" in panel.renderable.plain

    # 4. Writer mode: Drafting phase
    view.update_thought(
        "The morning mist drifted quietly across the sleepy mountain valley...", is_thinking=False
    )
    panel = view._build_panel()
    assert "Drafting:" in panel.renderable.plain
    assert "mountain valley" in panel.renderable.plain

    # 5. Planner mode: Formulating plan
    view.set_agent("planner", "Formulating task architecture...")
    view.update_thought(
        '{"summary": "Decompose website into modular index.html and style.css components"}',
        is_thinking=False,
    )
    panel = view._build_panel()
    assert "Formulating Plan:" in panel.renderable.plain
    assert "Decompose website" in panel.renderable.plain

    # 6. Reviewer mode: Auditing solution
    view.set_agent("reviewer", "Auditing code...")
    view.update_thought(
        '{"feedback": "All requirements satisfied, syntax is valid without defects."}',
        is_thinking=False,
    )
    panel = view._build_panel()
    assert "Auditing:" in panel.renderable.plain
    assert "syntax is valid" in panel.renderable.plain

    # 7. Chat/Assistant mode: Responding
    view.set_agent("assistant", "Thinking...")
    view.update_thought(
        "Hello! I am Aglibol Agent, how can I assist you with your project today?",
        is_thinking=False,
    )
    panel = view._build_panel()
    assert "Generating:" in panel.renderable.plain
    assert "Aglibol Agent" in panel.renderable.plain
