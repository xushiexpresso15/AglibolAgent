"""Unit tests for SessionManager lifecycle, session deletion, ghost session pruning, and TUI custom specs."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from aglibol.cli.interactive.commands import SlashCommandHandler
from aglibol.cli.interactive.repl import InteractiveSession
from aglibol.cli.interactive.selector import SelectorItem, TUISelector
from aglibol.core.types import HardwareTier
from aglibol.storage.session import SessionManager


def test_session_manager_create_and_delete():
    with tempfile.TemporaryDirectory() as tmpdir:
        sm = SessionManager(Path(tmpdir))
        meta = sm.create_session(goal="Test goal", tier=HardwareTier.TIER_1_4GB)
        assert meta.session_id.startswith("sess_")
        assert (Path(tmpdir) / meta.session_id / "meta.json").exists()

        # Delete session
        assert sm.delete_session(meta.session_id) is True
        assert not (Path(tmpdir) / meta.session_id).exists()

        # Attempting to delete non-existent session returns False
        assert sm.delete_session("non_existent_id") is False


def test_session_manager_prune_empty_sessions():
    with tempfile.TemporaryDirectory() as tmpdir:
        sm = SessionManager(Path(tmpdir))

        # 1. Create a ghost session (empty)
        ghost = sm.create_session(goal="Interactive REPL Session", tier=HardwareTier.TIER_1_4GB)

        # 2. Create a meaningful session
        real = sm.create_session(goal="Implement user authentication", tier=HardwareTier.TIER_1_4GB)
        real.step_count = 2
        sm.save_meta(real)

        # Check is_empty_session
        assert sm.is_empty_session(ghost) is True
        assert sm.is_empty_session(real) is False

        # Prune empty sessions
        pruned_count = sm.prune_empty_sessions()
        assert pruned_count == 1

        # Verify ghost is gone and real remains
        assert sm.get_meta(ghost.session_id) is None
        assert sm.get_meta(real.session_id) is not None


def test_session_listing_filters_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        sm = SessionManager(Path(tmpdir))

        # Ghost session
        sm.create_session(goal="Interactive REPL Session", tier=HardwareTier.TIER_1_4GB)

        # Meaningful session with checkpoint
        meaningful = sm.create_session(
            goal="Optimize database queries", tier=HardwareTier.TIER_1_4GB
        )
        meaningful.step_count = 1
        sm.save_meta(meaningful)

        # Filtered by default
        active_sessions = sm.list_sessions(filter_empty=True)
        assert len(active_sessions) == 1
        assert active_sessions[0].session_id == meaningful.session_id

        # Unfiltered includes both
        all_sessions = sm.list_all_sessions(filter_empty=False)
        assert len(all_sessions) == 2


def test_tui_selector_custom_specs(monkeypatch):
    items = [
        SelectorItem(key="s1", label="Session 1", specs="Active session", badge="[OK]"),
    ]
    monkeypatch.setattr("rich.prompt.Prompt.ask", lambda prompt, default="": "1")
    chosen = TUISelector.choose(
        title="Test Selector",
        items=items,
        allow_custom=True,
        custom_specs="Enter custom session ID to restore",
    )
    assert chosen == "s1"


@pytest.mark.asyncio
async def test_slash_command_sessions_delete_and_prune():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        session = InteractiveSession(workspace_root=root)
        handler = SlashCommandHandler(session)

        # Create a historical session to delete
        target = session.sm.create_session(goal="Old completed task", tier=HardwareTier.TIER_1_4GB)
        target.step_count = 3
        target.status = "completed"
        session.sm.save_meta(target)

        # Delete historical session via slash command with force flag
        assert await handler.handle(f"/sessions delete {target.session_id} -y") is True
        assert session.sm.get_meta(target.session_id) is None

        # Cannot delete active running session
        assert await handler.handle(f"/sessions delete {session.session_id}") is True
        assert session.sm.get_meta(session.session_id) is not None

        # Run prune command
        assert await handler.handle("/sessions prune") is True


@pytest.mark.asyncio
async def test_resume_updates_session_meta_and_cleans_prior_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        session = InteractiveSession(workspace_root=root)
        handler = SlashCommandHandler(session)
        initial_sid = session.session_id

        # Create a historical session with a goal
        target = session.sm.create_session(goal="Build a web crawler", tier=HardwareTier.TIER_1_4GB)
        target.step_count = 5
        target.status = "completed"
        session.sm.save_meta(target)

        # Resume the target session
        assert await handler.handle(f"/resume {target.session_id}") is True

        # Verify active session updated
        assert session.session_id == target.session_id
        assert session.meta.session_id == target.session_id
        assert session.meta.user_goal == "Build a web crawler"

        # Verify the prior unused empty session was cleaned up
        assert session.sm.get_meta(initial_sid) is None


@pytest.mark.asyncio
async def test_slash_command_unified_session_actions():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        session = InteractiveSession(workspace_root=root)
        handler = SlashCommandHandler(session)

        # 1. Test /session list
        assert await handler.handle("/session list") is True

        # 2. Create historical sessions
        s1 = session.sm.create_session(goal="First task", tier=HardwareTier.TIER_1_4GB)
        s1.step_count = 2
        s1.status = "completed"
        session.sm.save_meta(s1)

        s2 = session.sm.create_session(goal="Second task", tier=HardwareTier.TIER_1_4GB)
        s2.step_count = 4
        s2.status = "completed"
        session.sm.save_meta(s2)

        # 3. Test /session <id> direct resume
        assert await handler.handle(f"/session {s1.session_id}") is True
        assert session.session_id == s1.session_id

        # 4. Test /session resume <id>
        assert await handler.handle(f"/session resume {s2.session_id}") is True
        assert session.session_id == s2.session_id

        # 5. Test /session delete <id> -y
        assert await handler.handle(f"/session delete {s1.session_id} -y") is True
        assert session.sm.get_meta(s1.session_id) is None

        # 6. Test /session prune
        assert await handler.handle("/session prune") is True


@pytest.mark.asyncio
async def test_interactive_tui_session_deletion(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        session = InteractiveSession(workspace_root=root)
        handler = SlashCommandHandler(session)

        target = session.sm.create_session(
            goal="To be deleted via TUI", tier=HardwareTier.TIER_1_4GB
        )
        target.step_count = 3
        target.status = "completed"
        session.sm.save_meta(target)

        # 1. User selects target session but cancels confirmation dialog -> session preserved
        choose_calls_cancel = [target.session_id, "cancel"]
        monkeypatch.setattr(TUISelector, "choose", lambda **kwargs: choose_calls_cancel.pop(0))
        assert await handler.handle("/session delete") is True
        assert session.sm.get_meta(target.session_id) is not None

        # 2. User selects target session and confirms deletion -> session deleted
        choose_calls_confirm = [target.session_id, "confirm"]
        monkeypatch.setattr(TUISelector, "choose", lambda **kwargs: choose_calls_confirm.pop(0))
        assert await handler.handle("/session delete") is True
        assert session.sm.get_meta(target.session_id) is None


@pytest.mark.asyncio
async def test_interactive_tui_session_resumption(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        session = InteractiveSession(workspace_root=root)
        handler = SlashCommandHandler(session)

        target = session.sm.create_session(
            goal="To be resumed via TUI", tier=HardwareTier.TIER_1_4GB
        )
        target.step_count = 3
        target.status = "completed"
        session.sm.save_meta(target)

        # Mock TUISelector.choose to simulate user selecting target session
        monkeypatch.setattr(TUISelector, "choose", lambda **kwargs: target.session_id)

        # Run /session resume with no args (triggers TUI picker)
        assert await handler.handle("/session resume") is True
        assert session.session_id == target.session_id


@pytest.mark.asyncio
async def test_interactive_tui_session_hub_flow(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        session = InteractiveSession(workspace_root=root)
        handler = SlashCommandHandler(session)

        # Create a session
        s = session.sm.create_session(goal="Hub item", tier=HardwareTier.TIER_1_4GB)
        s.step_count = 1
        session.sm.save_meta(s)

        # Simulate user choosing "list", then exiting (returning None)
        actions = ["list", None]
        monkeypatch.setattr(TUISelector, "choose", lambda **kwargs: actions.pop(0))

        # Launch /session without arguments
        assert await handler.handle("/session") is True
