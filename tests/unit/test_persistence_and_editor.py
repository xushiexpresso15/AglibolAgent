"""Unit tests for configuration persistence, editor keybindings, clipboard, and logo rendering."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.selection import SelectionType

from aglibol.cli.interactive.clipboard import get_clipboard_text, set_clipboard_text
from aglibol.cli.interactive.commands import SlashCommandHandler
from aglibol.cli.interactive.hitl import SafetyMode
from aglibol.cli.interactive.keybindings import build_editor_keybindings
from aglibol.cli.interactive.logo import render_logo
from aglibol.cli.interactive.repl import InteractiveSession
from aglibol.core.config import AppConfig, ConfigManager
from aglibol.core.types import AgentMode


def test_logo_render():
    """Verify the ASCII logo renders into Rich Text without errors."""
    text = render_logo()
    assert len(text) > 3000
    assert len(text.plain.splitlines()) == 40


def test_clipboard_roundtrip():
    """Verify clipboard get and set operations."""
    sample = "Aglibol Agent Test Payload 42"
    assert set_clipboard_text(sample) is True
    assert get_clipboard_text() == sample


def test_config_persistence_and_reload(monkeypatch):
    """Verify all user settings are persisted to config.yaml and reloaded."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_home = Path(tmpdir)
        monkeypatch.setattr(Path, "home", lambda: fake_home)
        monkeypatch.setattr(
            "os.path.expanduser", lambda path: str(fake_home / path.replace("~/", ""))
        )

        # 1. Persist models
        ConfigManager.set_user_config_value("models.coder", "custom-coder:14b")
        ConfigManager.set_user_config_value("models.chat", "custom-chat:8b")
        ConfigManager.set_user_config_value("models.planner", "custom-planner:8b")

        # 2. Persist agent_mode and safety_mode
        ConfigManager.set_user_config_value("agent_mode", "coder")
        ConfigManager.set_user_config_value("safety_mode", "autonomous")

        # 3. Persist workflow and default workspace
        ConfigManager.set_user_config_value("workflow.name", "review_strict")
        ConfigManager.set_user_config_value("default_workspace", str(fake_home / "project_alpha"))

        # Verify config file on disk
        cfg_file = fake_home / ".aglibol" / "config.yaml"
        assert cfg_file.exists()
        with open(cfg_file, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["aglibol"]["models"]["coder"] == "custom-coder:14b"
        assert data["aglibol"]["agent_mode"] == "coder"
        assert data["aglibol"]["safety_mode"] == "autonomous"
        assert data["aglibol"]["workflow"]["name"] == "review_strict"
        assert data["aglibol"]["default_workspace"] == str(fake_home / "project_alpha")

        # Reload via ConfigManager
        app_cfg: AppConfig = ConfigManager.load()
        assert app_cfg.models.coder == "custom-coder:14b"
        assert app_cfg.models.chat == "custom-chat:8b"
        assert app_cfg.agent_mode == "coder"
        assert app_cfg.safety_mode == "autonomous"
        assert app_cfg.workflow.name == "review_strict"
        assert app_cfg.default_workspace == str(fake_home / "project_alpha")


@pytest.mark.asyncio
async def test_session_restores_saved_preferences(monkeypatch):
    """Verify InteractiveSession initializes with values saved in ~/.aglibol/config.yaml."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_home = Path(tmpdir)
        monkeypatch.setattr(Path, "home", lambda: fake_home)
        monkeypatch.setattr(
            "os.path.expanduser", lambda path: str(fake_home / path.replace("~/", ""))
        )

        # Pre-seed user config
        ConfigManager.set_user_config_value("agent_mode", "reviewer")
        ConfigManager.set_user_config_value("safety_mode", "strict")
        ConfigManager.set_user_config_value("models.chat", "my-chat:7b")
        ConfigManager.set_user_config_value("models.coder", "my-coder:7b")

        session = InteractiveSession()
        assert session.agent_mode == AgentMode.REVIEWER
        assert session.security_gate.mode == SafetyMode.STRICT
        assert session.model_overrides["chat"] == "my-chat:7b"
        assert session.model_overrides["coder"] == "my-coder:7b"


@pytest.mark.asyncio
async def test_slash_command_persists_settings(monkeypatch):
    """Verify slash commands update both in-memory session and persisted YAML."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_home = Path(tmpdir)
        monkeypatch.setattr(Path, "home", lambda: fake_home)
        monkeypatch.setattr(
            "os.path.expanduser", lambda path: str(fake_home / path.replace("~/", ""))
        )

        session = InteractiveSession(workspace_root=fake_home / "ws")
        handler = SlashCommandHandler(session)

        # 1. /model coder new-model:7b
        await handler.handle("/model coder new-model:7b")
        assert session.model_overrides["coder"] == "new-model:7b"
        cfg = ConfigManager.load()
        assert cfg.models.coder == "new-model:7b"

        # 2. /mode writer
        await handler.handle("/mode writer")
        assert session.agent_mode == AgentMode.WRITER
        cfg = ConfigManager.load()
        assert cfg.agent_mode == "writer"

        # 3. /safety autonomous
        await handler.handle("/safety autonomous")
        assert session.security_gate.mode == SafetyMode.AUTONOMOUS
        cfg = ConfigManager.load()
        assert cfg.safety_mode == "autonomous"

        # 4. /workflow custom_wf
        await handler.handle("/workflow custom_wf")
        assert session.workflow_name == "custom_wf"
        cfg = ConfigManager.load()
        assert cfg.workflow.name == "custom_wf"


def test_auto_resolve_models_persists(monkeypatch):
    """Verify auto_resolve_models updates config file so user isn't warned on every restart."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_home = Path(tmpdir)
        monkeypatch.setattr(Path, "home", lambda: fake_home)
        monkeypatch.setattr(
            "os.path.expanduser", lambda path: str(fake_home / path.replace("~/", ""))
        )

        session = InteractiveSession(workspace_root=fake_home / "ws")
        # Installed has only coder model
        session.installed_models_cache = ["qwen2.5-coder:7b"]
        session.model_overrides["chat"] = "qwen2.5:7b"

        session.auto_resolve_models()

        # Should be auto-routed to qwen2.5-coder:7b and persisted
        assert session.model_overrides["chat"] == "qwen2.5-coder:7b"
        cfg = ConfigManager.load()
        assert cfg.models.chat == "qwen2.5-coder:7b"


def _find_binding(kb, key_name: str):
    for b in kb.bindings:
        if len(b.keys) == 1 and getattr(b.keys[0], "value", str(b.keys[0])) == key_name:
            return b
    return None


def test_editor_keybindings_shift_selection():
    """Verify Shift+Left / Shift+Right creates and expands text selection range."""
    session_mock = MagicMock()
    kb = build_editor_keybindings(session=session_mock)

    binding_s_right = _find_binding(kb, "s-right")
    assert binding_s_right is not None

    buff = Buffer()
    buff.set_document(Document("hello world", 0))

    mock_event = MagicMock()
    mock_event.current_buffer = buff

    # Trigger Shift+Right
    binding_s_right.call(mock_event)
    assert buff.selection_state is not None
    assert buff.cursor_position == 1

    # Shift+Right again
    binding_s_right.call(mock_event)
    assert buff.cursor_position == 2
    from_pos, to_pos = buff.document.selection_range()
    assert buff.text[from_pos:to_pos] == "he"


def test_editor_keybindings_select_all_and_copy():
    """Verify Ctrl+A selects all, and Ctrl+C copies selection to clipboard."""
    session_mock = MagicMock()
    kb = build_editor_keybindings(session=session_mock)

    binding_ca = _find_binding(kb, "c-a")
    binding_cc = _find_binding(kb, "c-c")
    assert binding_ca is not None
    assert binding_cc is not None

    buff = Buffer()
    buff.set_document(Document("select all this text", 5))

    mock_event = MagicMock()
    mock_event.current_buffer = buff

    # Ctrl+A
    binding_ca.call(mock_event)
    assert buff.selection_state is not None
    from_pos, to_pos = buff.document.selection_range()
    assert buff.text[from_pos:to_pos] == "select all this text"

    # Ctrl+C with active selection
    binding_cc.call(mock_event)
    assert get_clipboard_text() == "select all this text"
    # Selection should now be exited
    assert buff.selection_state is None


def test_editor_keybindings_double_ctrl_c_exit():
    """Verify Ctrl+C clears line when text is present, and double Ctrl+C exits when buffer is empty."""
    session_mock = MagicMock()
    session_mock.should_exit = False
    session_mock.status_notice = ""
    kb = build_editor_keybindings(session=session_mock)

    binding_cc = _find_binding(kb, "c-c")
    assert binding_cc is not None

    buff = Buffer()
    buff.set_document(Document("some incomplete line", 5))

    mock_event = MagicMock()
    mock_event.current_buffer = buff

    # 1. First Ctrl+C with text: clears buffer without exiting
    binding_cc.call(mock_event)
    assert buff.text == ""
    assert session_mock.should_exit is False
    mock_event.app.exit.assert_not_called()

    # 2. First Ctrl+C with empty buffer: sets notice
    binding_cc.call(mock_event)
    assert session_mock.status_notice == "Press Ctrl+C again to exit"
    assert session_mock.should_exit is False
    mock_event.app.exit.assert_not_called()

    # 3. Second Ctrl+C immediately after: triggers clean exit
    binding_cc.call(mock_event)
    assert session_mock.should_exit is True
    mock_event.app.exit.assert_called_once_with(result="/exit")


def test_editor_keybindings_paste():
    """Verify Ctrl+V pastes clipboard content and replaces active selection."""
    session_mock = MagicMock()
    kb = build_editor_keybindings(session=session_mock)

    binding_cv = _find_binding(kb, "c-v")
    assert binding_cv is not None

    set_clipboard_text("PASTED_TEXT")

    buff = Buffer()
    buff.set_document(Document("initial line", 8))

    mock_event = MagicMock()
    mock_event.current_buffer = buff

    # 1. Simple paste at cursor
    binding_cv.call(mock_event)
    assert "PASTED_TEXT" in buff.text

    # 2. Paste replacing selection
    buff.set_document(Document("foo REPLACE_ME bar", 4))
    buff.start_selection(SelectionType.CHARACTERS)
    buff.cursor_position = 14  # after REPLACE_ME
    set_clipboard_text("NEW")

    binding_cv.call(mock_event)
    assert "REPLACE_ME" not in buff.text
    assert "foo NEW bar" in buff.text


def test_editor_keybindings_cut():
    """Verify Ctrl+X cuts selected text or current line to clipboard."""
    session_mock = MagicMock()
    kb = build_editor_keybindings(session=session_mock)

    binding_cx = _find_binding(kb, "c-x")
    assert binding_cx is not None

    buff = Buffer()
    buff.set_document(Document("cut this text out", 0))
    buff.cursor_position = 4
    buff.start_selection(SelectionType.CHARACTERS)
    buff.cursor_position = 8  # selected "this"

    mock_event = MagicMock()
    mock_event.current_buffer = buff

    binding_cx.call(mock_event)
    assert get_clipboard_text() == "this"
    assert "this" not in buff.text


def test_editor_keybindings_undo_redo():
    """Verify Ctrl+Z and Ctrl+Y perform undo and redo operations."""
    session_mock = MagicMock()
    kb = build_editor_keybindings(session=session_mock)

    binding_cz = _find_binding(kb, "c-z")
    binding_cy = _find_binding(kb, "c-y")
    assert binding_cz is not None
    assert binding_cy is not None

    buff = Buffer()
    buff.save_to_undo_stack()
    buff.insert_text("first step")
    assert buff.text == "first step"

    mock_event = MagicMock()
    mock_event.current_buffer = buff

    # Undo
    binding_cz.call(mock_event)
    assert buff.text == ""

    # Redo
    binding_cy.call(mock_event)
    assert buff.text == "first step"


def test_editor_keybindings_shift_enter_and_submit():
    """Verify Shift+Enter (c-j) inserts newline and Enter (c-m) submits input or applies completion."""
    session_mock = MagicMock()
    kb = build_editor_keybindings(session=session_mock)

    binding_cj = _find_binding(kb, "c-j")
    cm_bindings = [
        b
        for b in kb.bindings
        if len(b.keys) == 1 and getattr(b.keys[0], "value", str(b.keys[0])) == "c-m"
    ]
    assert binding_cj is not None
    assert len(cm_bindings) == 2

    # Identify submit and completion handlers
    submit_binding = next(
        b for b in cm_bindings if getattr(b.handler, "__name__", "") == "_submit_input"
    )
    select_binding = next(
        b for b in cm_bindings if getattr(b.handler, "__name__", "") == "_select_completion"
    )

    buff = Buffer()
    buff.insert_text("line 1")

    mock_event = MagicMock()
    mock_event.current_buffer = buff

    # Shift+Enter inserts newline
    binding_cj.call(mock_event)
    assert buff.text == "line 1\n"

    # Enter (when not selecting completion) calls validate_and_handle
    mock_buff = MagicMock()
    mock_event.current_buffer = mock_buff
    submit_binding.call(mock_event)
    mock_buff.validate_and_handle.assert_called_once()

    # Enter (when completion selected) applies completion
    comp_buff = MagicMock()
    mock_event.current_buffer = comp_buff
    comp_buff.complete_state.current_completion = MagicMock()
    select_binding.call(mock_event)
    comp_buff.apply_completion.assert_called_once_with(comp_buff.complete_state.current_completion)


def test_windows_console_event_modifier_decoding():
    """Verify Windows ConsoleInputReader correctly decodes Ctrl+letter shortcuts and Shift+Enter."""
    import sys

    if sys.platform != "win32":
        pytest.skip("Windows only test")

    from prompt_toolkit.input.win32 import KEY_EVENT_RECORD, ConsoleInputReader
    from prompt_toolkit.keys import Keys

    reader = ConsoleInputReader()

    def make_ev(vk: int, ctrl: bool = False, shift: bool = False, u_char: str = ""):
        ev = KEY_EVENT_RECORD()
        ev.KeyDown = True
        ev.VirtualKeyCode = vk
        ev.uChar.UnicodeChar = u_char
        state = 0
        if ctrl:
            state |= reader.LEFT_CTRL_PRESSED
        if shift:
            state |= reader.SHIFT_PRESSED
        ev.ControlKeyState = state
        return ev

    # 1. Shift+Enter -> Keys.ControlJ (\n)
    ev_shift_enter = make_ev(13, shift=True, u_char="\r")
    kps = reader._event_to_key_presses(ev_shift_enter)
    assert len(kps) == 1
    assert kps[0].key == Keys.ControlJ
    assert kps[0].data == "\n"

    # 2. Ctrl+Enter -> Keys.ControlJ (\n)
    ev_ctrl_enter = make_ev(13, ctrl=True, u_char="\n")
    kps = reader._event_to_key_presses(ev_ctrl_enter)
    assert len(kps) == 1
    assert kps[0].key == Keys.ControlJ

    # 3. Plain Enter -> Keys.ControlM (\r)
    ev_enter = make_ev(13, u_char="\r")
    kps = reader._event_to_key_presses(ev_enter)
    assert len(kps) == 1
    assert kps[0].key == Keys.ControlM

    # 4. Ctrl+C (even when IME delivers u_char='c') -> Keys.ControlC (\x03)
    ev_ctrl_c = make_ev(67, ctrl=True, u_char="c")
    kps = reader._event_to_key_presses(ev_ctrl_c)
    assert len(kps) == 1
    assert kps[0].key == Keys.ControlC
    assert kps[0].data == "\x03"

    # 5. Ctrl+V (even when IME delivers u_char='v') -> Keys.ControlV (\x16)
    ev_ctrl_v = make_ev(86, ctrl=True, u_char="v")
    kps = reader._event_to_key_presses(ev_ctrl_v)
    assert len(kps) == 1
    assert kps[0].key == Keys.ControlV
    assert kps[0].data == "\x16"

    # 6. Ctrl+A -> Keys.ControlA
    ev_ctrl_a = make_ev(65, ctrl=True, u_char="a")
    kps = reader._event_to_key_presses(ev_ctrl_a)
    assert len(kps) == 1
    assert kps[0].key == Keys.ControlA

    # 7. Ctrl+Z -> Keys.ControlZ
    ev_ctrl_z = make_ev(90, ctrl=True, u_char="z")
    kps = reader._event_to_key_presses(ev_ctrl_z)
    assert len(kps) == 1
    assert kps[0].key == Keys.ControlZ

    # 8. Plain 'c' without Ctrl -> plain character 'c'
    ev_plain_c = make_ev(67, u_char="c")
    kps = reader._event_to_key_presses(ev_plain_c)
    assert len(kps) == 1
    assert kps[0].key == "c"
    assert kps[0].data == "c"
