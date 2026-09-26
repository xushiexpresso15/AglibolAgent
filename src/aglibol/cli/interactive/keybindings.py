"""Editor-like keybindings for the interactive prompt session."""

from __future__ import annotations

import sys
import time
from typing import TYPE_CHECKING

import prompt_toolkit.key_binding.bindings.basic as _basic_mod
import prompt_toolkit.key_binding.defaults as _def_mod
from prompt_toolkit.filters import completion_is_selected, has_selection
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.selection import SelectionType

from aglibol.cli.interactive.clipboard import get_clipboard_text, set_clipboard_text

if TYPE_CHECKING:
    from aglibol.cli.interactive.repl import InteractiveSession

# Patch basic keybindings so ControlJ (\n) is NOT aliased to ControlM (\r) via _newline2
try:
    _orig_load_basic = _basic_mod.load_basic_bindings

    def _patched_load_basic():
        kb = _orig_load_basic()
        kb._bindings = [b for b in kb.bindings if getattr(b.handler, "__name__", "") != "_newline2"]
        kb._clear_cache()
        return kb

    _basic_mod.load_basic_bindings = _patched_load_basic
    _def_mod.load_basic_bindings = _patched_load_basic
except Exception:
    pass

# 2. Register standard Shift+Enter escape sequences in VT100 parser
try:
    from prompt_toolkit.input.vt100_parser import ANSI_SEQUENCES
    from prompt_toolkit.keys import Keys as _PTKeys

    ANSI_SEQUENCES["\x1b[13;2u"] = _PTKeys.ControlJ
    ANSI_SEQUENCES["\x1b[27;2;13~"] = _PTKeys.ControlJ
except Exception:
    pass

# 3. Patch Windows console input readers for native conhost and Windows Terminal
if sys.platform == "win32":
    try:
        import prompt_toolkit.input.win32 as _w32_mod
        from prompt_toolkit.input.win32 import ConsoleInputReader
        from prompt_toolkit.key_binding.key_processor import KeyPress
        from prompt_toolkit.keys import Keys

        # Ensure Win32Input uses robust ConsoleInputReader with full KEY_EVENT_RECORD decoding
        _w32_mod._is_win_vt100_input_enabled = lambda: False

        _orig_event_to_key_presses = ConsoleInputReader._event_to_key_presses

        # Explicit VK mappings for Ctrl+Letter shortcuts (works with all IMEs and console settings)
        _CTRL_VK_MAP = {
            65: (Keys.ControlA, "\x01"),  # VK_A
            67: (Keys.ControlC, "\x03"),  # VK_C
            86: (Keys.ControlV, "\x16"),  # VK_V
            88: (Keys.ControlX, "\x18"),  # VK_X
            89: (Keys.ControlY, "\x19"),  # VK_Y
            90: (Keys.ControlZ, "\x1a"),  # VK_Z
        }

        def _patched_event_to_key_presses(self, ev):
            if hasattr(ev, "KeyDown") and ev.KeyDown:
                ctrl_pressed = bool(
                    ev.ControlKeyState & (self.LEFT_CTRL_PRESSED | self.RIGHT_CTRL_PRESSED)
                )
                shift_pressed = bool(ev.ControlKeyState & self.SHIFT_PRESSED)
                alt_pressed = bool(
                    ev.ControlKeyState & (self.LEFT_ALT_PRESSED | self.RIGHT_ALT_PRESSED)
                )
                vk = ev.VirtualKeyCode

                # Shift+Enter or Ctrl+Enter -> emit ControlJ (newline)
                if vk == 13:  # VK_RETURN
                    if shift_pressed or ctrl_pressed:
                        return [KeyPress(Keys.ControlJ, "\n")]
                    elif alt_pressed:
                        return [KeyPress(Keys.Escape, ""), KeyPress(Keys.ControlM, "\r")]
                    return [KeyPress(Keys.ControlM, "\r")]

                # When Ctrl is held without Alt: guarantee Ctrl+A/C/V/X/Y/Z mapping
                if ctrl_pressed and not alt_pressed and vk in _CTRL_VK_MAP:
                    k, d = _CTRL_VK_MAP[vk]
                    return [KeyPress(k, d)]

            return _orig_event_to_key_presses(self, ev)

        ConsoleInputReader._event_to_key_presses = _patched_event_to_key_presses
    except Exception:
        pass


def build_editor_keybindings(session: InteractiveSession | None = None) -> KeyBindings:
    """
    Construct editor-grade keybindings for prompt_toolkit:
    - Shift + Left/Right/Up/Down: Range text selection.
    - Left/Right/Up/Down (without Shift): Deselect and navigate.
    - Ctrl + A: Select all text in prompt buffer.
    - Ctrl + C:
        * If text selected: copy to OS clipboard and deselect.
        * If no selection & buffer has text: clear current buffer line.
        * If no selection & buffer empty: double press within 2.0s exits cleanly.
    - Ctrl + V: Paste from OS clipboard (replacing active selection if any).
    - Ctrl + X: Cut selected text (or current line if no selection) to OS clipboard.
    - Ctrl + Z: Undo buffer mutation.
    - Ctrl + Y: Redo buffer mutation.
    - Shift + Enter: Insert newline without submitting.
    - Enter (without completion active): Submit prompt.
    - Backspace / Delete: Delete selection range when text is selected.
    """
    kb = KeyBindings()
    last_ctrl_c_time = 0.0

    # 1. Shift + Left / Right / Up / Down: Start / expand text selection
    @kb.add("s-left")
    def _select_left(event) -> None:
        buff = event.current_buffer
        if buff.selection_state is None:
            buff.start_selection(selection_type=SelectionType.CHARACTERS)
        buff.cursor_left(count=1)

    @kb.add("s-right")
    def _select_right(event) -> None:
        buff = event.current_buffer
        if buff.selection_state is None:
            buff.start_selection(selection_type=SelectionType.CHARACTERS)
        buff.cursor_right(count=1)

    @kb.add("s-up")
    def _select_up(event) -> None:
        buff = event.current_buffer
        if buff.selection_state is None:
            buff.start_selection(selection_type=SelectionType.CHARACTERS)
        buff.cursor_up(count=1)

    @kb.add("s-down")
    def _select_down(event) -> None:
        buff = event.current_buffer
        if buff.selection_state is None:
            buff.start_selection(selection_type=SelectionType.CHARACTERS)
        buff.cursor_down(count=1)

    # 2. Deselect on regular arrow navigation when selection is active
    @kb.add("left", filter=has_selection)
    def _deselect_left(event) -> None:
        buff = event.current_buffer
        buff.exit_selection()
        buff.cursor_left(count=1)

    @kb.add("right", filter=has_selection)
    def _deselect_right(event) -> None:
        buff = event.current_buffer
        buff.exit_selection()
        buff.cursor_right(count=1)

    @kb.add("up", filter=has_selection)
    def _deselect_up(event) -> None:
        buff = event.current_buffer
        buff.exit_selection()
        buff.cursor_up(count=1)

    @kb.add("down", filter=has_selection)
    def _deselect_down(event) -> None:
        buff = event.current_buffer
        buff.exit_selection()
        buff.cursor_down(count=1)

    # 3. Ctrl + A: Select all text
    @kb.add("c-a", eager=True)
    def _select_all(event) -> None:
        buff = event.current_buffer
        buff.cursor_position = 0
        buff.start_selection(selection_type=SelectionType.CHARACTERS)
        buff.cursor_position = len(buff.text)

    # 4. Ctrl + C: Copy if text selected; clear line or double-press to exit
    @kb.add("c-c", eager=True)
    def _copy_or_cancel_or_exit(event) -> None:
        nonlocal last_ctrl_c_time
        buff = event.current_buffer

        # If text is selected: copy to OS clipboard and deselect
        if buff.selection_state is not None:
            from_pos, to_pos = buff.document.selection_range()
            selected_text = buff.text[from_pos:to_pos]
            if selected_text:
                set_clipboard_text(selected_text)
            buff.exit_selection()
            return

        # If buffer contains text: clear buffer like standard CLI
        if buff.text:
            buff.save_to_undo_stack()
            buff.reset()
            if session:
                session.status_notice = ""
                event.app.invalidate()
            return

        # Buffer is empty: check for double Ctrl+C exit
        now = time.time()
        if now - last_ctrl_c_time <= 2.0:
            if session:
                session.should_exit = True
            event.app.exit(result="/exit")
        else:
            last_ctrl_c_time = now
            if session:
                session.status_notice = "Press Ctrl+C again to exit"
                event.app.invalidate()

    # 5. Ctrl + V: Paste from system clipboard
    @kb.add("c-v", eager=True)
    def _paste(event) -> None:
        buff = event.current_buffer
        clip = get_clipboard_text()
        if clip:
            buff.save_to_undo_stack()
            if buff.selection_state is not None:
                buff.cut_selection()
            buff.insert_text(clip)

    # 6. Ctrl + X: Cut selection to clipboard (or current line if no selection)
    @kb.add("c-x", eager=True)
    def _cut(event) -> None:
        buff = event.current_buffer
        if buff.selection_state is not None:
            from_pos, to_pos = buff.document.selection_range()
            selected_text = buff.text[from_pos:to_pos]
            if selected_text:
                set_clipboard_text(selected_text)
            buff.save_to_undo_stack()
            buff.cut_selection()
        elif buff.text:
            # Cut entire current line
            cur_line = buff.document.current_line
            if cur_line:
                set_clipboard_text(cur_line)
                buff.save_to_undo_stack()
                buff.transform_current_line(lambda _: "")

    # 7. Ctrl + Z: Undo
    @kb.add("c-z", eager=True)
    def _undo(event) -> None:
        event.current_buffer.undo()

    # 8. Ctrl + Y: Redo
    @kb.add("c-y", eager=True)
    def _redo(event) -> None:
        event.current_buffer.redo()

    # 9. Shift + Enter / Alt + Enter: Insert newline without submitting
    @kb.add("c-j", eager=True)
    def _insert_newline(event) -> None:
        buff = event.current_buffer
        buff.save_to_undo_stack()
        buff.insert_text("\n")

    @kb.add("escape", "enter", eager=True)
    def _alt_enter_newline(event) -> None:
        buff = event.current_buffer
        buff.save_to_undo_stack()
        buff.insert_text("\n")

    # 10. Enter handling:
    # If a completion item is actively selected in dropdown (via tab/down arrow), apply it
    @kb.add("c-m", filter=completion_is_selected, eager=True)
    def _select_completion(event) -> None:
        buff = event.current_buffer
        if buff.complete_state and buff.complete_state.current_completion:
            buff.apply_completion(buff.complete_state.current_completion)

    # If no completion item is actively selected, submit immediately on first Enter
    @kb.add("c-m", filter=~completion_is_selected, eager=True)
    def _submit_input(event) -> None:
        if session:
            session.status_notice = ""
        event.current_buffer.validate_and_handle()

    # 11. Backspace / Delete with active selection deletes selection range
    @kb.add("backspace", filter=has_selection, eager=True)
    def _delete_selection_bs(event) -> None:
        event.current_buffer.save_to_undo_stack()
        event.current_buffer.cut_selection()

    @kb.add("delete", filter=has_selection, eager=True)
    def _delete_selection_del(event) -> None:
        event.current_buffer.save_to_undo_stack()
        event.current_buffer.cut_selection()

    return kb
