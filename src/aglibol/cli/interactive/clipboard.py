"""Cross-platform system clipboard interface for Aglibol interactive CLI."""

from __future__ import annotations

import shutil
import subprocess
import sys
import time

_MEMORY_CLIPBOARD: str = ""


def _win32_open_clipboard_with_retry(user32, max_retries: int = 5, delay: float = 0.01) -> bool:
    """Attempt to open Windows clipboard with brief retry if locked by another application."""
    for _ in range(max_retries):
        if user32.OpenClipboard(None):
            return True
        time.sleep(delay)
    return False


def get_clipboard_text() -> str:
    """Retrieve plain text from the OS clipboard."""
    global _MEMORY_CLIPBOARD

    # 1. Windows native via ctypes (configured with 64-bit pointer argtypes/restypes)
    if sys.platform == "win32":
        try:
            import ctypes

            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32

            user32.OpenClipboard.restype = ctypes.c_bool
            user32.OpenClipboard.argtypes = [ctypes.c_void_p]
            user32.CloseClipboard.restype = ctypes.c_bool
            user32.CloseClipboard.argtypes = []
            user32.GetClipboardData.restype = ctypes.c_void_p
            user32.GetClipboardData.argtypes = [ctypes.c_uint]

            kernel32.GlobalLock.restype = ctypes.c_void_p
            kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
            kernel32.GlobalUnlock.restype = ctypes.c_bool
            kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]

            cf_unicodetext = 13  # CF_UNICODETEXT
            if not _win32_open_clipboard_with_retry(user32):
                return _MEMORY_CLIPBOARD

            try:
                handle = user32.GetClipboardData(cf_unicodetext)
                if not handle:
                    return _MEMORY_CLIPBOARD
                ptr = kernel32.GlobalLock(handle)
                if not ptr:
                    return _MEMORY_CLIPBOARD
                try:
                    val = ctypes.c_wchar_p(ptr).value
                    if val is not None:
                        _MEMORY_CLIPBOARD = val
                        return val
                    return _MEMORY_CLIPBOARD
                finally:
                    kernel32.GlobalUnlock(handle)
            finally:
                user32.CloseClipboard()
        except Exception:
            return _MEMORY_CLIPBOARD

    # 2. macOS via pbpaste
    if sys.platform == "darwin" and shutil.which("pbpaste"):
        try:
            res = subprocess.run(["pbpaste"], capture_output=True, text=True, timeout=2.0)
            if res.returncode == 0:
                _MEMORY_CLIPBOARD = res.stdout
                return res.stdout
        except Exception:
            pass

    # 3. Linux via xclip / wl-paste
    if shutil.which("wl-paste"):
        try:
            res = subprocess.run(
                ["wl-paste", "--no-newline"], capture_output=True, text=True, timeout=2.0
            )
            if res.returncode == 0:
                _MEMORY_CLIPBOARD = res.stdout
                return res.stdout
        except Exception:
            pass
    elif shutil.which("xclip"):
        try:
            res = subprocess.run(
                ["xclip", "-selection", "clipboard", "-o"],
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            if res.returncode == 0:
                _MEMORY_CLIPBOARD = res.stdout
                return res.stdout
        except Exception:
            pass

    return _MEMORY_CLIPBOARD


def set_clipboard_text(text: str) -> bool:
    """Copy plain text into the OS clipboard."""
    global _MEMORY_CLIPBOARD
    _MEMORY_CLIPBOARD = text

    # 1. Windows native via ctypes (configured with 64-bit pointer argtypes/restypes)
    if sys.platform == "win32":
        try:
            import ctypes

            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32

            user32.OpenClipboard.restype = ctypes.c_bool
            user32.OpenClipboard.argtypes = [ctypes.c_void_p]
            user32.CloseClipboard.restype = ctypes.c_bool
            user32.CloseClipboard.argtypes = []
            user32.EmptyClipboard.restype = ctypes.c_bool
            user32.EmptyClipboard.argtypes = []
            user32.SetClipboardData.restype = ctypes.c_void_p
            user32.SetClipboardData.argtypes = [ctypes.c_uint, ctypes.c_void_p]

            kernel32.GlobalAlloc.restype = ctypes.c_void_p
            kernel32.GlobalAlloc.argtypes = [ctypes.c_uint, ctypes.c_size_t]
            kernel32.GlobalLock.restype = ctypes.c_void_p
            kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
            kernel32.GlobalUnlock.restype = ctypes.c_bool
            kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
            kernel32.GlobalFree.restype = ctypes.c_void_p
            kernel32.GlobalFree.argtypes = [ctypes.c_void_p]

            cf_unicodetext = 13  # CF_UNICODETEXT
            gmem_moveable = 0x0002

            if not _win32_open_clipboard_with_retry(user32):
                return False

            try:
                user32.EmptyClipboard()
                encoded = (text + "\0").encode("utf-16le")
                handle = kernel32.GlobalAlloc(gmem_moveable, len(encoded))
                if not handle:
                    return False
                ptr = kernel32.GlobalLock(handle)
                if not ptr:
                    kernel32.GlobalFree(handle)
                    return False
                try:
                    ctypes.memmove(ptr, encoded, len(encoded))
                finally:
                    kernel32.GlobalUnlock(handle)
                res = user32.SetClipboardData(cf_unicodetext, handle)
                if not res:
                    kernel32.GlobalFree(handle)
                    return False
                return True
            finally:
                user32.CloseClipboard()
        except Exception:
            return False

    # 2. macOS via pbcopy
    if sys.platform == "darwin" and shutil.which("pbcopy"):
        try:
            proc = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
            proc.communicate(text.encode("utf-8"), timeout=2.0)
            return proc.returncode == 0
        except Exception:
            pass

    # 3. Linux via wl-copy / xclip
    if shutil.which("wl-copy"):
        try:
            proc = subprocess.Popen(["wl-copy"], stdin=subprocess.PIPE)
            proc.communicate(text.encode("utf-8"), timeout=2.0)
            return proc.returncode == 0
        except Exception:
            pass
    return True


from prompt_toolkit.clipboard import Clipboard, ClipboardData


class OSClipboard(Clipboard):
    """Bridge prompt_toolkit clipboard operations directly to the OS system clipboard."""

    def set_data(self, data: ClipboardData) -> None:
        set_clipboard_text(data.text)

    def get_data(self) -> ClipboardData:
        return ClipboardData(get_clipboard_text())
