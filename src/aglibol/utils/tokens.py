"""Token estimation utilities."""

from __future__ import annotations


def estimate_tokens(text: str) -> int:
    """Accurate heuristic token count for mixed English, code, and CJK text."""
    if not text:
        return 0
    cjk_count = sum(1 for c in text if ord(c) > 0x2000)
    ascii_count = len(text) - cjk_count
    return int((ascii_count / 3.8) + (cjk_count / 1.5)) + 1
