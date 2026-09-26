"""Tests for context and token management."""

from __future__ import annotations

from aglibol.core.context import ContextManager
from aglibol.core.types import ChatMessage


def test_estimate_tokens():
    text_en = "Hello world, this is a test of token estimation."
    tokens_en = ContextManager.estimate_tokens(text_en)
    assert 5 <= tokens_en <= 25

    text_non_ascii = "Internationalization \u2014 unicode symbols testing \u2192 \u2264"
    tokens_non_ascii = ContextManager.estimate_tokens(text_non_ascii)
    assert 5 <= tokens_non_ascii <= 30


def test_compress_messages():
    system = ChatMessage(role="system", content="System instruction")
    msgs = [system]
    for i in range(20):
        msgs.append(
            ChatMessage(role="user", content=f"User turn {i} with long detailed description " * 10)
        )
        msgs.append(
            ChatMessage(role="assistant", content=f"Assistant response {i} with code " * 10)
        )

    compressed = ContextManager.compress_messages(msgs, max_context_tokens=300, headroom_ratio=0.8)
    assert len(compressed) < len(msgs)
    assert compressed[0].role == "system"
    assert "System instruction" in compressed[0].content
