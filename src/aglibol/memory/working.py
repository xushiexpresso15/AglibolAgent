"""In-memory working context with disk-sync capabilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import msgpack

from aglibol.core.context import ContextManager
from aglibol.core.types import ChatMessage


class WorkingMemory:
    """Manages ephemeral working memory for an active agent turn with fast disk checkpointing."""

    def __init__(self, max_context_tokens: int = 4096) -> None:
        self.max_context_tokens = max_context_tokens
        self.messages: list[ChatMessage] = []
        self.scratchpad: dict[str, Any] = {}

    def add_message(self, role: str, content: str) -> None:
        """Append a chat message and automatically compress context if threshold exceeded."""
        self.messages.append(ChatMessage(role=role, content=content))
        self.compress_if_needed()

    def set_system_prompt(self, prompt: str) -> None:
        """Ensure system prompt is placed at the beginning of the context."""
        # Replace or insert system message at index 0
        if self.messages and self.messages[0].role == "system":
            self.messages[0] = ChatMessage(role="system", content=prompt)
        else:
            self.messages.insert(0, ChatMessage(role="system", content=prompt))

    def get_messages(self) -> list[ChatMessage]:
        """Return current conversation messages."""
        return self.messages

    def compress_if_needed(self) -> None:
        """Run context compression when message tokens exceed headroom."""
        self.messages = ContextManager.compress_messages(
            self.messages,
            max_context_tokens=self.max_context_tokens,
            headroom_ratio=0.8,
        )

    def dump_to_file(self, filepath: Path) -> None:
        """Serialize working memory to disk via msgpack for fast zero-overhead model swapping."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "messages": [m.model_dump() for m in self.messages],
            "scratchpad": self.scratchpad,
            "max_context_tokens": self.max_context_tokens,
        }
        with open(filepath, "wb") as f:
            f.write(msgpack.packb(data, default=str))

    def load_from_file(self, filepath: Path) -> None:
        """Rehydrate working memory from disk snapshot."""
        if not filepath.exists():
            return
        with open(filepath, "rb") as f:
            raw = msgpack.unpackb(f.read(), raw=False)
            self.messages = [ChatMessage(**m) for m in raw.get("messages", [])]
            self.scratchpad = raw.get("scratchpad", {})
            self.max_context_tokens = raw.get("max_context_tokens", 4096)

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from working scratchpad."""
        return self.scratchpad.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Store a key-value pair into working scratchpad."""
        self.scratchpad[key] = value

    def clear(self) -> None:
        """Reset messages and scratchpad."""
        self.messages = []
        self.scratchpad = {}
