"""Context manager for token estimation, sliding window, and context compression."""

from __future__ import annotations

from typing import Any

from aglibol.core.types import ChatMessage


class ContextManager:
    """Manages chat context tokens and applies sliding window compression when nearing limits."""

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Heuristic token estimator (~4 characters per token for English/code, ~1.5 for CJK)."""
        if not text:
            return 0
        # Check for non-ASCII characters (e.g. CJK)
        cjk_chars = sum(1 for c in text if ord(c) > 0x2000)
        ascii_chars = len(text) - cjk_chars
        return int((ascii_chars / 3.8) + (cjk_chars / 1.5)) + 1

    @classmethod
    def estimate_messages_tokens(cls, messages: list[ChatMessage]) -> int:
        """Estimate total tokens consumed by a sequence of chat messages."""
        total = 0
        for msg in messages:
            total += 4  # Formatting overhead per message
            total += cls.estimate_tokens(msg.content)
            for tc in msg.tool_calls:
                total += cls.estimate_tokens(str(tc.arguments))
        return total

    @classmethod
    def compress_messages(
        cls,
        messages: list[ChatMessage],
        max_context_tokens: int,
        headroom_ratio: float = 0.75,
        episodic_memory: Any | None = None,
    ) -> list[ChatMessage]:
        """
        Compress context if total tokens exceed (max_context_tokens * headroom_ratio).
        Algorithm:
        1. Always preserve messages[0] (system) and messages[1] (user goal).
        2. Always preserve the most recent 2 messages.
        3. For intermediate messages, apply Observation Masking to large tool outputs.
        4. If still over budget, offload oldest intermediate messages to episodic storage on disk.
        5. Overflow protection: truncate messages[1] if system + user > target.
        """
        target_tokens = int(max_context_tokens * headroom_ratio)
        current_tokens = cls.estimate_messages_tokens(messages)

        if current_tokens <= target_tokens:
            return messages

        if not messages:
            return messages

        # Make a copy since we will mutate
        compressed = [msg.model_copy() for msg in messages]

        # 1. Preservation & Overflow protection
        preserved_head = []
        if len(compressed) > 0:
            preserved_head.append(compressed[0])
        if len(compressed) > 1:
            preserved_head.append(compressed[1])

        # Check overflow on just the head
        head_tokens = cls.estimate_messages_tokens(preserved_head)
        if head_tokens > target_tokens and len(preserved_head) > 1:
            sys_tokens = cls.estimate_messages_tokens([preserved_head[0]])
            allowed_user_tokens = max(10, target_tokens - sys_tokens - 10)
            allowed_chars = int(allowed_user_tokens * 3.8)
            if len(preserved_head[1].content) > allowed_chars:
                preserved_head[1].content = (
                    preserved_head[1].content[:allowed_chars] + "... [Truncated]"
                )

        if len(compressed) <= 4:
            return preserved_head + compressed[2:]

        # Separate into head, middle, tail
        tail = compressed[-2:]
        middle = compressed[2:-2]

        # 2. Observation Masking on middle messages
        for msg in middle:
            if msg.role == "tool" and len(msg.content) > 200:
                lines = msg.content.splitlines()
                if len(lines) > 15:
                    head_lines = "\n".join(lines[:10])
                    tail_lines = "\n".join(lines[-5:])
                    hidden_count = len(lines) - 15
                    msg.content = (
                        f"{head_lines}\n"
                        f"... [Output of {msg.tool_name or 'tool'} masked ({hidden_count} lines, {len(msg.content)} chars). Context preserved in disk memory. Proceed with implementation.] ...\n"
                        f"{tail_lines}"
                    )
                else:
                    old_len = len(msg.content)
                    msg.content = f"[Output of {msg.tool_name or 'tool'} masked ({old_len} chars). Context preserved in disk memory. Proceed with implementation.]"

        # Re-evaluate tokens
        current_tokens = cls.estimate_messages_tokens(preserved_head + middle + tail)

        # 4. Progressively offload oldest intermediate messages to disk
        offloaded_count = 0
        while current_tokens > target_tokens and len(middle) > 0:
            dropped = middle.pop(0)
            offloaded_count += 1
            if episodic_memory and hasattr(episodic_memory, "log_turn"):
                try:
                    episodic_memory.log_turn(
                        agent="context_offload",
                        step=offloaded_count,
                        message=dropped,
                        extra={"offloaded_to_disk": True},
                    )
                except Exception:
                    pass
            current_tokens = cls.estimate_messages_tokens(preserved_head + middle + tail)

        if offloaded_count > 0:
            memory_notice = ChatMessage(
                role="system",
                content=(
                    f"[Archived Memory: {offloaded_count} earlier conversation step(s) offloaded to disk storage "
                    f"to conserve local VRAM/context. Key context preserved in disk episodic memory.]"
                ),
            )
            return preserved_head + [memory_notice] + middle + tail

        return preserved_head + middle + tail
