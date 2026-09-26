"""Streaming token and response parser for Ollama NDJSON streams with early-stop capability."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Callable
from typing import Any


class OllamaStreamParser:
    """Parses streaming lines from Ollama HTTP API responses with early-stopping and tool call buffering."""

    @staticmethod
    async def parse_chat_stream(
        stream: AsyncGenerator[str, None],
        on_token: Callable[[str], Any] | None = None,
        early_stop_fn: Callable[[dict[str, Any]], bool] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Yield structured message chunks from streaming chat response.
        Supports:
        - on_token callback for live terminal UI rendering
        - early_stop_fn callback to stop processing when a condition is met (e.g. tool call completed)
        """
        async for line in stream:
            cleaned = line.strip()
            if not cleaned:
                continue
            try:
                data = json.loads(cleaned)
            except json.JSONDecodeError:
                continue

            msg = data.get("message", {})
            token = msg.get("content", "")
            if token and on_token:
                try:
                    on_token(token)
                except Exception:
                    pass

            yield data

            # Check early stop trigger
            if early_stop_fn and early_stop_fn(data):
                break
