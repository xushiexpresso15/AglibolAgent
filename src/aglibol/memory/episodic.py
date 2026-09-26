"""Append-only disk episodic memory recorded as JSONL for audit and replay."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from aglibol.core.types import ChatMessage


class EpisodicMemory:
    """Records complete conversation history and tool outputs to a JSONL log file on SSD."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_turn(
        self, agent: str, step: int, message: ChatMessage, extra: dict[str, Any] | None = None
    ) -> None:
        """Append a message turn to the JSONL log file."""
        record = {
            "timestamp": time.time(),
            "agent": agent,
            "step": step,
            "role": message.role,
            "content": message.content,
            "tool_calls": [tc.model_dump() for tc in (message.tool_calls or [])],
            "extra": extra or {},
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def read_all_turns(self) -> list[dict[str, Any]]:
        """Read all recorded turns from disk."""
        if not self.log_path.exists():
            return []
        records: list[dict[str, Any]] = []
        with open(self.log_path, encoding="utf-8") as f:
            for line in f:
                cleaned = line.strip()
                if cleaned:
                    try:
                        records.append(json.loads(cleaned))
                    except json.JSONDecodeError:
                        continue
        return records

    def get_recent_turns(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get the most recent N turns from the log."""
        all_turns = self.read_all_turns()
        return all_turns[-limit:]

    def get_task_history(self, task_id: str) -> list[dict[str, Any]]:
        """Retrieve all recorded turns matching a specific task ID."""
        all_turns = self.read_all_turns()
        return [t for t in all_turns if t.get("extra", {}).get("task_id") == task_id]

    def get_summary_for_context(self, limit: int = 5) -> str:
        """Build a compact, clean text summary of recent turns for prompt injection."""
        recent = self.get_recent_turns(limit=limit)
        if not recent:
            return ""
        lines = []
        for r in recent:
            agent = r.get("agent", "agent")
            role = r.get("role", "")
            content = str(r.get("content", "")).strip()
            if len(content) > 160:
                content = content[:157] + "..."
            lines.append(f"[{agent}/{role}]: {content}")
        return "\n".join(lines)
