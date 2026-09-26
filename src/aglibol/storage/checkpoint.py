"""SQLite WAL-backed checkpoint manager for full session resumption and rollback."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from aglibol.core.types import AgentState


class CheckpointStore:
    """Persists workflow step snapshots to a local SQLite database in WAL mode."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database tables with WAL mode for fast zero-lock concurrent writes."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    active_agent TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_step ON checkpoints(session_id, step);"
            )
            conn.commit()
        finally:
            conn.close()

    def save_checkpoint(self, state: AgentState) -> int:
        """Save a snapshot of the current AgentState."""
        state_json = state.model_dump_json()
        now = time.time()
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO checkpoints (session_id, step, active_agent, state_json, created_at)
                VALUES (?, ?, ?, ?, ?);
                """,
                (state.session_id, state.current_step, state.active_agent, state_json, now),
            )
            conn.commit()
            return cursor.lastrowid or 0
        finally:
            conn.close()

    def get_latest_checkpoint(self, session_id: str) -> AgentState | None:
        """Retrieve the most recent checkpoint for a given session."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT state_json FROM checkpoints
                WHERE session_id = ?
                ORDER BY step DESC, id DESC
                LIMIT 1;
                """,
                (session_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            data = json.loads(row[0])
            return AgentState(**data)
        finally:
            conn.close()

    def get_checkpoint_by_id(self, checkpoint_id: int) -> AgentState | None:
        """Retrieve a specific checkpoint by ID (useful for time-travel debugging)."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT state_json FROM checkpoints WHERE id = ?;",
                (checkpoint_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            data = json.loads(row[0])
            return AgentState(**data)
        finally:
            conn.close()

    def list_checkpoints(self, session_id: str, limit: int = 100) -> list[dict]:
        """List all checkpoint summaries for a session."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, step, active_agent, created_at
                FROM checkpoints
                WHERE session_id = ?
                ORDER BY step ASC, id ASC
                LIMIT ?;
                """,
                (session_id, limit),
            )
            rows = cursor.fetchall()
            return [
                {
                    "checkpoint_id": r[0],
                    "step": r[1],
                    "agent": r[2],
                    "created_at": r[3],
                }
                for r in rows
            ]
        finally:
            conn.close()

    def prune_old_checkpoints(self, session_id: str, keep_last: int = 50) -> int:
        """Retain only the latest N checkpoints for a session to bound disk usage."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                DELETE FROM checkpoints
                WHERE session_id = ? AND id NOT IN (
                    SELECT id FROM checkpoints
                    WHERE session_id = ?
                    ORDER BY step DESC, id DESC
                    LIMIT ?
                );
                """,
                (session_id, session_id, keep_last),
            )
            deleted = cursor.rowcount
            conn.commit()
            return deleted
        finally:
            conn.close()
