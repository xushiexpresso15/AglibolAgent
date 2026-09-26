"""Session lifecycle manager for creating, querying, and resuming agent sessions."""

from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path

from pydantic import BaseModel, Field

from aglibol.core.types import HardwareTier


class SessionMeta(BaseModel):
    """Metadata recorded for each run session."""

    session_id: str
    user_goal: str
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    tier: HardwareTier = HardwareTier.TIER_1_4GB
    status: str = "active"  # "active", "completed", "failed", "aborted"
    step_count: int = 0
    total_tokens: int = 0

    @property
    def goal(self) -> str:
        return self.user_goal


class SessionManager:
    """Manages session metadata files on disk."""

    def __init__(self, sessions_dir: Path) -> None:
        self.sessions_dir = sessions_dir
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def create_session(self, goal: str, tier: HardwareTier) -> SessionMeta:
        """Create a new session record and folder."""
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        meta = SessionMeta(session_id=session_id, user_goal=goal, tier=tier)
        self.save_meta(meta)
        return meta

    def save_meta(self, meta: SessionMeta) -> None:
        """Write session metadata to meta.json."""
        s_dir = self.sessions_dir / meta.session_id
        s_dir.mkdir(parents=True, exist_ok=True)
        meta.updated_at = time.time()
        meta_file = s_dir / "meta.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            f.write(meta.model_dump_json(indent=2))

    def get_meta(self, session_id: str) -> SessionMeta | None:
        """Read session metadata from meta.json."""
        if "/" in session_id or "\\" in session_id or session_id in (".", ".."):
            return None
        meta_file = self.sessions_dir / session_id / "meta.json"
        if not meta_file.exists():
            return None
        try:
            with open(meta_file, encoding="utf-8") as f:
                data = json.load(f)
                return SessionMeta(**data)
        except Exception:
            return None

    def is_empty_session(self, meta: SessionMeta, session_dir: Path | None = None) -> bool:
        """Determine whether a session is an empty ghost session with no user interactions."""
        if meta.step_count > 0:
            return False
        if meta.user_goal.strip() not in ("Interactive REPL Session", ""):
            return False

        s_dir = session_dir or (self.sessions_dir / meta.session_id)
        if not s_dir.exists():
            return True

        db_path = s_dir / "checkpoints.db"
        if db_path.exists():
            try:
                import sqlite3

                conn = sqlite3.connect(db_path)
                try:
                    count = conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
                    if count > 0:
                        return False
                finally:
                    conn.close()
            except Exception:
                pass

        episodic = s_dir / "episodic.jsonl"
        if episodic.exists() and episodic.stat().st_size > 0:
            return False

        return True

    def delete_session(self, session_id: str) -> bool:
        """Safely delete all stored data for a specific session."""
        if "/" in session_id or "\\" in session_id or session_id in (".", ".."):
            return False
        target_dir = (self.sessions_dir / session_id).resolve()
        if (
            not target_dir.is_relative_to(self.sessions_dir.resolve())
            or target_dir == self.sessions_dir.resolve()
        ):
            return False
        if target_dir.exists() and target_dir.is_dir():
            shutil.rmtree(target_dir)
            return True
        return False

    def prune_empty_sessions(self) -> int:
        """Purge all empty ghost sessions that contain no steps or dialogue. Returns count of deleted sessions."""
        deleted = 0
        if not self.sessions_dir.exists():
            return 0
        for s_dir in list(self.sessions_dir.iterdir()):
            if s_dir.is_dir():
                meta = self.get_meta(s_dir.name)
                if meta and self.is_empty_session(meta, s_dir):
                    if self.delete_session(s_dir.name):
                        deleted += 1
                elif not meta:
                    try:
                        shutil.rmtree(s_dir)
                        deleted += 1
                    except Exception:
                        pass
        return deleted

    def list_all_sessions(self, filter_empty: bool = False) -> list[SessionMeta]:
        """Scan sessions directory and return sorted session metadata."""
        results: list[SessionMeta] = []
        if not self.sessions_dir.exists():
            return []
        for s_dir in self.sessions_dir.iterdir():
            if s_dir.is_dir():
                meta = self.get_meta(s_dir.name)
                if meta:
                    if filter_empty and self.is_empty_session(meta, s_dir):
                        continue
                    results.append(meta)
        results.sort(key=lambda x: x.created_at, reverse=True)
        return results

    def list_sessions(
        self, limit: int | None = None, filter_empty: bool = True
    ) -> list[SessionMeta]:
        """Scan sessions directory and return sorted session metadata with optional limit and ghost filtering."""
        sessions = self.list_all_sessions(filter_empty=filter_empty)
        if limit is not None:
            return sessions[:limit]
        return sessions
