"""Brain manager for disk filesystem layout and persistence."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


class Brain:
    """Manages the disk-first storage root (~/.aglibol) for session state and artifacts."""

    def __init__(self, root_dir: str | Path | None = None) -> None:
        if root_dir:
            self.root = Path(os.path.expanduser(str(root_dir))).resolve()
        else:
            self.root = Path(os.path.expanduser("~/.aglibol")).resolve()

        self.sessions_dir = self.root / "sessions"
        self.knowledge_dir = self.root / "knowledge"
        self.artifacts_dir = self.root / "artifacts"
        self.logs_dir = self.root / "logs"
        self.cache_dir = self.root / "cache"

        self._init_dirs()

    def _init_dirs(self) -> None:
        """Create storage hierarchy if not already present."""
        for d in [
            self.root,
            self.sessions_dir,
            self.knowledge_dir,
            self.artifacts_dir,
            self.logs_dir,
            self.cache_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def get_session_dir(self, session_id: str) -> Path:
        """Get or create the storage directory for a specific session."""
        path = self.sessions_dir / session_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def list_sessions(self) -> list[str]:
        """List all active or completed session IDs on disk."""
        if not self.sessions_dir.exists():
            return []
        return [p.name for p in self.sessions_dir.iterdir() if p.is_dir()]

    def clean_session(self, session_id: str) -> bool:
        """Delete all persisted files for a specific session."""
        if "/" in session_id or "\\" in session_id or session_id in (".", ".."):
            return False
        path = (self.sessions_dir / session_id).resolve()
        if (
            not path.is_relative_to(self.sessions_dir.resolve())
            or path == self.sessions_dir.resolve()
        ):
            return False
        if path.exists() and path.is_dir():
            shutil.rmtree(path)
            return True
        return False

    def prune_empty_sessions(self) -> int:
        """Purge all empty ghost sessions that contain no steps or dialogue."""
        from aglibol.storage.session import SessionManager

        sm = SessionManager(self.sessions_dir)
        return sm.prune_empty_sessions()

    def clean_all(self) -> None:
        """Delete all cached sessions and logs."""
        if self.sessions_dir.exists():
            shutil.rmtree(self.sessions_dir)
            self.sessions_dir.mkdir(parents=True, exist_ok=True)
        if self.logs_dir.exists():
            shutil.rmtree(self.logs_dir)
            self.logs_dir.mkdir(parents=True, exist_ok=True)

    def clean_older_than(self, days: int) -> int:
        """Delete sessions modified more than N days ago. Returns count of deleted sessions."""
        import time

        cutoff = time.time() - (days * 86400)
        deleted = 0
        if self.sessions_dir.exists():
            for p in self.sessions_dir.iterdir():
                if p.is_dir():
                    try:
                        mtime = p.stat().st_mtime
                        if mtime < cutoff:
                            shutil.rmtree(p)
                            deleted += 1
                    except Exception:
                        pass
        return deleted

    def get_disk_usage_mb(self) -> float:
        """Calculate total size of ~/.aglibol in Megabytes."""
        total = 0
        for dirpath, _, filenames in os.walk(self.root):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    pass
        return round(total / (1024**2), 2)
