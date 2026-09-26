"""Artifact storage for generated source code, patches, and documentation."""

from __future__ import annotations

from pathlib import Path


class ArtifactStore:
    """Manages file artifacts produced during an agent session."""

    def __init__(self, session_dir: Path) -> None:
        self.artifacts_dir = session_dir / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def save_artifact(self, filename: str, content: str) -> Path:
        """Save text content to the artifacts directory."""
        target_path = self.artifacts_dir / filename
        target_path = target_path.resolve()
        if not target_path.is_relative_to(self.artifacts_dir.resolve()):
            raise ValueError("Path traversal detected")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(content)
        return target_path

    def read_artifact(self, filename: str) -> str | None:
        """Read artifact content from disk."""
        target_path = self.artifacts_dir / filename
        target_path = target_path.resolve()
        if not target_path.is_relative_to(self.artifacts_dir.resolve()):
            raise ValueError("Path traversal detected")
        if not target_path.exists():
            return None
        with open(target_path, encoding="utf-8") as f:
            return f.read()

    def list_artifacts(self) -> list[str]:
        """List all artifact filenames relative to artifacts_dir."""
        if not self.artifacts_dir.exists():
            return []
        return [
            str(p.relative_to(self.artifacts_dir)).replace("\\", "/")
            for p in self.artifacts_dir.rglob("*")
            if p.is_file()
        ]
