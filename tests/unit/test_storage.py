"""Tests for Brain, CheckpointStore, and ArtifactStore."""

from __future__ import annotations

import tempfile
from pathlib import Path

from aglibol.core.types import AgentState
from aglibol.storage.artifact_store import ArtifactStore
from aglibol.storage.brain import Brain
from aglibol.storage.checkpoint import CheckpointStore


def test_brain_initialization():
    with tempfile.TemporaryDirectory() as tmpdir:
        brain = Brain(tmpdir)
        assert brain.sessions_dir.exists()
        assert brain.artifacts_dir.exists()
        assert brain.knowledge_dir.exists()
        assert brain.logs_dir.exists()


def test_checkpoint_store_save_and_restore():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "checkpoints.db"
        store = CheckpointStore(db_path)

        state = AgentState(session_id="test_sess_01", user_goal="Build a tool", current_step=1)
        state.artifacts["main.py"] = "print('hello')"

        ckpt_id = store.save_checkpoint(state)
        assert ckpt_id > 0

        restored = store.get_latest_checkpoint("test_sess_01")
        assert restored is not None
        assert restored.session_id == "test_sess_01"
        assert restored.artifacts["main.py"] == "print('hello')"


def test_artifact_store():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ArtifactStore(Path(tmpdir))
        store.save_artifact("app/main.py", "def run(): pass")

        content = store.read_artifact("app/main.py")
        assert content == "def run(): pass"

        artifacts = store.list_artifacts()
        assert "app/main.py" in artifacts
