"""Unit tests for system diagnostics and doctor command."""

import pytest

from aglibol.cli.commands.doctor import run_diagnostics


@pytest.mark.asyncio
async def test_run_diagnostics():
    results = await run_diagnostics(auto_fix=False)
    assert len(results) >= 6

    categories = {r.category for r in results}
    assert "Runtime" in categories
    assert "Hardware" in categories
    assert "Ollama" in categories
    assert "Storage" in categories

    # Python version must pass
    py_check = next(r for r in results if r.name == "Python Version")
    assert py_check.status == "PASS"

    # Brain persistence must pass
    brain_check = next(r for r in results if r.name == "Brain Persistence")
    assert brain_check.status == "PASS"


@pytest.mark.asyncio
async def test_doctor_daemon_fix_action_is_callable():
    results = await run_diagnostics(auto_fix=False)
    daemon_check = next((r for r in results if r.name == "Daemon Connection"), None)
    assert daemon_check is not None
    if daemon_check.status == "FAIL":
        assert daemon_check.fix_available is True
        assert daemon_check.fix_action is not None
        assert callable(daemon_check.fix_action)


@pytest.mark.asyncio
async def test_ollama_client_start_daemon_when_healthy(monkeypatch):
    from aglibol.ollama.client import OllamaClient

    client = OllamaClient()

    # Mock health_check to return True immediately
    async def mock_health():
        return True

    monkeypatch.setattr(client, "health_check", mock_health)

    started = await client.start_daemon(timeout_seconds=1.0)
    assert started is True
