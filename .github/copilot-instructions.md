# AI Contributor & Assistant Instructions

This repository is **Aglibol Agent**, an open-source, Ollama-native multi-agent AI framework engineered for personal devices with limited VRAM.

## Architectural Principles

1. **Hardware Tier Awareness**:
   - Hardware capabilities range from Tier 1 (4GB VRAM) to Tier 6 (48GB+ workstation).
   - Any multi-agent workflows must assume strict VRAM constraints unless high-capacity mode is derived.
   - Do NOT introduce heavy mandatory in-memory daemons, uncompressed caches, or multi-gigabyte memory allocations.

2. **Sequential Model Scheduling**:
   - Models are loaded and scheduled sequentially via `ModelScheduler`.
   - VRAM residency is managed with `keep_alive` values derived by `ResourceOptimizer`.

3. **Security Boundaries**:
   - File reading, writing, and directory listing tools MUST validate paths against `workspace_root`.
   - Path traversal (`../`, absolute paths outside workspace) must be rejected with errors.
   - Shell execution commands are filtered through `BLOCKED_PATTERNS` and environment variables are scrubbed of sensitive API tokens case-insensitively.

4. **Coding Style**:
   - Python >= 3.11 with modern type hints (`from __future__ import annotations`).
   - Use Google-style docstrings.
   - Keep CLI output user-friendly and styled via `rich`.
   - Run tests with `pytest tests/ -v`.
