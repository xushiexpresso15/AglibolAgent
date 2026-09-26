# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-26

### Added
- **Ollama-Native Multi-Agent Core**: Autonomous Planner, Coder, Reviewer, and Writer agents collaborating on tasks.
- **Dedicated Writer Agent**: Automated classification and copywriting specialist for technical documentation, articles, and whitepapers.
- **Unified Live Telemetry & Thought Streaming**: Real-time thinking animation across all agent modes (Planner, Coder, Reviewer, Writer, Assistant).
- **AST Self-Deadlock & Syntax Gate**: Reviewer agent static analysis catching recursive `threading.Lock` re-entrancy deadlocks before execution.
- **Tool Loop Breaker & Deliverable Enforcement**: Path caching, auto-continuation role awareness, and fallback to direct code generation preventing empty chatter.
- **Continuous Dynamic Hardware Optimization**: Mathematical VRAM & KV-cache profiling supporting 6 hardware tiers (from pure CPU up to 48GB+ workstation GPUs).
- **Sequential Model Scheduler**: VRAM swap-in and swap-out management preventing Out-of-Memory crashes.
- **Interactive Developer Environment (REPL)**:
  - Command palette with 18+ slash commands (`/help`, `/doctor`, `/update`, `/status`, `/model`, `/diff`, `/undo`, `/commit`, `/context`, etc.).
  - `@file` fuzzy workspace autocompletion.
  - Multi-level parameter autocompletion (`/model <role> <model>`, `/mode <strict|balanced|auto>`, `/resume <session_id>`).
  - Real-time prompt with bottom telemetry toolbar.
- **Automated Update Detection (`aglibol update`)**:
  - Non-blocking startup check against GitHub releases with 12-hour TTL cache.
  - Interactive one-click self-upgrade supporting git, pip, and uv installations.
- **Comprehensive System Doctor (`aglibol doctor`)**:
  - Diagnostics for Python runtime, GPU/VRAM, Ollama service, local models, Git, and storage permissions.
  - Automated background service repair and recommended model pull.
- **Security Hardening**:
  - Human-in-the-Loop (HITL) approval gate (strict, balanced, autonomous modes).
  - Workspace path traversal validation for tools, artifacts, and sessions.
  - Windows case-insensitive environment variable token scrubbing.
  - Process tree isolation on Windows and Unix with 1MB output buffer limits.
- **Workflow Engine Resumption**:
  - Resuming interrupted sessions with accurate step continuity and next-node transition resolution.
- **Community & Open Source Standards**:
  - Full GitHub Community Profile adherence (`CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CITATION.cff`, `CODEOWNERS`).
  - EditorConfig and GitAttributes cross-platform normalization.
  - CodeQL SAST and OpenSSF Scorecard supply chain security workflows.
- **Zero-Friction Distribution**:
  - NPX launcher (`npx aglibol`, `npx aglibol-agent`).
  - Universal Linux & macOS installer (`install.sh`).
  - Windows PowerShell native installer (`install.ps1`).
  - PyPI wheel and source tarball.
