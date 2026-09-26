# Contributing to Aglibol Agent 🚀

Thank you for your interest in contributing to **Aglibol Agent**! We are committed to building an open, accessible, and high-performance multi-agent framework designed specifically for personal computers, consumer GPUs, and local Ollama runtimes.

This document outlines the guidelines and workflow for contributing to the project.

---

## 📜 Table of Contents

1. [Code of Conduct](#-code-of-conduct)
2. [Getting Started & Development Setup](#-getting-started--development-setup)
3. [Coding Standards & Conventions](#-coding-standards--conventions)
4. [Testing Guidelines](#-testing-guidelines)
5. [Pull Request (PR) Workflow](#-pull-request-pr-workflow)
6. [Reporting Bugs & Feature Requests](#-reporting-bugs--feature-requests)
7. [Security Vulnerabilities](#-security-vulnerabilities)

---

## 🤝 Code of Conduct

All contributors and maintainers are expected to follow our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it to understand our community pledge and standards of respectful collaboration.

---

## 🛠 Getting Started & Development Setup

### Prerequisites

- **Python**: `>= 3.11` (Python 3.11, 3.12, and 3.13 are actively tested)
- **Git**: For version control
- **Ollama**: (Optional for unit tests, required for live testing) installed and running locally via `ollama serve`

### Step-by-Step Setup

1. **Fork and Clone the Repository**:
   ```bash
   git clone https://github.com/<your-username>/AglibolAgent.git
   cd AglibolAgent
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv .venv

   # On Linux / macOS:
   source .venv/bin/activate

   # On Windows (PowerShell):
   .venv\Scripts\Activate.ps1
   ```

3. **Install Development Dependencies in Editable Mode**:
   ```bash
   pip install --upgrade pip
   pip install -e ".[dev]"
   ```

4. **Verify Installation**:
   ```bash
   pytest tests/ -v
   ruff check src/ tests/
   ```

---

## 📐 Coding Standards & Conventions

To maintain a clean, maintainable, and globally accessible codebase, we adhere to the following standards:

### 1. Strict English-Only Policy in Code
- **Rule**: All source code files under `src/` and `tests/` must be written exclusively in **English** (variable names, functions, classes, comments, and docstrings).
- **Internationalization (i18n)**: If multilingual keywords or intent-routing phrases are required, use Python Unicode escape sequences (e.g., `\u7db2\u7ad9` for `網站`).
- User-facing terminal displays and documentation may include multilingual translations in designated documentation directories.

### 2. Code Style & Formatting
- **Linter & Formatter**: We use `ruff` for ultra-fast linting and formatting.
  ```bash
  # Check for lint errors
  ruff check src/ tests/

  # Auto-fix lint errors
  ruff check --fix src/ tests/

  # Check format
  ruff format --check src/ tests/
  ```
- **Line Length**: Max 100 characters.
- **Type Annotations**: All public methods and functions must include PEP 484 type hints. Verify types with `mypy`:
  ```bash
  mypy src/aglibol/
  ```

### 3. Hardware-Aware Engineering Principles
- **Zero-OOM Philosophy**: Aglibol Agent is built to run on personal devices with limited VRAM (4 GB to 16 GB).
- Never introduce mandatory background daemon processes, memory leaks, or unconstrained in-memory caches.
- Always use the `ModelScheduler` when invoking Ollama models to ensure sequential unloading (`keep_alive: 0`) in memory-constrained environments.

---

## 🧪 Testing Guidelines

Aglibol Agent maintains a comprehensive test suite. Every new feature, bug fix, or agent role must include corresponding tests.

### Running Tests

```bash
# Run all unit tests
pytest tests/ -v

# Run with test coverage report
pytest --cov=aglibol tests/

# Run a specific test file
pytest tests/unit/test_workflow.py -v
```

### Mocking Ollama & Hardware
- Use the `MockOllamaClient` in `tests/conftest.py` for unit tests to avoid requiring a live GPU or network calls during CI.
- Use `mock_tier1_profile` or `mock_tier3_profile` fixtures to test hardware profiling logic across diverse VRAM configurations.

---

## 🔄 Pull Request (PR) Workflow

1. **Create a Topic Branch**:
   ```bash
   git checkout -b feat/your-feature-name
   # or: git checkout -b fix/issue-description
   ```

2. **Conventional Commits**:
   Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:
   - `feat: add AST self-deadlock detector in reviewer agent`
   - `fix: resolve relative path in workflow YAML loader`
   - `docs: update hardware tier recommendations in README`
   - `test: add unit test for session resumption`
   - `refactor: optimize token observation masking`

3. **Pre-PR Self-Check**:
   Before submitting your PR, verify:
   - [ ] All tests pass: `pytest tests/ -v`
   - [ ] Linter passes with 0 errors: `ruff check src/ tests/`
   - [ ] No non-English characters in `src/` and `tests/`
   - [ ] Code is properly typed and documented

4. **Submit Your PR**:
   - Push your branch to your fork and open a PR against the `main` branch.
   - Fill out the provided [Pull Request Template](.github/pull_request_template.md).
   - Link any related issues (e.g., `Fixes #42`).

---

## 🏷️ Versioning Policy

Aglibol Agent strictly adheres to [Semantic Versioning 2.0.0](https://semver.org/) (`MAJOR.MINOR.PATCH`):

- **MAJOR (`X.0.0`)**: Incompatible API or architectural breaking changes (e.g., modifying `BaseAgent` core signatures, workflow state transitions, or backward-incompatible config schemas).
- **MINOR (`0.X.0` or `X.Y.0`)**: Backward-compatible feature additions, new agent roles (e.g., Tester, Vision), new built-in tools, new hardware tier profiles, or new REPL slash commands.
- **PATCH (`0.1.X` or `X.Y.Z`)**: Backward-compatible bug fixes, security hardening, prompt engineering adjustments, doc improvements, and performance optimizations.
- **Git Tags**: Official releases are tagged with the prefix `v` (e.g., `v0.1.0`). Pushing a `v*.*.*` tag triggers the automated GitHub Actions release pipeline for PyPI and NPM.

### Version Synchronization Checklist
When cutting a new release, maintain version parity across:
1. `pyproject.toml`: `version = "X.Y.Z"`
2. `src/aglibol/__init__.py`: `__version__ = "X.Y.Z"`
3. `npm/package.json`: `"version": "X.Y.Z"`
4. `config/default.yaml`: `version: "X.Y.Z"`
5. `CITATION.cff`: `version: X.Y.Z`
6. `CHANGELOG.md`: create new section for `[X.Y.Z]`

---

## 🐛 Reporting Bugs & Feature Requests

- **Bug Reports**: Please use our [Bug Report Form](https://github.com/Aglibol/AglibolAgent/issues/new?template=bug_report.yml) and include your detected hardware tier, OS, and reproduction steps.
- **Feature Requests**: Please use our [Feature Request Form](https://github.com/Aglibol/AglibolAgent/issues/new?template=feature_request.yml) and describe the motivation, use case, and proposed interface.

---

## 🔒 Security Vulnerabilities

Please **do not report security vulnerabilities via public GitHub issues**. Refer to our [Security Policy](SECURITY.md) for instructions on confidential disclosure via GitHub Security Advisories or `security@aglibol.ai`.

---

Thank you for helping make Aglibol Agent the premier local multi-agent AI framework! 🌟
