# Contributing to Aglibol Agent

Thank you for your interest in contributing to Aglibol Agent. We welcome contributions from developers of all backgrounds to help build an accessible, high-performance multi-agent framework designed for personal devices, consumer GPUs, and local Ollama runtimes.

This document outlines the development workflow, coding standards, testing requirements, and contribution guidelines.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Development Setup](#development-setup)
- [Coding Standards and Conventions](#coding-standards-and-conventions)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Workflow](#pull-request-workflow)
- [Versioning Policy](#versioning-policy)
- [Issue Reporting](#issue-reporting)
- [Security Disclosures](#security-disclosures)

---

## Code of Conduct

All contributors and maintainers are expected to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please review it to understand our community standards and expectations for respectful collaboration.

---

## Development Setup

### Prerequisites

- **Python**: `>= 3.11` (Python 3.11, 3.12, and 3.13 are verified in CI)
- **Git**: For source version control
- **Ollama**: (Optional for unit tests with mocks, required for live testing) installed and accessible locally

### Setup Instructions

1. **Fork and Clone the Repository**:
   ```bash
   git clone https://github.com/<your-username>/AglibolAgent.git
   cd AglibolAgent
   ```

2. **Create and Activate a Virtual Environment**:
   ```bash
   python -m venv .venv

   # Linux / macOS:
   source .venv/bin/activate

   # Windows (PowerShell):
   .venv\Scripts\Activate.ps1
   ```

3. **Install Development Dependencies**:
   ```bash
   python -m pip install --upgrade pip
   pip install -e ".[dev]"
   ```

4. **Verify the Environment**:
   ```bash
   python -m pytest tests/ -v
   ruff check src/ tests/
   ruff format --check src/ tests/
   ```

---

## Coding Standards and Conventions

### 1. English-Only Policy in Source Files
- All source files under `src/` and `tests/` must be written strictly in **English** (identifiers, variable names, classes, functions, inline comments, and docstrings).
- For multilingual keyword matching or intent recognition, use standard Python Unicode escape sequences (for example, `\u7db2\u7ad9` for `網站`).
- User-facing terminal localizations and documentation translations should reside in designated documentation or locale directories.

### 2. Code Formatting and Linting
- We use `ruff` for code style enforcement and linting.
  ```bash
  # Check for lint errors
  ruff check src/ tests/

  # Automatically fix fixable issues
  ruff check --fix src/ tests/

  # Check code formatting
  ruff format --check src/ tests/

  # Format codebase
  ruff format src/ tests/
  ```
- **Line Length**: 100 characters maximum.
- **Type Annotations**: Public methods, classes, and helper functions must include PEP 484 type hints. Run `mypy` to verify static type checking:
  ```bash
  mypy src/aglibol/
  ```

### 3. Hardware-Aware Engineering Principles
- Aglibol Agent is built to prevent CUDA Out-Of-Memory (OOM) failures on devices with limited memory (4 GB to 16 GB VRAM).
- Avoid unconstrained in-memory caches, background daemon overhead, or heavy third-party runtime dependencies.
- Always route model requests through `ModelScheduler` to manage sequential residency and eviction (`keep_alive: 0`) under memory-constrained profiles.

---

## Testing Guidelines

Every new feature, bug fix, or agent role must include automated test coverage.

### Running Tests

```bash
# Run all unit tests
python -m pytest tests/ -v

# Run with test coverage measurement
python -m pytest --cov=aglibol tests/

# Run a targeted test module
python -m pytest tests/unit/test_workflow.py -v
```

### Mocking and Test Isolation
- Use `MockOllamaClient` defined in `tests/conftest.py` for unit testing to prevent external network dependencies or live GPU requirements.
- Use `mock_tier1_profile` or `mock_tier3_profile` fixtures to simulate specific hardware constraints deterministically.
- Unit tests must be hermetic and execute cleanly on Linux, macOS, and Windows runners without requiring a running Ollama daemon.

---

## Pull Request Workflow

1. **Create a Topic Branch**:
   ```bash
   git checkout -b feat/your-feature-name
   # or: git checkout -b fix/issue-description
   ```

2. **Commit Messages**:
   Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:
   - `feat: add AST self-deadlock detector in reviewer agent`
   - `fix: resolve relative path in workflow YAML loader`
   - `docs: update hardware tier recommendations in README`
   - `test: add unit test for session resumption`
   - `refactor: optimize token observation masking`

3. **Pre-Submission Checklist**:
   Prior to opening a pull request, ensure:
   - [ ] All unit tests pass locally: `python -m pytest tests/ -v`
   - [ ] Linter reports 0 errors: `ruff check src/ tests/`
   - [ ] Formatter reports 0 discrepancies: `ruff format --check src/ tests/`
   - [ ] Zero non-English characters in `src/` and `tests/`
   - [ ] Public symbols and functions include type annotations and docstrings

4. **Submitting the PR**:
   - Push your branch to your GitHub fork and submit a PR to `main`.
   - Complete the [Pull Request Template](.github/pull_request_template.md).
   - Reference any associated issue (for example, `Fixes #42`).

---

## Versioning Policy

Aglibol Agent adheres to [Semantic Versioning 2.0.0](https://semver.org/) (`MAJOR.MINOR.PATCH`):

- **MAJOR (`X.0.0`)**: Incompatible API or architectural breaking changes (e.g., core agent signatures, workflow state schemas).
- **MINOR (`0.X.0` or `X.Y.0`)**: Backward-compatible feature additions, new agent roles, built-in tools, hardware profiles, or slash commands.
- **PATCH (`0.1.X` or `X.Y.Z`)**: Backward-compatible bug fixes, security patches, prompt adjustments, and documentation updates.
- **Git Tags**: Release tags use the `v` prefix (for example, `v0.1.0`). Pushing a release tag initiates the automated Release workflow for packaging and distribution.

### Version Parity Across Project Files
When releasing a new version, verify consistency across:
1. `pyproject.toml`: `version = "X.Y.Z"`
2. `src/aglibol/__init__.py`: `__version__ = "X.Y.Z"`
3. `npm/package.json`: `"version": "X.Y.Z"`
4. `config/default.yaml`: `version: "X.Y.Z"`
5. `CITATION.cff`: `version: X.Y.Z`
6. `CHANGELOG.md`: dedicated release notes section

---

## Issue Reporting

- **Bug Reports**: Please open an issue using the [Bug Report Template](https://github.com/xushiexpresso15/AglibolAgent/issues/new?template=bug_report.yml) and include detected hardware profile, operating system, and reproduction steps.
- **Feature Requests**: Submit proposals via the [Feature Request Template](https://github.com/xushiexpresso15/AglibolAgent/issues/new?template=feature_request.yml) with problem statements and suggested design.

---

## Security Disclosures

Do not report potential security vulnerabilities through public GitHub issues. Please review our [Security Policy](SECURITY.md) for confidential reporting procedures via GitHub Security Advisories or maintainer contact channels.
