## Description
<!-- Please provide a clear and concise summary of the changes and the rationale behind them. -->

## Related Issues
<!-- Link relevant issues: e.g. Fixes #123, Resolves #456 -->

## Type of Change
- [ ] 🐛 Bug fix (non-breaking change which fixes an issue)
- [ ] ✨ New feature (non-breaking change which adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] 📝 Documentation update
- [ ] ⚡ Performance / VRAM optimization
- [ ] 🔒 Security hardening
- [ ] 🧪 Testing and CI improvement

## Contributor Checklist
- [ ] My code follows the PEP 8 / Google Python style guidelines with type hints.
- [ ] **Strict English-only policy**: All code, comments, docstrings, and tests in `src/` and `tests/` are written in English (Unicode escapes used for multilingual keywords).
- [ ] I have self-reviewed my changes and removed debug print statements.
- [ ] I have added unit tests that prove my fix is effective or that my feature works.
- [ ] All existing and new tests pass locally (`pytest tests/ -v`).
- [ ] Linting checks pass with zero errors (`ruff check src/ tests/`).
- [ ] Formatting checks pass (`ruff format --check src/ tests/`).
- [ ] Hardware-Awareness: My changes do NOT introduce mandatory background daemons, heavy dependencies, or large unconstrained memory footprints.
- [ ] I have read and agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md) and [Contributing Guide](CONTRIBUTING.md).
