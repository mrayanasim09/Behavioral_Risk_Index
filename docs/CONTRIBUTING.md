# Contributing Guide

Thank you for your interest in contributing to Behavioral Risk Index (BRI)!

## Development workflow
- Create a feature branch from `main`.
- Keep changes small and self-contained.
- Add/modify tests under `tests/` when changing logic.
- Run unit tests locally before opening a PR.

## Code style
- Python 3.11+.
- Follow PEP8; prefer clear, readable code.
- Avoid hardcoded secrets, tokens, and credentials.
- Configuration belongs in `config.yaml` (or environment variables), not code.

## Data hygiene
- Never commit raw data (`data/raw/`), outputs (`output/`), or logs (`logs/`).
- Use `process_from_raw.py` for processing from local CSVs.
- Use `train_test_from_raw.py` for strict no-leakage train/test.

## No-leakage policy
- Fit scalers/thresholds/weights on training only.
- Apply frozen parameters to test; do not recompute using test.
- Validate out-of-sample metrics separately (e.g., `validation_test.json`).

## PR checklist
- [ ] Code compiles and tests pass
- [ ] No secrets added/modified
- [ ] Updated docs if behavior changed
- [ ] Added tests or validation where applicable

## Getting help
- Open an issue with a minimal reproducible example.
- Tag issues with: `bug`, `feature`, `documentation`, `question`.
