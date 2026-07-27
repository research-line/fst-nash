# Changelog

All notable changes to `research-line/fst-nash` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.1] - 2026-07-27

### Fixed
- Make the calibrated GroEL table headers compile on the supported Python 3.10 baseline.

### Changed
- Perform automated technical hygiene and documentation check; update `llms.txt` header to 2026-07-27, verify passing pytest test suite [G 2026-07-27].

## [1.3.0] - 2026-07-25

### Added
- Standardized PEP 621 `pyproject.toml` package metadata and Pytest configuration.
- GitHub Actions CI workflow (`.github/workflows/ci.yml`) testing Python 3.10, 3.11, and 3.12.
- Shields.io status badges (License, Python 3.10+, LLM-Ready, CI) and AI/LLM integration callout in `README.md`.
- `CHANGELOG.md` following Keep a Changelog standards.

### Changed
- Updated `llms.txt` header timestamp to `Last-checked: 2026-07-25`.
- Verified 100% Python compilation and reproducible pre-review evidence ledger execution (`scripts/validation_evidence_ledger.py`).
