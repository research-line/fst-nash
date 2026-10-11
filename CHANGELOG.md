# Changelog

All notable changes to `research-line/fst-nash` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.4.0] - 2026-10-11

### Changed
- Synchronized `pyproject.toml` package metadata (version 1.4.0, corrective title, Zenodo DOI 10.5281/zenodo.23082682).
- Added `tests/test_metadata.py` and expanded `tests/test_validation_ledger.py` for formula and publication artifact parity (7/7 tests green).
- Neutralized internal register references in `scripts/extension_B/` preregistration notes.
- Synchronized calibrated XCL1 reconstruction scripts and JSON evidence ledgers with conservative wording.

## Paper [1.4] - 2026-10-01

### Corrected
- Restricted S4/PG, atlas, hold-out, Nash multiplicity, and thermodynamic interpretations to their construction and scale assumptions.
- Synchronized the English and German manuscripts and eight bibliography entries; retained the 0/8 full-stack validation boundary.
- Released separate English/German sources and PDFs and the historical C2 note with explicit errata. No new experiments or claim upgrades.

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
