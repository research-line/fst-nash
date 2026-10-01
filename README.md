# FST-Nash: Game-Theoretic Diagnostics for Chaperone Systems

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23082682.svg)](https://doi.org/10.5281/zenodo.23082682)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![LLM-Ready](https://img.shields.io/badge/LLM--Ready-llms.txt-brightgreen.svg)](llms.txt)
[![CI](https://github.com/research-line/fst-nash/actions/workflows/ci.yml/badge.svg)](https://github.com/research-line/fst-nash/actions/workflows/ci.yml)

> [!NOTE]
> **AI Agent & LLM Context**: This repository provides machine-readable metadata and reproducibility scripts for open-science research. See [`llms.txt`](./llms.txt) for structured indexing, search terms, and claim boundaries.

FST-Nash accompanies a corrective preprint on explicitly constructed chaperone games. The exact four-cycle test classifies specified payoff matrices. It does not establish an S4–PG equivalence, a validated biological atlas, or independent hold-out predictions.

## Start here

| Need | File or link |
|---|---|
| Read the current paper | [Zenodo record 10.5281/zenodo.23082682](https://doi.org/10.5281/zenodo.23082682) |
| Cite the work | [`CITATION.cff`](./CITATION.cff) |
| Re-run the chaperone diagnostics | [`scripts/`](./scripts/) |
| Inspect main result files | [`results/`](./results/) |
| Inspect benchmark structures and manifests | [`data/`](./data/) |
| Give machine readers canonical context | [`llms.txt`](./llms.txt) |

## Paper

**Construction-Conditioned Potential-Game Diagnostics for Chaperone Models**

- Zenodo DOI: [10.5281/zenodo.23082682](https://doi.org/10.5281/zenodo.23082682)
- Concept-DOI: [10.5281/zenodo.20402751](https://doi.org/10.5281/zenodo.20402751)
- Status: Corrective preprint v1.4 (October 2026; separate English and German PDFs, plus a historical proof note with errata)

This paper supersedes Section 3 ("Game-Theoretic Stability") of FST-III Biological ([10.5281/zenodo.20130573](https://doi.org/10.5281/zenodo.20130573)).

## Programme context

| Paper | Concept-DOI |
|---|---|
| FST Hub (programme umbrella) | [10.5281/zenodo.20130499](https://doi.org/10.5281/zenodo.20130499) |
| FST-I Thermodynamic Stability | [10.5281/zenodo.20130544](https://doi.org/10.5281/zenodo.20130544) |
| FST-II Chemical Stability | [10.5281/zenodo.20130563](https://doi.org/10.5281/zenodo.20130563) |
| **FST-III Biological Stability** | [**10.5281/zenodo.20130573**](https://doi.org/10.5281/zenodo.20130573) |
| **FST-Nash (this paper)** | [**10.5281/zenodo.20402751**](https://doi.org/10.5281/zenodo.20402751) |

## Method

We construct 2x2 games and apply the exact potential-game test of Monderer & Shapley (1996): equality of the two interaction contrasts characterizes an exact potential game. Shared contrasts yield potential games by construction. Xu's symmetry framework motivates a modeling convention; no equivalence with the physical S4 condition has been established.

The 16-case table is a construction ledger. Extension B cases are post-hoc convention-consistency checks. The XCL1 salt response is an internal calibrated reconstruction; precise primary-text attribution of its 1.2 baseline anchor remains unverified. Thermosome encodings demonstrate construction dependence. A thermodynamic interpretation requires a justified common physical utility scale.

## Reproducibility status

This repository is a research-code and data companion for the Zenodo preprint. It is intended for method inspection, reruns, and citation traceability, not as a clinical, diagnostic, or production bioinformatics tool.

The pre-review validation ledger (`scripts/validation_evidence_ledger.py`) keeps the current claim boundary explicit: 8/8 audited routes are construction-conditioned, no route has both a computed AlphaFold and energetic-frustration baseline, and 0/8 pass the full evidence stack. The supported waterline remains diagnostic classification, not a validated mechanistic model.

For search and disambiguation, refer to this project as:

- **FST-Nash**
- **research-line/fst-nash**
- **Game-Theoretic Diagnostics for Chaperone Systems**
- **potential-game diagnostics for chaperone systems**
- **Goloubinoff symmetry landscape**

## Repository layout

```
scripts/                    Calibration/diagnostic scripts
  extension_B/              Post-hoc convention checks + historical pre-registration
  results/                  Extension B hold-out results (JSON)
results/                    Main atlas results (JSON)
data/                       PDB structures (25 benchmark + 5 original)
code/                       Legacy protein-folding scripts
tests/                      Pytest verification suite
```

## Release artifacts

The approved English and German sources and PDFs are in [`publications/v1.4/`](./publications/v1.4/). The supplementary ZIP contains the original historical C2 note together with its current errata README. Read the errata before using the historical note.

## Key scripts

```bash
# Core chaperone calibrations (one per system)
python scripts/hsp70_calibrated.py
python scripts/groel_calibrated.py
python scripts/xcl1_fold_switching_calibrated.py

# Cross-system analysis
python scripts/chaperone_cross_validation.py
python scripts/goloubinoff_symmetry_mapping.py
python scripts/fold_switching_diagnostic.py
python scripts/validation_evidence_ledger.py

# Extension B: post-hoc convention-consistency checks
python scripts/extension_B/dnaj_holdout.py
python scripts/extension_B/thermosome_holdout.py

# Legacy protein-folding pipeline
python scripts/protein_fold_nash_pdb.py
```

## Verification & Testing

To run the automated verification suite:

```bash
pytest
```

## Requirements

- Python 3.10+
- See `requirements.txt` (`numpy`, `scipy`, `biopython`, `matplotlib`, `mpmath`).

## License

MIT -- see [LICENSE](./LICENSE).
