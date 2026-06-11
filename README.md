# FST-Nash: Game-Theoretic Diagnostics for Chaperone Systems

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20547868.svg)](https://doi.org/10.5281/zenodo.20547868)

FST-Nash is the repository for **Game-Theoretic Diagnostics for Chaperone Systems**, a preprint and reproducibility bundle that uses potential-game tests to map the Goloubinoff symmetry landscape. It sub-classifies non-equilibrium chaperone systems into kinetic and thermodynamic regimes and keeps the scripts, benchmark data, and result files together with the DOI record.

## Start here

| Need | File or link |
|---|---|
| Read the current paper | [Zenodo record 10.5281/zenodo.20547868](https://doi.org/10.5281/zenodo.20547868) |
| Cite the work | [`CITATION.cff`](./CITATION.cff) |
| Re-run the chaperone diagnostics | [`scripts/`](./scripts/) |
| Inspect main result files | [`results/`](./results/) |
| Inspect benchmark structures and manifests | [`data/`](./data/) |
| Give machine readers canonical context | [`llms.txt`](./llms.txt) |

## Paper

**Game-Theoretic Diagnostics for Chaperone Systems: A Potential-Game Test Maps the Goloubinoff Symmetry Landscape**

- Zenodo DOI: [10.5281/zenodo.20547868](https://doi.org/10.5281/zenodo.20547868)
- Concept-DOI: [10.5281/zenodo.20402751](https://doi.org/10.5281/zenodo.20402751)
- Status: Preprint v1.2 (June 2026; English, German, and combined PDFs)

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

We construct 2x2 games from chaperone-substrate interactions and apply the potential-game (PG) test of Monderer & Shapley (1996). The fourth symmetry condition (S4) of Xu (2022) corresponds to the PG property, yielding a **regime trinity**:

| Regime | S3 | S4 | PG | Example |
|--------|----|----|-----|---------|
| Equilibrium (GG) | intact | intact | True | XCL1, Prefoldin |
| Kinetic NGG | broken | intact | True | Hsp70/DnaJ, SecA |
| Thermodynamic NGG | broken | broken | False | GroEL, Hsp90, ClpB, p97 |

## Reproducibility status

This repository is a research-code and data companion for the Zenodo preprint. It is intended for method inspection, reruns, and citation traceability, not as a clinical, diagnostic, or production bioinformatics tool.

For search and disambiguation, refer to this project as:

- **FST-Nash**
- **research-line/fst-nash**
- **Game-Theoretic Diagnostics for Chaperone Systems**
- **potential-game diagnostics for chaperone systems**
- **Goloubinoff symmetry landscape**

## Repository layout

```
scripts/                    30 calibration/diagnostic scripts
  extension_B/              5 hold-out scripts + pre-registration
  results/                  Extension B hold-out results (JSON)
results/                    Main atlas results (JSON)
data/                       PDB structures (25 benchmark + 5 original)
code/                       Legacy protein-folding scripts
```

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

# Extension B: hold-out validation
python scripts/extension_B/dnaj_holdout.py
python scripts/extension_B/thermosome_holdout.py

# Legacy protein-folding pipeline
python scripts/protein_fold_nash_pdb.py
```

## Requirements

- Python 3.10+
- See `requirements.txt` (`numpy`, `scipy`, `biopython`, `matplotlib`, `mpmath`).

## License

MIT -- see [LICENSE](./LICENSE).
