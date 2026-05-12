# FST-Nash: Protein Folding via Nash Equilibrium

Game-theoretic approach to protein structure prediction using continuous potential games. Companion code repository for the **Functional Stability Theory III: Biological Stability and Nash Frustration** paper.

## Programme context

This repository is the computational arm of **FST-III**, one of four application-scale companions in the **Functional Stability Theory (FST)** programme:

| Paper | Concept-DOI |
|---|---|
| FST Hub (programme umbrella) | [10.5281/zenodo.20130499](https://doi.org/10.5281/zenodo.20130499) |
| FST-I — Thermodynamic Stability of Fundamental Parameters | [10.5281/zenodo.20130544](https://doi.org/10.5281/zenodo.20130544) |
| FST-II — Chemical Stability and Autocatalytic Selection | [10.5281/zenodo.20130563](https://doi.org/10.5281/zenodo.20130563) |
| **FST-III — Biological Stability and Nash Frustration** | [**10.5281/zenodo.20130573**](https://doi.org/10.5281/zenodo.20130573) |
| FST-IV — Cosmological Stability (collector slot) | *forthcoming* |
| FST-DE — Dark Energy as Residual Vacuum Free Energy | [10.5281/zenodo.19036235](https://doi.org/10.5281/zenodo.19036235) |

The companion paper is also linked from [research-line/functional-stability-theory](https://github.com/research-line/functional-stability-theory) (the LaTeX sources for the entire FST programme).

## Method

Each residue is treated as a player choosing backbone angles (phi, psi). The protein folds to a Nash equilibrium where no residue can unilaterally improve its local energy.

**Potential function:**

- Single-residue terms: amino-acid-specific preferences for (phi, psi) angles
- Pairwise coupling: contact-dependent angular correlation via RBF-weighted interactions

**Pipeline:**

1. **Reverse engineering** — Learn interaction parameters from known structures (PDB).
2. **Stability analysis** — Hessian eigenvalue analysis at native state.
3. **Prediction** — Multi-start gradient descent to find Nash equilibria.
4. **Best-response dynamics** — True game-theoretic convergence analysis.
5. **Mutation scoring** — Predict pathogenicity via frustration changes (TP53 ClinVar).

The framework introduces **Nash frustration**, a game-theoretic complement to energetic frustration (Ferreiro et al.), and validates it against NMR chemical-shift perturbation data (Spearman rho_S = 0.44, p = 0.033, n = 24).

## Data

- `data/1PGA.pdb`, `data/1YRF.pdb`, `data/2XWR.pdb` — Training/test protein structures.
- `data/tp53_clinvar_dbd.csv` — TP53 DNA-binding-domain mutations with ClinVar annotations.
- `data/tp53_full_scores.json` — Mutation pathogenicity predictions.
- `results/` — Current benchmark, eta-calibration, eta-scan, extended-analysis, and server-log outputs.
- Root-level `*.pdb` / `*.json` / `*.csv` files are kept for backward compatibility with the initial scripts.

The local raw ClinVar dump `data/variant_summary.txt` is intentionally not tracked because it is about 3.86 GB and can be regenerated with `code/fetch_clinvar_variants.py`.

## Repository layout

- `code/` — exploratory and validation scripts plus intermediate JSON outputs.
- `scripts/` — reproducible analysis entry points used for the current result set.
- `data/` — compact input data and derived ClinVar tables.
- `results/` — generated figures, JSON summaries, LaTeX tables, and run logs.

## Usage

```bash
pip install -r requirements.txt
python scripts/protein_fold_nash_pdb.py    # Main folding + reverse engineering
python scripts/nash_mutation_score.py      # TP53 mutation pathogenicity scoring
python scripts/run_extended_analysis.py    # Batch analysis for current result set
python scripts/eta_calibration_bmrb.py     # eta-calibration against BMRB chemical shifts
python scripts/eta_scan.py                 # eta-scan over the canonical grid
python scripts/frustration_benchmark.py    # Frustration benchmark on the curated set
```

## Requirements

- Python 3.10+
- See `requirements.txt` (`numpy`, `scipy`, `biopython`, `matplotlib`, `mpmath`).

## Status

The FST-III paper is on Zenodo at v1.1 (Concept-DOI [10.5281/zenodo.20130573](https://doi.org/10.5281/zenodo.20130573)). Open follow-ups for v1.2+ (η-calibration refinement, TP53 reanalysis with calibrated η, additional proteins, falsifiable predictions with quantitative thresholds) are tracked in the main FST programme TODO. Internal-only research notes (BEWEISNOTIZ, KONZEPT, AKTIONSPLAN, review chains) remain out of this repository.

## License

MIT — see [LICENSE](./LICENSE).
