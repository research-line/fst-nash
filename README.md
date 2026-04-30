# FST-Nash: Protein Folding via Nash Equilibrium

Game-theoretic approach to protein structure prediction using continuous potential games.

## Method

Each residue is treated as a player choosing backbone angles (phi, psi). The protein folds to a Nash equilibrium where no residue can unilaterally improve its local energy.

**Potential function:**
- Single-residue terms: amino-acid-specific preferences for (phi, psi) angles
- Pairwise coupling: contact-dependent angular correlation via RBF-weighted interactions

**Pipeline:**
1. **Reverse engineering** - Learn interaction parameters from known structures (PDB)
2. **Stability analysis** - Hessian eigenvalue analysis at native state
3. **Prediction** - Multi-start gradient descent to find Nash equilibria
4. **Best-response dynamics** - True game-theoretic convergence analysis
5. **Mutation scoring** - Predict pathogenicity via frustration changes (TP53 ClinVar)

## Data

- `data/1PGA.pdb`, `data/1YRF.pdb`, `data/2XWR.pdb` - Training/test protein structures
- `data/tp53_clinvar_dbd.csv` - TP53 DNA-binding domain mutations with ClinVar annotations
- `data/tp53_full_scores.json` - Mutation pathogenicity predictions
- `results/` - Current benchmark, eta-calibration, eta-scan, extended-analysis, and server-log outputs
- Root-level PDB/JSON/CSV files are kept for backward compatibility with the initial scripts.

The local raw ClinVar dump `data/variant_summary.txt` is intentionally not tracked because it is about 3.86 GB and can be regenerated with `code/fetch_clinvar_variants.py`.

## Repository layout

- `code/` - exploratory and validation scripts plus intermediate JSON outputs
- `scripts/` - reproducible analysis entry points used for the current result set
- `data/` - compact input data and derived ClinVar tables
- `results/` - generated figures, JSON summaries, LaTeX tables, and run logs

## Usage

```bash
python scripts/protein_fold_nash_pdb.py    # Main folding + reverse engineering
python scripts/nash_mutation_score.py      # TP53 mutation pathogenicity scoring
python scripts/run_extended_analysis.py    # Batch analysis for current result set
```

## Requirements

- Python 3.10+
- `pip install -r requirements.txt`

## License

MIT
