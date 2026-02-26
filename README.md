# FST-Nash: Protein Folding via Nash Equilibrium

Game-theoretic approach to protein structure prediction using continuous potential games.

## Method

Each residue is treated as a player choosing backbone angles (phi, psi). The protein folds to a Nash equilibrium where no residue can unilaterally improve its local energy.

**Potential function:**
- Single-residue terms: amino-acid-specific preferences for (phi, psi) angles
- Pairwise coupling: contact-dependent angular correlation via RBF-weighted interactions

**Pipeline:**
1. **Reverse engineering** — Learn interaction parameters from known structures (PDB)
2. **Stability analysis** — Hessian eigenvalue analysis at native state
3. **Prediction** — Multi-start gradient descent to find Nash equilibria
4. **Best-response dynamics** — True game-theoretic convergence analysis
5. **Mutation scoring** — Predict pathogenicity via frustration changes (TP53 ClinVar)

## Data

- `1PGA.pdb`, `1YRF.pdb`, `2XWR.pdb` — Training/test protein structures
- `tp53_clinvar_dbd.csv` — TP53 DNA-binding domain mutations with ClinVar annotations
- `tp53_full_scores.json` — Mutation pathogenicity predictions
- `nash_full_results.json` — Folding results (frustration scores, Hessian stats)

## Usage

```bash
python protein_fold_nash_pdb.py    # Main folding + reverse engineering
python nash_mutation_score.py      # TP53 mutation pathogenicity scoring
```

## Requirements

- Python 3.10+
- numpy

## License

MIT
