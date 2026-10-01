# Historical supplementary note and current errata

**Current paper:** *Construction-Conditioned Potential-Game Diagnostics for Chaperone Models*, corrective preprint v1.4 (October 2026).
**Concept DOI:** [10.5281/zenodo.20402751](https://doi.org/10.5281/zenodo.20402751).

`C2_tautology_proof.md` is the original note dated 2026-05-25, publicly released on 2026-08-13. Its original wording is preserved for traceability. Several assertions in that historical text are incorrect or unsupported. The corrected paper and the following errata govern the current interpretation. This release does not certify the complete historical note as a proof.

## Errata — 2026-10-01

1. **Global-shift zero mode.** The displayed pair-only potential F(theta) = sum w_ij (1 - cos(theta_i - theta_j - delta_ij)) is invariant under theta -> theta + c*1. Differentiating gives H*1 = 0. On the full torus it therefore has no non-degenerate strict local minimum; J = I - eta*H has an eigenvalue 1, hence rho(J) >= 1. Generic non-degeneracy and positive definiteness do not hold for this ungauged pair-only formula. A reduced coordinate system or symmetry-breaking terms changes the problem and requires its own checks.
2. **Conditional lemma retained.** For a C2 potential at a local minimum with a non-degenerate Hessian, H is positive definite. Then 0 < eta < 2/lambda_max(H) implies |1 - eta*lambda_k(H)| < 1 for every eigenvalue, so rho(I - eta*H) < 1. Non-degeneracy and the step bound are prerequisites. This is a gradient-update stability criterion; it alone provides no Nash-equilibrium or biological certificate.
3. **Frustrated couplings.** A global minimum need not have every cosine equal to 1. Incompatible phase offsets can prevent simultaneous satisfaction of all pair terms.
4. **Optimizer termination.** L-BFGS or gradient descent does not guarantee a strict, non-degenerate local minimum. Such a claim requires checking stationarity, curvature, constraints, and numerical tolerances.
5. **Implemented potential differs.** Inspection of `scripts/protein_fold_nash_pdb.py` shows unary terms k_phi[a_i]*(1-cos(phi_i-mu_phi[a_i])) and corresponding psi terms, in addition to pair couplings. The historical pair-only display omits those terms. This is a source-code observation, not a new run or proof that all implemented endpoints have positive-definite Hessians.
6. **Numerical evidence remains limited.** The earlier inventory could substantiate the reported positive minimum Hessian eigenvalue only for 10 fitted runs, not all 60 endpoints. The historical 50/50 random and 10/10 fitted spectral-radius outputs are reported calculations, not independent biological or Nash validation. No new computation is asserted here.

## Archive contents

The supplementary ZIP contains exactly this README and the unchanged historical C2 note. It excludes private working notes, open research directions, and host-specific files.
