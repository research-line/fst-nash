#!/usr/bin/env python3
"""
Random-Parameter-Kontrolle: Ist Nash-Stabilität bei θ* generisch?

Zieht zufällige JointParams (gleiche Skalenprofile wie gefittete), relaxiert
Proteine zu θ*, misst ρ(J) und torus-RMS zu nativ.

Vergleicht gegen die gefitteten Parameter.

Zwei Outcomes:
  A) Random gibt auch ρ<1 mit RMS≈1.8 → Nash-Stabilität generisch, Training wertlos
  B) Random gibt ρ>1 oder andere Statistik → Training hat Struktur gelernt

Usage:
    PYTHONIOENCODING=utf-8 python random_param_control.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from joint_training import (
    load_protein, energy_and_grad_v2, JointParams,
    N_GROUP_PAIRS,
)
from relaxation_check import (
    load_joint_params, relax_to_critical_point, torus_rms, compute_rho,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
CONFIG_PATH = SCRIPT_DIR / "dataset_config.json"
RESULTS_DIR = SCRIPT_DIR.parent / "results"


def generate_random_params(fitted_params, rng, B=12):
    """Random JointParams mit gleichem Skalenprofil wie gefittete."""
    mu_phi_scale = np.std(fitted_params.mu_phi)
    mu_psi_scale = np.std(fitted_params.mu_psi)
    k_phi_scale = np.mean(np.abs(fitted_params.k_phi))
    k_psi_scale = np.mean(np.abs(fitted_params.k_psi))
    w0_scale = np.std(fitted_params.w0)
    w_rbf_scale = np.std(fitted_params.w_rbf)
    delta_phi_scale = np.std(fitted_params.delta_phi)
    delta_psi_scale = np.std(fitted_params.delta_psi)

    n_groups = len(fitted_params.mu_phi)
    w0_shape = fitted_params.w0.shape
    w_rbf_shape = fitted_params.w_rbf.shape

    return JointParams(
        mu_phi=rng.normal(0, max(mu_phi_scale, 0.1), n_groups),
        mu_psi=rng.normal(0, max(mu_psi_scale, 0.1), n_groups),
        k_phi=np.abs(rng.normal(k_phi_scale, k_phi_scale * 0.5, n_groups)),
        k_psi=np.abs(rng.normal(k_psi_scale, k_psi_scale * 0.5, n_groups)),
        w0=rng.normal(0, max(w0_scale, 0.01), w0_shape),
        w_rbf=rng.normal(0, max(w_rbf_scale, 0.01), w_rbf_shape),
        delta_phi=rng.normal(0, max(delta_phi_scale, 0.05), N_GROUP_PAIRS),
        delta_psi=rng.normal(0, max(delta_psi_scale, 0.05), N_GROUP_PAIRS),
        r_centers=fitted_params.r_centers.copy(),
        model=fitted_params.model,
    )


def relax_lbfgs(prot, params, sigma=1.0, max_iter=5000):
    """L-BFGS-B Relaxation als Alternative zu GD."""
    N = len(prot.seq)
    x0 = np.concatenate([prot.phi, prot.psi])

    def objective(x):
        ph, ps = x[:N], x[N:]
        F, gph, gps = energy_and_grad_v2(
            ph, ps, prot.aa_idx, prot.edges, prot.r, prot.group_pair_idx,
            params.mu_phi, params.mu_psi, params.k_phi, params.k_psi,
            params.w0, params.w_rbf, params.r_centers, sigma,
            params.delta_phi, params.delta_psi,
        )
        return float(F), np.concatenate([gph, gps])

    result = minimize(objective, x0, jac=True, method='L-BFGS-B',
                      options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-8})

    phi_star = (result.x[:N] + np.pi) % (2 * np.pi) - np.pi
    psi_star = (result.x[N:] + np.pi) % (2 * np.pi) - np.pi
    grad_norm = float(np.linalg.norm(result.jac))
    return phi_star, psi_star, result.nit, grad_norm, float(result.fun)


def run_control(proteins, fitted_params, n_seeds=5):
    """Haupttest: Random vs Fitted."""
    rng = np.random.default_rng(42)

    print("=" * 90)
    print("RANDOM-PARAMETER-KONTROLLE")
    print("=" * 90)

    # --- Fitted: L-BFGS Relaxation (besser als GD) ---
    print("\n--- Fitted Parameter (L-BFGS) ---")
    print(f"{'PDB':>5s}  {'|∇F|_θ*':>9s}  {'ρ_θ*':>7s}  {'φ_RMS':>6s}  {'ψ_RMS':>6s}  {'λ_min':>8s}  {'Iter':>5s}")
    print("-" * 60)

    fitted_results = []
    for prot in proteins:
        phi_s, psi_s, nit, gnorm, Fval = relax_lbfgs(prot, fitted_params)
        rho, n_unst, lmin, pfrac = compute_rho(prot, phi_s, psi_s, fitted_params)
        pr = torus_rms(phi_s, prot.phi)
        qr = torus_rms(psi_s, prot.psi)
        ok = "✓" if rho < 1.0 else "✗"
        print(f"{prot.pdb_id:>5s}  {gnorm:9.2e}  {rho:7.4f}{ok}  {pr:6.3f}  {qr:6.3f}  {lmin:+8.4f}  {nit:5d}")
        fitted_results.append({
            "pdb_id": prot.pdb_id, "rho": rho, "phi_rms": pr, "psi_rms": qr,
            "lambda_min": lmin, "grad_norm": gnorm,
        })

    fitted_rhos = [r["rho"] for r in fitted_results]
    fitted_phi_rms = [r["phi_rms"] for r in fitted_results]
    fitted_psi_rms = [r["psi_rms"] for r in fitted_results]

    print(f"\nFitted: ρ̄ = {np.mean(fitted_rhos):.4f} ± {np.std(fitted_rhos):.4f}, "
          f"pass={sum(1 for r in fitted_rhos if r < 1)}/{len(fitted_rhos)}")
    print(f"        φ-RMS = {np.mean(fitted_phi_rms):.3f}, ψ-RMS = {np.mean(fitted_psi_rms):.3f}")

    # --- Random seeds ---
    all_random_rhos = []
    all_random_phi_rms = []
    all_random_psi_rms = []

    for seed_idx in range(n_seeds):
        print(f"\n--- Random Seed {seed_idx + 1}/{n_seeds} ---")
        rand_params = generate_random_params(fitted_params, rng)

        seed_rhos = []
        seed_phi = []
        seed_psi = []
        for prot in proteins:
            phi_s, psi_s, nit, gnorm, Fval = relax_lbfgs(prot, rand_params)
            rho, _, lmin, _ = compute_rho(prot, phi_s, psi_s, rand_params)
            pr = torus_rms(phi_s, prot.phi)
            qr = torus_rms(psi_s, prot.psi)
            ok = "✓" if rho < 1.0 else "✗"
            print(f"  {prot.pdb_id:>5s}: ρ={rho:.4f}{ok}  φ-RMS={pr:.3f}  ψ-RMS={qr:.3f}  "
                  f"|∇F|={gnorm:.2e}  λ_min={lmin:+.4f}")
            seed_rhos.append(rho)
            seed_phi.append(pr)
            seed_psi.append(qr)

        pass_rate = sum(1 for r in seed_rhos if r < 1) / len(seed_rhos)
        print(f"  Seed {seed_idx+1}: ρ̄={np.mean(seed_rhos):.4f}, pass={pass_rate:.0%}, "
              f"φ-RMS={np.mean(seed_phi):.3f}, ψ-RMS={np.mean(seed_psi):.3f}")
        all_random_rhos.extend(seed_rhos)
        all_random_phi_rms.extend(seed_phi)
        all_random_psi_rms.extend(seed_psi)

    # --- Zusammenfassung ---
    print("\n" + "=" * 90)
    print("ZUSAMMENFASSUNG")
    print("=" * 90)

    random_pass = sum(1 for r in all_random_rhos if r < 1) / len(all_random_rhos)
    fitted_pass = sum(1 for r in fitted_rhos if r < 1) / len(fitted_rhos)

    print(f"\n{'':>20s}  {'ρ̄':>7s}  {'ρ_std':>7s}  {'pass%':>6s}  {'φ-RMS':>7s}  {'ψ-RMS':>7s}")
    print("-" * 65)
    print(f"{'Fitted':>20s}  {np.mean(fitted_rhos):7.4f}  {np.std(fitted_rhos):7.4f}  "
          f"{fitted_pass:6.0%}  {np.mean(fitted_phi_rms):7.3f}  {np.mean(fitted_psi_rms):7.3f}")
    print(f"{'Random (n={})'.format(len(all_random_rhos)):>20s}  "
          f"{np.mean(all_random_rhos):7.4f}  {np.std(all_random_rhos):7.4f}  "
          f"{random_pass:6.0%}  {np.mean(all_random_phi_rms):7.3f}  {np.mean(all_random_psi_rms):7.3f}")
    print(f"{'Uniform-Torus-RMS':>20s}  {'':>7s}  {'':>7s}  {'':>6s}  {np.pi/np.sqrt(3):7.3f}  {np.pi/np.sqrt(3):7.3f}")

    if random_pass > 0.6:
        print("\n→ ERGEBNIS A: Nash-Stabilität bei θ* ist GENERISCH.")
        print("  Random-Parameter erzeugen ebenfalls ρ<1 bei θ*.")
        print("  Das Training hat nichts Strukturelles über nativ hinaus gelernt.")
    elif random_pass < 0.3:
        print("\n→ ERGEBNIS B: Nash-Stabilität bei θ* ist NICHT generisch.")
        print("  Das Training hat tatsächlich etwas Strukturelles gelernt,")
        print("  nur landen die Minima nicht bei der nativen Konformation.")
    else:
        print(f"\n→ ERGEBNIS: Uneindeutig (random pass={random_pass:.0%}).")
        print("  Mehr Seeds oder andere Skalenprofile nötig.")

    return {
        "fitted": fitted_results,
        "random_rhos": all_random_rhos,
        "random_phi_rms": all_random_phi_rms,
        "random_psi_rms": all_random_psi_rms,
        "n_seeds": n_seeds,
        "fitted_pass_rate": fitted_pass,
        "random_pass_rate": random_pass,
    }


def main():
    for name in ["joint_training_v2_results.json", "joint_training_results.json"]:
        path = RESULTS_DIR / name
        if path.exists():
            params_path = path
            break
    else:
        print("Keine Ergebnisse gefunden")
        return

    print(f"Lade gefittete Parameter: {params_path.name}")
    fitted_params = load_joint_params(params_path)

    with open(CONFIG_PATH) as f:
        config = json.load(f)

    proteins = []
    for fold_class, prots in config["proteins"].items():
        for p in prots:
            pdb_file = DATA_DIR / f"{p['pdb']}.pdb"
            if not pdb_file.exists():
                continue
            prot = load_protein(p["pdb"], p["chain"], DATA_DIR,
                                name=p["name"], fold_class=fold_class,
                                split=p["split"])
            proteins.append(prot)
            if len(proteins) >= 10:
                break
        if len(proteins) >= 10:
            break

    print(f"Proteine geladen: {len(proteins)}")

    results = run_control(proteins, fitted_params, n_seeds=5)

    out_path = RESULTS_DIR / "random_param_control_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=float)
    print(f"\nErgebnisse: {out_path}")


if __name__ == "__main__":
    main()
