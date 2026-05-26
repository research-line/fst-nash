#!/usr/bin/env python3
"""
Critical point relaxation check.

From native angles, minimize F to convergence → θ*.
Then compute ρ(J) at θ* (the proper Nash stability test)
and measure ‖θ* − θ_native‖ (how far the critical point is from native).

Also: lr sensitivity check — does ρ(lr) track 1 + lr·|λ_min|?

Usage:
    PYTHONIOENCODING=utf-8 python relaxation_check.py
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from joint_training import (
    load_protein, energy_and_grad_v2, JointParams,
    N_GROUP_PAIRS,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
CONFIG_PATH = SCRIPT_DIR / "dataset_config.json"
RESULTS_DIR = SCRIPT_DIR.parent / "results"


def load_joint_params(results_path: Path) -> JointParams:
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    p = data["params"]
    model = data["config"].get("model", "v1")
    return JointParams(
        mu_phi=np.array(p["mu_phi"]), mu_psi=np.array(p["mu_psi"]),
        k_phi=np.array(p["k_phi"]), k_psi=np.array(p["k_psi"]),
        w0=np.array(p["w0"]), w_rbf=np.array(p["w_rbf"]),
        delta_phi=np.array(p["delta_phi"]), delta_psi=np.array(p["delta_psi"]),
        r_centers=np.array(p["r_centers"]), model=model,
    )


def relax_to_critical_point(prot, params, sigma=1.0, lr_relax=0.005,
                             max_steps=50000, tol=1e-6):
    phi = prot.phi.copy()
    psi = prot.psi.copy()

    for step in range(max_steps):
        F, gphi, gpsi = energy_and_grad_v2(
            phi, psi, prot.aa_idx, prot.edges, prot.r, prot.group_pair_idx,
            params.mu_phi, params.mu_psi, params.k_phi, params.k_psi,
            params.w0, params.w_rbf, params.r_centers, sigma,
            params.delta_phi, params.delta_psi,
        )
        grad_norm = np.sqrt(np.sum(gphi**2) + np.sum(gpsi**2))

        if grad_norm < tol:
            break

        phi -= lr_relax * gphi
        psi -= lr_relax * gpsi
        phi = (phi + np.pi) % (2 * np.pi) - np.pi
        psi = (psi + np.pi) % (2 * np.pi) - np.pi

    return phi, psi, step + 1, float(grad_norm), float(F)


def torus_rms(a, b):
    d = np.arctan2(np.sin(a - b), np.cos(a - b))
    return float(np.sqrt(np.mean(d**2)))


def compute_rho(prot, phi, psi, params, sigma=1.0, lr=0.05, hess_eps=1e-4):
    N = len(prot.seq)
    x0 = np.concatenate([phi, psi])

    def grad_fn(x):
        ph, ps = x[:N], x[N:]
        _, gph, gps = energy_and_grad_v2(
            ph, ps, prot.aa_idx, prot.edges, prot.r, prot.group_pair_idx,
            params.mu_phi, params.mu_psi, params.k_phi, params.k_psi,
            params.w0, params.w_rbf, params.r_centers, sigma,
            params.delta_phi, params.delta_psi,
        )
        return np.concatenate([gph, gps])

    H = np.zeros((2 * N, 2 * N), dtype=float)
    for j in range(2 * N):
        dx = np.zeros_like(x0)
        dx[j] = hess_eps
        gp = grad_fn(x0 + dx)
        gm = grad_fn(x0 - dx)
        H[:, j] = (gp - gm) / (2.0 * hess_eps)
    H = 0.5 * (H + H.T)

    evals_H = np.linalg.eigvalsh(H)
    lambda_min = float(np.min(evals_H))
    pos_frac = float(np.mean(evals_H > 1e-8))

    J = np.eye(2 * N) - lr * H
    eigvals_J = np.linalg.eigvals(J)
    rho = float(np.max(np.abs(eigvals_J)))
    n_unstable = int(np.sum(np.abs(eigvals_J) >= 1.0))

    return rho, n_unstable, lambda_min, pos_frac


def main():
    for name in ["joint_training_v2_results.json", "joint_training_results.json"]:
        path = RESULTS_DIR / name
        if path.exists():
            params_path = path
            break
    else:
        print("Keine Ergebnisse gefunden")
        return

    print(f"Lade Parameter: {params_path.name}")
    params = load_joint_params(params_path)

    with open(CONFIG_PATH) as f:
        config = json.load(f)

    test_proteins = []
    for fold_class, proteins in config["proteins"].items():
        for p in proteins:
            pdb_file = DATA_DIR / f"{p['pdb']}.pdb"
            if not pdb_file.exists():
                continue
            prot = load_protein(p["pdb"], p["chain"], DATA_DIR,
                                name=p["name"], fold_class=fold_class,
                                split=p["split"])
            test_proteins.append(prot)
            if len(test_proteins) >= 10:
                break
        if len(test_proteins) >= 10:
            break

    print(f"\nAnalyse: {len(test_proteins)} Proteine")
    print("=" * 85)

    print(f"\n{'PDB':>5s}  {'Split':>5s}  {'|∇F|_nat':>9s}  {'Steps':>6s}  "
          f"{'|∇F|_θ*':>9s}  {'φ_RMS':>6s}  {'ψ_RMS':>6s}  "
          f"{'ρ_nat':>7s}  {'ρ_θ*':>7s}  {'λ_min':>8s}")
    print("-" * 85)

    all_results = []
    phi_star_first = None
    psi_star_first = None
    for prot in test_proteins:
        _, gphi_nat, gpsi_nat = energy_and_grad_v2(
            prot.phi, prot.psi, prot.aa_idx, prot.edges, prot.r,
            prot.group_pair_idx,
            params.mu_phi, params.mu_psi, params.k_phi, params.k_psi,
            params.w0, params.w_rbf, params.r_centers, 1.0,
            params.delta_phi, params.delta_psi,
        )
        grad_nat = float(np.linalg.norm(np.concatenate([gphi_nat, gpsi_nat])))

        rho_nat, _, _, _ = compute_rho(prot, prot.phi, prot.psi, params)

        phi_star, psi_star, steps, grad_star, F_star = relax_to_critical_point(
            prot, params, lr_relax=0.005, max_steps=100000, tol=1e-6)

        if phi_star_first is None:
            phi_star_first = phi_star.copy()
            psi_star_first = psi_star.copy()

        phi_rms = torus_rms(phi_star, prot.phi)
        psi_rms = torus_rms(psi_star, prot.psi)

        rho_star, n_unstable, lambda_min, pos_frac = compute_rho(
            prot, phi_star, psi_star, params)

        ok = "✓" if rho_star < 1.0 else "✗"
        print(f"{prot.pdb_id:>5s}  {prot.split:>5s}  {grad_nat:9.2e}  {steps:6d}  "
              f"{grad_star:9.2e}  {phi_rms:6.3f}  {psi_rms:6.3f}  "
              f"{rho_nat:7.4f}  {rho_star:7.4f}{ok}  {lambda_min:+8.4f}")

        all_results.append({
            "pdb_id": prot.pdb_id, "split": prot.split,
            "grad_native": grad_nat, "grad_star": grad_star,
            "phi_rms": phi_rms, "psi_rms": psi_rms,
            "rho_native": rho_nat, "rho_star": rho_star,
            "lambda_min": lambda_min, "pos_frac": pos_frac,
            "n_unstable": n_unstable, "steps": steps,
        })

    rho_star_vals = [r["rho_star"] for r in all_results]
    rho_nat_vals = [r["rho_native"] for r in all_results]
    pass_star = sum(1 for r in all_results if r["rho_star"] < 1.0)

    print(f"\n{'='*85}")
    print(f"ρ am nativen Punkt:      {np.mean(rho_nat_vals):.4f} ± {np.std(rho_nat_vals):.4f}")
    print(f"ρ am kritischen Punkt:   {np.mean(rho_star_vals):.4f} ± {np.std(rho_star_vals):.4f}")
    print(f"Nash-stabil bei θ*:      {pass_star}/{len(all_results)}")
    print(f"φ-RMS (θ* vs native):    {np.mean([r['phi_rms'] for r in all_results]):.3f} rad")
    print(f"ψ-RMS (θ* vs native):    {np.mean([r['psi_rms'] for r in all_results]):.3f} rad")

    print(f"\n--- lr-Sensitivitätstest (1. Protein: {test_proteins[0].pdb_id}) ---")
    p0 = test_proteins[0]
    for lr_test in [0.01, 0.02, 0.05, 0.1, 0.2]:
        rho_nat_lr, _, _, _ = compute_rho(p0, p0.phi, p0.psi, params, lr=lr_test)
        rho_star_lr, _, lmin, _ = compute_rho(p0, phi_star_first, psi_star_first, params, lr=lr_test)
        predicted = 1.0 + lr_test * abs(lmin) if lmin < 0 else None
        print(f"  lr={lr_test:.2f}: ρ_nat={rho_nat_lr:.4f}, ρ_θ*={rho_star_lr:.4f}, "
              f"λ_min={lmin:+.4f}" +
              (f", predicted=1+lr|λ|={predicted:.4f}" if predicted else ""))

    out_path = RESULTS_DIR / "relaxation_check_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nErgebnisse: {out_path}")


if __name__ == "__main__":
    main()
