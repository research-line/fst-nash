#!/usr/bin/env python3
"""
Joint training of Nash potential parameters across multiple proteins.

Extends the original per-protein model with AA-group-specific pair offsets:
  F_pair = sum_{(i,j)} w(r_ij) * [1 - cos(phi_i - phi_j - delta_phi_g)]
                     + w(r_ij) * [1 - cos(psi_i - psi_j - delta_psi_g)]

where g = group_pair(aa_i, aa_j) indexes one of 21 unique AA-group pairs.

The key scientific point: parameters are shared across ALL training proteins
and evaluated on HELD-OUT test proteins. This avoids the circularity of
per-protein fitting + per-protein evaluation.

Usage:
    PYTHONIOENCODING=utf-8 python joint_training.py [--data-dir ../data]
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import least_squares

from protein_fold_nash_pdb import (
    AA20, AA_TO_IDX,
    load_ca_coords_and_seq, extract_phi_psi, build_contacts,
    rbf_features, numerical_hessian,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR.parent / "data"
CONFIG_PATH = SCRIPT_DIR / "dataset_config.json"

N_GROUPS = 6
N_GROUP_PAIRS = N_GROUPS * (N_GROUPS + 1) // 2  # 21

AA_TO_GROUP = {
    'A': 0, 'G': 0, 'P': 0,
    'V': 1, 'I': 1, 'L': 1, 'M': 1,
    'F': 2, 'W': 2, 'Y': 2,
    'S': 3, 'T': 3, 'N': 3, 'Q': 3, 'C': 3,
    'D': 4, 'E': 4,
    'K': 5, 'R': 5, 'H': 5,
}

GROUP_NAMES = ["small/Pro", "hydrophobic", "aromatic", "polar", "negative", "positive"]


def group_pair_index(g1: int, g2: int) -> int:
    a, b = min(g1, g2), max(g1, g2)
    return a * (2 * N_GROUPS - a - 1) // 2 + (b - a)


def build_group_pair_labels() -> List[str]:
    labels = []
    for g1 in range(N_GROUPS):
        for g2 in range(g1, N_GROUPS):
            labels.append(f"{GROUP_NAMES[g1]}×{GROUP_NAMES[g2]}")
    return labels


@dataclass
class JointParams:
    mu_phi: np.ndarray       # (20,) per-AA preferred phi
    mu_psi: np.ndarray       # (20,) per-AA preferred psi
    k_phi: np.ndarray        # (20,) per-AA spring constants phi
    k_psi: np.ndarray        # (20,) per-AA spring constants psi
    w0: np.ndarray           # (G,) or (1,) coupling bias per group-pair
    w_rbf: np.ndarray        # (G,B) or (1,B) RBF weights
    delta_phi: np.ndarray    # (21,) group-pair angle offsets phi
    delta_psi: np.ndarray    # (21,) group-pair angle offsets psi
    r_centers: np.ndarray    # (B,) RBF centers
    model: str = "v1"        # "v1" = global w, "v2" = per-group w_g


def unpack_joint_params(x: np.ndarray, B: int, model: str = "v1"):
    mu_phi = x[0:20]
    mu_psi = x[20:40]
    k_phi = np.exp(x[40:60])
    k_psi = np.exp(x[60:80])

    if model == "v1":
        w0 = x[80:81]
        w_rbf = x[81:81 + B].reshape(1, B)
        off = 81 + B
    else:
        G = N_GROUP_PAIRS
        w0 = x[80:80 + G]
        w_rbf = x[80 + G:80 + G + G * B].reshape(G, B)
        off = 80 + G + G * B

    delta_phi = x[off:off + N_GROUP_PAIRS]
    delta_psi = x[off + N_GROUP_PAIRS:off + 2 * N_GROUP_PAIRS]
    return mu_phi, mu_psi, k_phi, k_psi, w0, w_rbf, delta_phi, delta_psi


def n_params(B: int, model: str = "v1") -> int:
    if model == "v1":
        return 80 + 1 + B + 2 * N_GROUP_PAIRS
    else:
        G = N_GROUP_PAIRS
        return 80 + G + G * B + 2 * G


@dataclass
class ProteinData:
    pdb_id: str
    name: str
    seq: str
    phi: np.ndarray
    psi: np.ndarray
    aa_idx: np.ndarray
    edges: np.ndarray
    r: np.ndarray
    ca: np.ndarray
    group_pair_idx: np.ndarray  # (M,) group-pair index per edge
    fold_class: str
    split: str


def load_protein(pdb_id: str, chain: str, data_dir: Path,
                 name: str = "", fold_class: str = "", split: str = "",
                 r_cut: float = 8.0) -> ProteinData:
    pdb_path = str(data_dir / f"{pdb_id}.pdb")
    ca, seq = load_ca_coords_and_seq(pdb_path, chain)
    phi, psi = extract_phi_psi(pdb_path, chain)
    edges, r = build_contacts(ca, r_cut=r_cut)
    aa_idx = np.array([AA_TO_IDX.get(a, 0) for a in seq], dtype=int)

    gp_idx = np.zeros(len(edges), dtype=int)
    for m, (i, j) in enumerate(edges):
        g1 = AA_TO_GROUP.get(seq[i], 0)
        g2 = AA_TO_GROUP.get(seq[j], 0)
        gp_idx[m] = group_pair_index(g1, g2)

    return ProteinData(
        pdb_id=pdb_id, name=name, seq=seq,
        phi=phi, psi=psi, aa_idx=aa_idx,
        edges=edges, r=r, ca=ca,
        group_pair_idx=gp_idx,
        fold_class=fold_class, split=split,
    )


def energy_and_grad_v2(phi, psi, aa_idx, edges, r, group_pair_idx,
                       mu_phi, mu_psi, k_phi, k_psi,
                       w0, w_rbf, centers, sigma,
                       delta_phi, delta_psi):
    """Energy and gradient with group-pair offsets.
    w0: (1,) or (G,), w_rbf: (1,B) or (G,B).
    When shapes are (1,...), global coupling; when (G,...), per-group coupling.
    """
    N = len(phi)
    mu_i_phi = mu_phi[aa_idx]
    mu_i_psi = mu_psi[aa_idx]
    k_i_phi = k_phi[aa_idx]
    k_i_psi = k_psi[aa_idx]

    F_u = np.sum(k_i_phi * (1.0 - np.cos(phi - mu_i_phi))) + \
          np.sum(k_i_psi * (1.0 - np.cos(psi - mu_i_psi)))
    g_phi = k_i_phi * np.sin(phi - mu_i_phi)
    g_psi = k_i_psi * np.sin(psi - mu_i_psi)

    Phi = rbf_features(r, centers, sigma)  # (M, B)

    if w0.shape[0] == 1:
        w = w0[0] + Phi @ w_rbf[0]  # (M,)
    else:
        gpi = group_pair_idx
        w = w0[gpi] + np.sum(Phi * w_rbf[gpi], axis=1)  # (M,)

    i_idx = edges[:, 0]
    j_idx = edges[:, 1]

    d_phi_g = delta_phi[group_pair_idx]
    d_psi_g = delta_psi[group_pair_idx]

    dphi = phi[i_idx] - phi[j_idx] - d_phi_g
    dpsi = psi[i_idx] - psi[j_idx] - d_psi_g

    F_p = np.sum(w * (1.0 - np.cos(dphi))) + np.sum(w * (1.0 - np.cos(dpsi)))

    sphi = w * np.sin(dphi)
    spsi = w * np.sin(dpsi)

    np.add.at(g_phi, i_idx, sphi)
    np.add.at(g_phi, j_idx, -sphi)
    np.add.at(g_psi, i_idx, spsi)
    np.add.at(g_psi, j_idx, -spsi)

    return float(F_u + F_p), g_phi, g_psi


def make_joint_params(x: np.ndarray, B: int, r_centers: np.ndarray,
                      model: str = "v1") -> JointParams:
    mu_phi, mu_psi, k_phi, k_psi, w0, w_rbf, delta_phi, delta_psi = \
        unpack_joint_params(x, B, model)
    return JointParams(
        mu_phi=mu_phi, mu_psi=mu_psi,
        k_phi=k_phi, k_psi=k_psi,
        w0=w0, w_rbf=w_rbf,
        delta_phi=delta_phi, delta_psi=delta_psi,
        r_centers=r_centers, model=model,
    )


def joint_fit(proteins: List[ProteinData], B: int = 12, sigma: float = 1.0,
              lam_reg: float = 1e-2, max_nfev: int = 10000,
              model: str = "v1",
              verbose: bool = True) -> Tuple[JointParams, dict]:
    all_r = np.concatenate([p.r for p in proteins])
    r_min, r_max = float(np.min(all_r)), float(np.max(all_r))
    r_centers = np.linspace(r_min, r_max, B)

    np_total = n_params(B, model)
    total_residues = sum(len(p.seq) for p in proteins)
    total_equations = 2 * total_residues

    model_name = "Global w + Offsets" if model == "v1" else "Per-Group w_g + Offsets"
    if verbose:
        print(f"\nJoint Training Setup ({model_name}):")
        print(f"  Proteine: {len(proteins)}")
        print(f"  Residuen gesamt: {total_residues}")
        print(f"  Gleichungen: {total_equations}")
        print(f"  Parameter: {np_total}")
        print(f"  Überbestimmung: {total_equations / np_total:.1f}x")
        print(f"  RBF-Basen: {B}, Sigma: {sigma}")
        print(f"  Regularisierung: {lam_reg}")

    x0 = np.zeros(np_total, dtype=float)
    if model == "v1":
        x0[80] = 0.1
    else:
        x0[80:80 + N_GROUP_PAIRS] = 0.1

    def residual(x):
        mu_phi, mu_psi, k_phi, k_psi, w0, w_rbf, delta_phi, delta_psi = \
            unpack_joint_params(x, B, model)

        all_resid = []
        for p in proteins:
            _, gphi, gpsi = energy_and_grad_v2(
                p.phi, p.psi, p.aa_idx, p.edges, p.r, p.group_pair_idx,
                mu_phi, mu_psi, k_phi, k_psi,
                w0, w_rbf, r_centers, sigma,
                delta_phi, delta_psi,
            )
            all_resid.append(gphi)
            all_resid.append(gpsi)

        stacked = np.concatenate(all_resid)
        reg = np.sqrt(lam_reg) * x
        return np.concatenate([stacked, reg])

    if verbose:
        print(f"\nOptimierung läuft (max_nfev={max_nfev})...")
        t0 = time.time()

    res = least_squares(residual, x0, method="trf", max_nfev=max_nfev,
                        verbose=2 if verbose else 0)

    dt = time.time() - t0 if verbose else 0

    params = make_joint_params(res.x, B, r_centers, model)

    fit_info = {
        "model": model,
        "cost": float(res.cost),
        "optimality": float(res.optimality),
        "nfev": res.nfev,
        "status": int(res.status),
        "message": res.message,
        "n_params": np_total,
        "n_equations": total_equations,
        "time_s": round(dt, 1),
    }

    if verbose:
        print(f"\nFit abgeschlossen in {dt:.1f}s")
        print(f"  Cost: {res.cost:.6f}")
        print(f"  Optimality: {res.optimality:.2e}")
        print(f"  nfev: {res.nfev}")
        print(f"  Status: {res.status} ({res.message})")

    return params, fit_info


def evaluate_protein(p: ProteinData, params: JointParams,
                     sigma: float = 1.0, lr: float = 0.05,
                     hess_eps: float = 1e-4,
                     verbose: bool = True) -> dict:
    _, gphi, gpsi = energy_and_grad_v2(
        p.phi, p.psi, p.aa_idx, p.edges, p.r, p.group_pair_idx,
        params.mu_phi, params.mu_psi, params.k_phi, params.k_psi,
        params.w0, params.w_rbf, params.r_centers, sigma,
        params.delta_phi, params.delta_psi,
    )
    grad_norm = float(np.linalg.norm(np.concatenate([gphi, gpsi])))

    N = len(p.seq)
    x0 = np.concatenate([p.phi, p.psi])
    H = np.zeros((2 * N, 2 * N), dtype=float)

    def grad_fn(x):
        ph, ps = x[:N], x[N:]
        _, gph, gps = energy_and_grad_v2(
            ph, ps, p.aa_idx, p.edges, p.r, p.group_pair_idx,
            params.mu_phi, params.mu_psi, params.k_phi, params.k_psi,
            params.w0, params.w_rbf, params.r_centers, sigma,
            params.delta_phi, params.delta_psi,
        )
        return np.concatenate([gph, gps])

    for j in range(2 * N):
        dx = np.zeros_like(x0)
        dx[j] = hess_eps
        gp = grad_fn(x0 + dx)
        gm = grad_fn(x0 - dx)
        H[:, j] = (gp - gm) / (2.0 * hess_eps)
    H = 0.5 * (H + H.T)

    J = np.eye(2 * N) - lr * H
    eigvals = np.linalg.eigvals(J)
    spectral_radius = float(np.max(np.abs(eigvals)))
    n_unstable = int(np.sum(np.abs(eigvals) >= 1.0))

    evals_H = np.linalg.eigvalsh(H)

    result = {
        "pdb_id": p.pdb_id,
        "name": p.name,
        "fold_class": p.fold_class,
        "split": p.split,
        "n_residues": N,
        "n_contacts": len(p.edges),
        "grad_norm": grad_norm,
        "spectral_radius": spectral_radius,
        "rho_lt_1": spectral_radius < 1.0,
        "n_unstable_modes": n_unstable,
        "hessian_min_eval": float(np.min(evals_H)),
        "hessian_pos_frac": float(np.mean(evals_H > 1e-8)),
    }

    if verbose:
        tag = "TRAIN" if p.split == "train" else "TEST "
        ok = "✓" if spectral_radius < 1.0 else "✗"
        print(f"  [{tag}] {p.pdb_id} ({p.name:30s}): "
              f"ρ={spectral_radius:.4f} {ok}  "
              f"|∇F|={grad_norm:.2e}  "
              f"unstable={n_unstable}/{2*N}")

    return result


def print_learned_offsets(params: JointParams):
    labels = build_group_pair_labels()
    is_v2 = params.w0.shape[0] > 1

    if is_v2:
        print("\nGelernte Gruppen-Parameter (v2):")
        print(f"  {'Paar':35s}  w0_g      δφ (°)   δψ (°)")
        print("  " + "-" * 70)
        for idx, label in enumerate(labels):
            dphi_deg = np.degrees(params.delta_phi[idx])
            dpsi_deg = np.degrees(params.delta_psi[idx])
            w0_g = params.w0[idx]
            marker = " ←" if abs(dphi_deg) > 20 or abs(dpsi_deg) > 20 else ""
            print(f"  {label:35s}  {w0_g:+8.4f}  {dphi_deg:+7.1f}  {dpsi_deg:+7.1f}{marker}")
    else:
        print("\nGelernte Gruppen-Offsets (v1):")
        print(f"  {'Paar':35s}  δφ (°)   δψ (°)")
        print("  " + "-" * 60)
        for idx, label in enumerate(labels):
            dphi_deg = np.degrees(params.delta_phi[idx])
            dpsi_deg = np.degrees(params.delta_psi[idx])
            marker = " ←" if abs(dphi_deg) > 20 or abs(dpsi_deg) > 20 else ""
            print(f"  {label:35s}  {dphi_deg:+7.1f}  {dpsi_deg:+7.1f}{marker}")
        print(f"\n  w0 = {params.w0[0]:.4f}")


def main():
    ap = argparse.ArgumentParser(description="Joint Training FST-Nash Potential")
    ap.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--config", type=str, default=str(CONFIG_PATH))
    ap.add_argument("--B", type=int, default=12)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--rcut", type=float, default=8.0)
    ap.add_argument("--max-nfev", type=int, default=10000)
    ap.add_argument("--hess-eps", type=float, default=1e-4)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--model", choices=["v1", "v2"], default="v1",
                    help="v1=global w + offsets, v2=per-group w_g + offsets")
    ap.add_argument("--train-only", action="store_true",
                    help="Skip held-out evaluation")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    model_label = "v1: Global w + Offsets" if args.model == "v1" else "v2: Per-Group w_g(r) + Offsets"
    print("=" * 70)
    print(f"FST-Nash Joint Training — {model_label}")
    print("=" * 70)

    all_proteins = []
    for fold_class, proteins in config["proteins"].items():
        for p in proteins:
            pdb_file = data_dir / f"{p['pdb']}.pdb"
            if not pdb_file.exists():
                print(f"  WARNUNG: {p['pdb']}.pdb nicht gefunden, übersprungen")
                continue
            try:
                pd = load_protein(
                    p["pdb"], p["chain"], data_dir,
                    name=p["name"], fold_class=fold_class, split=p["split"],
                    r_cut=args.rcut,
                )
                all_proteins.append(pd)
                print(f"  Geladen: {p['pdb']} ({p['name']}, {len(pd.seq)} res, "
                      f"{len(pd.edges)} contacts, {fold_class}, {p['split']})")
            except Exception as e:
                print(f"  FEHLER: {p['pdb']}: {e}")

    train = [p for p in all_proteins if p.split == "train"]
    test = [p for p in all_proteins if p.split == "test"]

    print(f"\nDatensatz: {len(train)} Train, {len(test)} Test, {len(all_proteins)} Gesamt")

    params, fit_info = joint_fit(
        train, B=args.B, sigma=args.sigma,
        lam_reg=args.lam, max_nfev=args.max_nfev,
        model=args.model,
    )

    print_learned_offsets(params)

    print("\n" + "=" * 70)
    print("Evaluation — Spektralradius ρ(J) auf allen Proteinen")
    print("=" * 70)
    print(f"  (lr={args.lr}, ρ < 1 = Nash-stabil)")
    print()

    results = []
    for p in train + test:
        r = evaluate_protein(p, params, sigma=args.sigma, lr=args.lr,
                             hess_eps=args.hess_eps)
        results.append(r)

    train_results = [r for r in results if r["split"] == "train"]
    test_results = [r for r in results if r["split"] == "test"]

    train_rho = [r["spectral_radius"] for r in train_results]
    test_rho = [r["spectral_radius"] for r in test_results]
    train_pass = sum(1 for r in train_results if r["rho_lt_1"])
    test_pass = sum(1 for r in test_results if r["rho_lt_1"])

    print(f"\n{'='*70}")
    print(f"ZUSAMMENFASSUNG:")
    print(f"  Train: {train_pass}/{len(train_results)} mit ρ < 1 "
          f"(ρ̄={np.mean(train_rho):.4f}, range {min(train_rho):.4f}-{max(train_rho):.4f})")
    print(f"  Test:  {test_pass}/{len(test_results)} mit ρ < 1 "
          f"(ρ̄={np.mean(test_rho):.4f}, range {min(test_rho):.4f}-{max(test_rho):.4f})")

    if test_pass == 0:
        print(f"\n  FAZIT: Alle Test-Proteine haben ρ ≥ 1.")
        print(f"  Das Gruppen-Offset-Modell reicht nicht für Nash-Stabilität.")
        print(f"  Nächster Schritt: Gruppenspezifische Kopplungsstärken w_g(r).")
    elif test_pass == len(test_results):
        print(f"\n  FAZIT: Alle Test-Proteine haben ρ < 1!")
        print(f"  Das Modell generalisiert über ungesehene Strukturen.")
    else:
        print(f"\n  FAZIT: Teilweise Nash-Stabilität auf Test-Set.")

    default_name = f"joint_training_{args.model}_results.json"
    out_path = args.out or str(data_dir.parent / "results" / default_name)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    output = {
        "config": {
            "model": args.model,
            "B": args.B, "sigma": args.sigma, "lam_reg": args.lam,
            "lr": args.lr, "rcut": args.rcut, "max_nfev": args.max_nfev,
        },
        "fit_info": fit_info,
        "params": {
            "mu_phi": params.mu_phi.tolist(),
            "mu_psi": params.mu_psi.tolist(),
            "k_phi": params.k_phi.tolist(),
            "k_psi": params.k_psi.tolist(),
            "w0": params.w0.tolist(),
            "w_rbf": params.w_rbf.tolist(),
            "r_centers": params.r_centers.tolist(),
            "delta_phi": params.delta_phi.tolist(),
            "delta_psi": params.delta_psi.tolist(),
            "group_pair_labels": build_group_pair_labels(),
        },
        "results": results,
        "summary": {
            "train_pass": train_pass,
            "train_total": len(train_results),
            "test_pass": test_pass,
            "test_total": len(test_results),
            "train_rho_mean": float(np.mean(train_rho)),
            "test_rho_mean": float(np.mean(test_rho)),
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nErgebnisse geschrieben: {out_path}")


if __name__ == "__main__":
    main()
