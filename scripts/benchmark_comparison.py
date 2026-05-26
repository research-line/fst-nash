#!/usr/bin/env python3
"""
Head-to-head Benchmark: Nash Frustration vs B-factors vs AlphaFold pLDDT.

Compares per-residue Nash frustration scores against:
1. Crystallographic B-factors (experimental flexibility)
2. AlphaFold pLDDT scores (prediction confidence = structural order)
3. Ferreiro Frustratometer (if available via server)

Metrics:
  - Pearson/Spearman correlation between score pairs
  - Agreement on top-k frustrated/flexible residues
  - Per-fold-class breakdown

Usage:
    PYTHONIOENCODING=utf-8 python benchmark_comparison.py [--data-dir ../data]
"""

import argparse
import json
import sys
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from protein_fold_nash_pdb import (
    AA20, AA_TO_IDX,
    load_ca_coords_and_seq, extract_phi_psi, build_contacts,
    rbf_features,
)
from joint_training import (
    load_protein, energy_and_grad_v2, JointParams,
    N_GROUP_PAIRS, AA_TO_GROUP, group_pair_index,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR.parent / "data"
CONFIG_PATH = SCRIPT_DIR / "dataset_config.json"


def extract_bfactors(pdb_path: str, chain_id: str) -> np.ndarray:
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("X", pdb_path)
    model = next(structure.get_models())
    chain = model[chain_id]

    residues = [r for r in chain.get_residues()
                if r.id[0] == " " and "N" in r and "CA" in r and "C" in r]

    bfactors = []
    for res in residues:
        bs = [atom.get_bfactor() for atom in res.get_atoms()]
        bfactors.append(np.mean(bs))

    return np.array(bfactors, dtype=float)


def fetch_alphafold_plddt(uniprot_id: str) -> Optional[np.ndarray]:
    url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        if not data:
            return None

        cif_url = data[0].get("cifUrl")
        if not cif_url:
            return None

        with urllib.request.urlopen(cif_url, timeout=30) as resp:
            cif_text = resp.read().decode()

        plddt = []
        for line in cif_text.split("\n"):
            if line.startswith("ATOM") and " CA " in line:
                parts = line.split()
                bfactor_col = parts[-2] if len(parts) > 10 else None
                if bfactor_col:
                    try:
                        plddt.append(float(bfactor_col))
                    except ValueError:
                        pass

        return np.array(plddt, dtype=float) if plddt else None
    except (urllib.error.URLError, json.JSONDecodeError, Exception):
        return None


UNIPROT_MAP = {
    "1YRF": "P02640", "1PGA": "P06654", "1UBQ": "P0CG48",
    "1ENH": "P02836", "1CEI": "P13479", "1SHG": "P07751",
    "1TEN": "P24821", "1BFG": "P09038", "3CHY": "P0AE67",
    "1ERU": "P10599", "1CRN": "P01542", "5PTI": "P00974",
}


def compute_nash_frustration(prot_data, params: JointParams,
                             sigma: float = 1.0, lr: float = 0.05,
                             hess_eps: float = 1e-4) -> np.ndarray:
    N = len(prot_data.seq)
    x0 = np.concatenate([prot_data.phi, prot_data.psi])

    def grad_fn(x):
        ph, ps = x[:N], x[N:]
        _, gph, gps = energy_and_grad_v2(
            ph, ps, prot_data.aa_idx, prot_data.edges, prot_data.r,
            prot_data.group_pair_idx,
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

    J = np.eye(2 * N) - lr * H
    eigvals, eigvecs = np.linalg.eig(J)

    frust = np.zeros(N, dtype=float)
    for k, lam in enumerate(eigvals):
        if abs(lam) >= 1.0:
            weight = abs(lam) - 1.0
            v = eigvecs[:, k].real
            for i in range(N):
                frust[i] += weight * (v[i]**2 + v[i + N]**2)

    if frust.max() > 0:
        frust /= frust.max()

    return frust


def correlate(x: np.ndarray, y: np.ndarray, name_x: str, name_y: str) -> dict:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 5:
        return {"n": len(x), "pearson_r": None, "spearman_rho": None}

    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)

    return {
        "n": len(x),
        "pearson_r": round(float(pr), 4),
        "pearson_p": round(float(pp), 4),
        "spearman_rho": round(float(sr), 4),
        "spearman_p": round(float(sp), 4),
    }


def top_k_overlap(scores_a: np.ndarray, scores_b: np.ndarray,
                  k: int = 10) -> float:
    top_a = set(np.argsort(-scores_a)[:k])
    top_b = set(np.argsort(-scores_b)[:k])
    return len(top_a & top_b) / k


def load_joint_params(results_path: Path) -> JointParams:
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    p = data["params"]
    model = data["config"].get("model", "v1")

    return JointParams(
        mu_phi=np.array(p["mu_phi"]),
        mu_psi=np.array(p["mu_psi"]),
        k_phi=np.array(p["k_phi"]),
        k_psi=np.array(p["k_psi"]),
        w0=np.array(p["w0"]),
        w_rbf=np.array(p["w_rbf"]),
        delta_phi=np.array(p["delta_phi"]),
        delta_psi=np.array(p["delta_psi"]),
        r_centers=np.array(p["r_centers"]),
        model=model,
    )


def main():
    ap = argparse.ArgumentParser(description="Benchmark: Nash Frustration vs B-factors/pLDDT")
    ap.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--config", type=str, default=str(CONFIG_PATH))
    ap.add_argument("--params", type=str, default=None,
                    help="Path to joint_training results JSON")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--rcut", type=float, default=8.0)
    ap.add_argument("--skip-plddt", action="store_true",
                    help="Skip AlphaFold pLDDT fetch")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    results_dir = data_dir.parent / "results"

    if args.params:
        params_path = Path(args.params)
    else:
        for name in ["joint_training_v2_results.json", "joint_training_v1_results.json"]:
            candidate = results_dir / name
            if candidate.exists():
                params_path = candidate
                break
        else:
            print("FEHLER: Keine Joint-Training-Ergebnisse gefunden. Erst joint_training.py ausführen.")
            sys.exit(1)

    print(f"Lade Parameter aus: {params_path}")
    params = load_joint_params(params_path)
    print(f"Modell: {params.model}")

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    print("\n" + "=" * 70)
    print("Head-to-head Benchmark: Nash Frustration vs B-Faktoren vs pLDDT")
    print("=" * 70)

    all_results = []

    for fold_class, proteins in config["proteins"].items():
        for p in proteins:
            pdb_id = p["pdb"]
            pdb_file = data_dir / f"{pdb_id}.pdb"
            if not pdb_file.exists():
                continue

            print(f"\n--- {pdb_id} ({p['name']}, {fold_class}, {p['split']}) ---")

            prot = load_protein(pdb_id, p["chain"], data_dir,
                                name=p["name"], fold_class=fold_class,
                                split=p["split"], r_cut=args.rcut)
            N = len(prot.seq)

            frust = compute_nash_frustration(prot, params, sigma=args.sigma,
                                             lr=args.lr)

            bfactors = extract_bfactors(str(pdb_file), p["chain"])
            if len(bfactors) != N:
                print(f"  WARNUNG: B-Faktor-Länge {len(bfactors)} != Residuen {N}")
                min_n = min(len(bfactors), N)
                bfactors = bfactors[:min_n]
                frust_cmp = frust[:min_n]
            else:
                frust_cmp = frust

            corr_bf = correlate(frust_cmp, bfactors, "Nash", "B-factor")
            k = max(5, N // 10)
            overlap_bf = top_k_overlap(frust_cmp, bfactors, k=k)

            print(f"  Nash vs B-Faktor: Pearson={corr_bf['pearson_r']}, "
                  f"Spearman={corr_bf['spearman_rho']}, "
                  f"Top-{k} Overlap={overlap_bf:.0%}")

            entry = {
                "pdb_id": pdb_id,
                "name": p["name"],
                "fold_class": fold_class,
                "split": p["split"],
                "n_residues": N,
                "nash_vs_bfactor": corr_bf,
                "top_k_overlap_bfactor": round(overlap_bf, 3),
                "top_k": k,
            }

            if not args.skip_plddt and pdb_id in UNIPROT_MAP:
                uniprot = UNIPROT_MAP[pdb_id]
                print(f"  Fetching AlphaFold pLDDT ({uniprot})...")
                plddt = fetch_alphafold_plddt(uniprot)
                if plddt is not None and len(plddt) >= N:
                    plddt_aligned = plddt[:N]
                    inv_plddt = 100.0 - plddt_aligned
                    corr_plddt = correlate(frust_cmp, inv_plddt, "Nash", "1-pLDDT")
                    overlap_plddt = top_k_overlap(frust_cmp, inv_plddt, k=k)
                    entry["nash_vs_plddt"] = corr_plddt
                    entry["top_k_overlap_plddt"] = round(overlap_plddt, 3)
                    print(f"  Nash vs (1-pLDDT): Pearson={corr_plddt['pearson_r']}, "
                          f"Spearman={corr_plddt['spearman_rho']}, "
                          f"Top-{k} Overlap={overlap_plddt:.0%}")
                else:
                    print(f"  pLDDT nicht verfügbar oder Länge passt nicht")

            all_results.append(entry)

    print("\n" + "=" * 70)
    print("ZUSAMMENFASSUNG")
    print("=" * 70)

    bf_pearson = [r["nash_vs_bfactor"]["pearson_r"] for r in all_results
                  if r["nash_vs_bfactor"]["pearson_r"] is not None]
    bf_spearman = [r["nash_vs_bfactor"]["spearman_rho"] for r in all_results
                   if r["nash_vs_bfactor"]["spearman_rho"] is not None]
    bf_overlap = [r["top_k_overlap_bfactor"] for r in all_results]

    print(f"\nNash vs B-Faktor ({len(bf_pearson)} Proteine):")
    print(f"  Pearson r:     {np.mean(bf_pearson):.3f} ± {np.std(bf_pearson):.3f} "
          f"(range {min(bf_pearson):.3f} – {max(bf_pearson):.3f})")
    print(f"  Spearman ρ:    {np.mean(bf_spearman):.3f} ± {np.std(bf_spearman):.3f}")
    print(f"  Top-k Overlap: {np.mean(bf_overlap):.1%} ± {np.std(bf_overlap):.1%}")

    plddt_results = [r for r in all_results if "nash_vs_plddt" in r]
    if plddt_results:
        pl_pearson = [r["nash_vs_plddt"]["pearson_r"] for r in plddt_results]
        pl_spearman = [r["nash_vs_plddt"]["spearman_rho"] for r in plddt_results]
        print(f"\nNash vs (1-pLDDT) ({len(pl_pearson)} Proteine):")
        print(f"  Pearson r:     {np.mean(pl_pearson):.3f} ± {np.std(pl_pearson):.3f}")
        print(f"  Spearman ρ:    {np.mean(pl_spearman):.3f} ± {np.std(pl_spearman):.3f}")

    for fc in ["all_alpha", "all_beta", "alpha_beta", "alpha_plus_beta", "small"]:
        fc_results = [r for r in all_results if r["fold_class"] == fc]
        if fc_results:
            fc_sp = [r["nash_vs_bfactor"]["spearman_rho"] for r in fc_results
                     if r["nash_vs_bfactor"]["spearman_rho"] is not None]
            if fc_sp:
                print(f"  {fc:20s}: Spearman ρ̄ = {np.mean(fc_sp):.3f} (n={len(fc_sp)})")

    out_path = args.out or str(results_dir / "benchmark_comparison_results.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "params_source": str(params_path),
            "model": params.model,
            "lr": args.lr,
            "results": all_results,
            "summary": {
                "bf_pearson_mean": round(float(np.mean(bf_pearson)), 4),
                "bf_spearman_mean": round(float(np.mean(bf_spearman)), 4),
                "bf_overlap_mean": round(float(np.mean(bf_overlap)), 4),
                "n_proteins": len(all_results),
            },
        }, f, indent=2, ensure_ascii=False)
    print(f"\nErgebnisse geschrieben: {out_path}")


if __name__ == "__main__":
    main()
