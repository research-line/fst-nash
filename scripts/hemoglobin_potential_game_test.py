"""
Hemoglobin α₂β₂ Potential-Game-Test
====================================
Prüft, ob ein allosterisches Hämoglobin-Modell ein Potential Game ist
(Monderer & Shapley 1996). Wenn ja → Nash GG = freie-Energie-Minimum,
Spieltheorie fügt keinen Mehrwert hinzu.

4 Spieler: α1, β1, α2, β2
2 Strategien: T (tense, 0) oder R (relaxed, 1)
16 Strategie-Profile

Test: Vier-Zyklen-Bedingung — für jedes Spielerpaar (i,j) und
      jede fixierte Strategie der übrigen Spieler muss die Zykelsumme = 0 sein.

Literatur-Parameter:
- α/β-Asymmetrie: Perutz/Raman (JACS 2014), Yonetani (PNAS 2001)
- Interface-Energie: ~4 kcal/mol am α1-β2-Switch-Interface
"""

import numpy as np
from itertools import product
import json
from pathlib import Path


def compute_utilities(params):
    """Berechne Utility-Matrix für alle Spieler und Profile.

    Returns: u[player][profile_index], profiles list
    """
    h_alpha = params["h_alpha"]
    h_beta = params["h_beta"]
    lam_alpha = params["lambda_alpha"]
    lam_beta = params["lambda_beta"]
    O2 = params["O2"]

    J_intra_ab = params.get("J_intra_ab", params.get("J_intra", 1.0))
    J_intra_ba = params.get("J_intra_ba", params.get("J_intra", 1.0))
    J_inter_ab = params.get("J_inter_ab", params.get("J_inter", 4.0))
    J_inter_ba = params.get("J_inter_ba", params.get("J_inter", 4.0))

    player_types = ["alpha", "beta", "alpha", "beta"]
    neighbors = {
        0: [(1, J_intra_ab), (3, J_inter_ab)],
        1: [(0, J_intra_ba), (2, J_inter_ba)],
        2: [(3, J_intra_ab), (1, J_inter_ab)],
        3: [(2, J_intra_ba), (0, J_inter_ba)],
    }

    profiles = list(product([0, 1], repeat=4))
    u = np.zeros((4, len(profiles)))

    for pi, profile in enumerate(profiles):
        for i in range(4):
            pt = player_types[i]
            si = profile[i]
            h = h_alpha if pt == "alpha" else h_beta
            lam = lam_alpha if pt == "alpha" else lam_beta
            val = h * (1 - si) + lam * si * O2
            for j_idx, J in neighbors[i]:
                if si == profile[j_idx]:
                    val += J
            u[i, pi] = val

    return u, profiles


def test_potential_game(u, profiles):
    """Vier-Zyklen-Test nach Monderer & Shapley (1996).

    Für jedes Spielerpaar (i,j) und jede fixierte Strategie der übrigen:
    Summiere Utility-Differenzen entlang eines geschlossenen Pfades
    (i=0,j=0) → (i=1,j=0) → (i=1,j=1) → (i=0,j=1) → (i=0,j=0).
    Wenn die Summe ≠ 0, ist es KEIN Potential Game.
    """
    n_players = 4
    max_violation = 0.0
    violations = []

    for i in range(n_players):
        for j in range(i + 1, n_players):
            others = [k for k in range(n_players) if k != i and k != j]
            for other_strats in product([0, 1], repeat=len(others)):

                def make_profile(si, sj):
                    p = [0] * n_players
                    p[i] = si
                    p[j] = sj
                    for k, ok in zip(others, other_strats):
                        p[k] = ok
                    return tuple(p)

                p00 = profiles.index(make_profile(0, 0))
                p10 = profiles.index(make_profile(1, 0))
                p01 = profiles.index(make_profile(0, 1))
                p11 = profiles.index(make_profile(1, 1))

                cycle_sum = (
                    (u[i, p00] - u[i, p10])
                    + (u[j, p10] - u[j, p11])
                    + (u[i, p11] - u[i, p01])
                    + (u[j, p01] - u[j, p00])
                )

                if abs(cycle_sum) > 1e-10:
                    violations.append(
                        {
                            "players": (i, j),
                            "others": list(other_strats),
                            "cycle_sum": float(cycle_sum),
                        }
                    )
                    max_violation = max(max_violation, abs(cycle_sum))

    return max_violation, violations


def find_nash_equilibria(u, profiles):
    """Finde alle reinen Nash-Gleichgewichte."""
    n_players = 4
    nash = []
    for pi, profile in enumerate(profiles):
        is_nash = True
        for i in range(n_players):
            alt_profile = list(profile)
            alt_profile[i] = 1 - profile[i]
            alt_pi = profiles.index(tuple(alt_profile))
            if u[i, alt_pi] > u[i, pi] + 1e-10:
                is_nash = False
                break
        if is_nash:
            labels = ["T", "R"]
            nash.append(
                {
                    "profile": tuple(labels[s] for s in profile),
                    "utilities": [float(u[i, pi]) for i in range(4)],
                    "total_utility": float(sum(u[i, pi] for i in range(4))),
                }
            )
    return nash


def main():
    results = {}

    # ── Test 1: Symmetrische Kopplung (Standard-Ising) ──
    params_symmetric = {
        "h_alpha": -1.3,
        "h_beta": 0.0,
        "lambda_alpha": 2.0,
        "lambda_beta": 3.4,
        "O2": 0.5,
        "J_intra": 1.0,
        "J_inter": 4.0,
    }

    u_sym, profiles = compute_utilities(params_symmetric)
    max_v, violations = test_potential_game(u_sym, profiles)
    nash_sym = find_nash_equilibria(u_sym, profiles)

    results["symmetric"] = {
        "description": "Symmetric coupling (J_ab = J_ba), asymmetric intrinsics (h_alpha != h_beta)",
        "params": params_symmetric,
        "is_potential_game": max_v < 1e-10,
        "max_cycle_violation": max_v,
        "n_violations": len(violations),
        "nash_equilibria": nash_sym,
    }

    print("=== Test 1: Symmetrische Kopplung ===")
    print(f"  Max Zykelverletzung: {max_v:.10f}")
    print(f"  Potential Game: {'JA' if max_v < 1e-10 else 'NEIN'}")
    print(f"  Nash-GG: {len(nash_sym)}")
    for n in nash_sym:
        print(f"    {n['profile']}  U_total={n['total_utility']:.2f}")

    # ── Test 2: Asymmetrische Kopplung (non-reciprocal) ──
    params_asym = {
        "h_alpha": -1.3,
        "h_beta": 0.0,
        "lambda_alpha": 2.0,
        "lambda_beta": 3.4,
        "O2": 0.5,
        "J_intra_ab": 1.0,
        "J_intra_ba": 0.6,
        "J_inter_ab": 4.0,
        "J_inter_ba": 2.5,
    }

    u_asym, _ = compute_utilities(params_asym)
    max_v_a, violations_a = test_potential_game(u_asym, profiles)
    nash_asym = find_nash_equilibria(u_asym, profiles)

    results["asymmetric"] = {
        "description": "Asymmetric coupling (J_ab != J_ba), non-reciprocal interface",
        "params": params_asym,
        "is_potential_game": max_v_a < 1e-10,
        "max_cycle_violation": max_v_a,
        "n_violations": len(violations_a),
        "violations": violations_a[:5],
        "nash_equilibria": nash_asym,
    }

    print("\n=== Test 2: Asymmetrische Kopplung ===")
    print(f"  Max Zykelverletzung: {max_v_a:.6f}")
    print(f"  Potential Game: {'JA' if max_v_a < 1e-10 else 'NEIN'}")
    print(f"  Nash-GG: {len(nash_asym)}")
    for n in nash_asym:
        print(f"    {n['profile']}  U_total={n['total_utility']:.2f}")
    if violations_a:
        print(f"  Erste Verletzung: Spieler {violations_a[0]['players']}, "
              f"Zykel={violations_a[0]['cycle_sum']:.4f}")

    # ── Test 3: Asymmetrie-Schwelle ──
    print("\n=== Test 3: Asymmetrie-Schwelle ===")
    thresholds = []
    for delta in np.linspace(0, 2.0, 21):
        params_sweep = {
            "h_alpha": -1.3,
            "h_beta": 0.0,
            "lambda_alpha": 2.0,
            "lambda_beta": 3.4,
            "O2": 0.5,
            "J_intra_ab": 1.0,
            "J_intra_ba": 1.0 - delta * 0.2,
            "J_inter_ab": 4.0,
            "J_inter_ba": 4.0 - delta,
        }
        u_s, _ = compute_utilities(params_sweep)
        mv, viol = test_potential_game(u_s, profiles)
        nash_s = find_nash_equilibria(u_s, profiles)
        thresholds.append({
            "delta": float(delta),
            "J_inter_ba": float(4.0 - delta),
            "max_violation": float(mv),
            "is_potential_game": mv < 1e-10,
            "n_nash": len(nash_s),
        })

    results["threshold_sweep"] = thresholds

    for t in thresholds:
        pg = "PG" if t["is_potential_game"] else "!PG"
        print(f"  δ={t['delta']:.1f}  J_inter_ba={t['J_inter_ba']:.1f}  "
              f"max_v={t['max_violation']:.4f}  {pg}  Nash={t['n_nash']}")

    # ── Test 4: Nash vs. Potential-Optimum bei asymmetrischer Kopplung ──
    print("\n=== Test 4: Nash vs. Energieminimum (asymmetrisch) ===")

    total_u = np.sum(u_asym, axis=0)
    opt_idx = np.argmax(total_u)
    opt_profile = profiles[opt_idx]
    opt_labels = tuple("T" if s == 0 else "R" for s in opt_profile)

    results["nash_vs_optimum"] = {
        "optimum_profile": opt_labels,
        "optimum_total": float(total_u[opt_idx]),
        "nash_profiles": [n["profile"] for n in nash_asym],
        "nash_is_optimum": any(
            n["profile"] == opt_labels for n in nash_asym
        ),
    }

    print(f"  Soziales Optimum: {opt_labels}  U={total_u[opt_idx]:.2f}")
    print(f"  Nash-GG sind Optimum: {results['nash_vs_optimum']['nash_is_optimum']}")
    for n in nash_asym:
        match = " ← OPTIMUM" if n["profile"] == opt_labels else ""
        print(f"    Nash: {n['profile']}  U={n['total_utility']:.2f}{match}")

    # ── Ergebnisse speichern ──
    def convert(obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    out_path = Path(__file__).parent.parent / "results" / "hemoglobin_potential_game_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=convert)
    print(f"\nErgebnisse gespeichert: {out_path}")


if __name__ == "__main__":
    main()
