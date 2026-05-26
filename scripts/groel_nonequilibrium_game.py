"""
GroEL/GroES als Nicht-Gleichgewichts-Spiel
==========================================
2 Spieler = 2 heptamere Ringe (cis, trans)
3 Strategien pro Ring: T (tense), R (relaxed), R'' (ATP-gebunden, GroES-capped)

Nicht-reziproke Kopplung: ATP-Hydrolyse im cis-Ring beeinflusst trans-Ring
stärker als umgekehrt (gerichtete Inter-Ring-Signalisierung).

Test: Ist das System ein Potential Game?
Vorhersage: NEIN — wegen nicht-reziproker Kopplung.

Literatur:
- Yifrach & Horovitz (2006) Nat Struct Mol Biol — allosterische Signalisierung
- Todd et al. (2010) PNAS — Out-of-equilibrium cycling
- Gruber & Bhatt (2023) J Chem Phys 157:184104 — stat. mech. Modelle
"""

import numpy as np
from itertools import product
import json
from pathlib import Path

STATES = {"T": 0, "R": 1, "R''": 2}
STATE_NAMES = ["T", "R", "R''"]


def compute_utilities_groel(params):
    """Berechne Utility für cis- und trans-Ring.

    Nicht-reziproke Kopplung:
    - Cis-Hydrolyse → starker Effekt auf Trans (J_ct)
    - Trans → schwacher Rückeffekt auf Cis (J_tc < J_ct)
    """
    # Intrinsische Präferenzen (abhängig vom Substrat-Zustand)
    # T: Substrat-Bindung, R: intermediär, R'': Faltungskammer aktiv
    h_cis = np.array(params["h_cis"])    # [h_T, h_R, h_R'']
    h_trans = np.array(params["h_trans"])

    # Kopplungsmatrizen: J[s_own][s_other]
    # Cis sieht Trans: J_ct[s_cis][s_trans]
    J_ct = np.array(params["J_ct"])  # 3x3
    # Trans sieht Cis: J_tc[s_trans][s_cis]
    J_tc = np.array(params["J_tc"])  # 3x3

    profiles = list(product(range(3), repeat=2))  # (s_cis, s_trans)

    u_cis = np.zeros(len(profiles))
    u_trans = np.zeros(len(profiles))

    for pi, (sc, st) in enumerate(profiles):
        u_cis[pi] = h_cis[sc] + J_ct[sc, st]
        u_trans[pi] = h_trans[st] + J_tc[st, sc]

    return u_cis, u_trans, profiles


def test_potential_game_2player(u1, u2, profiles, n_strategies=3):
    """Vier-Zyklen-Test für 2-Spieler-Spiel mit n Strategien.

    Für jedes Strategiepaar (a,b) und (c,d) von Spieler 1 und 2:
    u1(a,c) - u1(b,c) + u2(b,d) - u2(b,c) + u1(b,d) - u1(a,d) + u2(a,c) - u2(a,d) = 0?

    Äquivalent: Für alle s1, s1', s2, s2':
    [u1(s1,s2) - u1(s1',s2)] - [u1(s1,s2') - u1(s1',s2')] =
    [u2(s1,s2) - u2(s1,s2')] - [u2(s1',s2) - u2(s1',s2')]
    """
    max_violation = 0.0
    violations = []

    for s1 in range(n_strategies):
        for s1p in range(n_strategies):
            if s1p == s1:
                continue
            for s2 in range(n_strategies):
                for s2p in range(n_strategies):
                    if s2p == s2:
                        continue

                    p_ss = profiles.index((s1, s2))
                    p_sp_s = profiles.index((s1p, s2))
                    p_s_sp = profiles.index((s1, s2p))
                    p_sp_sp = profiles.index((s1p, s2p))

                    lhs = (u1[p_ss] - u1[p_sp_s]) - (u1[p_s_sp] - u1[p_sp_sp])
                    rhs = (u2[p_ss] - u2[p_s_sp]) - (u2[p_sp_s] - u2[p_sp_sp])

                    diff = abs(lhs - rhs)
                    if diff > 1e-10:
                        violations.append({
                            "s1": STATE_NAMES[s1], "s1'": STATE_NAMES[s1p],
                            "s2": STATE_NAMES[s2], "s2'": STATE_NAMES[s2p],
                            "lhs": float(lhs), "rhs": float(rhs),
                            "violation": float(diff),
                        })
                        max_violation = max(max_violation, diff)

    return max_violation, violations


def find_nash_2player(u1, u2, profiles, n_strategies=3):
    """Finde reine Nash-GG für 2-Spieler-Spiel."""
    nash = []
    for pi, (s1, s2) in enumerate(profiles):
        is_nash = True
        for s1_alt in range(n_strategies):
            if s1_alt == s1:
                continue
            pi_alt = profiles.index((s1_alt, s2))
            if u1[pi_alt] > u1[pi] + 1e-10:
                is_nash = False
                break
        if not is_nash:
            continue
        for s2_alt in range(n_strategies):
            if s2_alt == s2:
                continue
            pi_alt = profiles.index((s1, s2_alt))
            if u2[pi_alt] > u2[pi] + 1e-10:
                is_nash = False
                break
        if is_nash:
            nash.append({
                "profile": (STATE_NAMES[s1], STATE_NAMES[s2]),
                "u_cis": float(u1[pi]),
                "u_trans": float(u2[pi]),
                "u_total": float(u1[pi] + u2[pi]),
            })
    return nash


def main():
    results = {}

    # ── Test 1: Symmetrische Kopplung (Gleichgewicht) ──
    params_sym = {
        "h_cis":  [0.0, 2.0, 5.0],   # T=baseline, R=moderate, R''=Faltung aktiv
        "h_trans": [0.0, 2.0, 5.0],   # Gleiche intrinsische Präferenzen
        "J_ct": [                       # Cis sieht Trans (symmetrisch)
            [3.0, -1.0, -2.0],         # Cis=T: bevorzugt Trans=T
            [-1.0, 2.0, 0.0],          # Cis=R: bevorzugt Trans=R
            [-2.0, 0.0, 1.0],          # Cis=R'': schwache Kopplung
        ],
        "J_tc": [                       # Trans sieht Cis (= J_ct transponiert)
            [3.0, -1.0, -2.0],
            [-1.0, 2.0, 0.0],
            [-2.0, 0.0, 1.0],
        ],
    }

    u1_s, u2_s, profiles = compute_utilities_groel(params_sym)
    max_v_s, viol_s = test_potential_game_2player(u1_s, u2_s, profiles)
    nash_s = find_nash_2player(u1_s, u2_s, profiles)

    results["symmetric"] = {
        "description": "Symmetric coupling (equilibrium, J_ct = J_tc^T)",
        "is_potential_game": bool(max_v_s < 1e-10),
        "max_violation": float(max_v_s),
        "n_violations": len(viol_s),
        "nash_equilibria": nash_s,
    }

    print("=== Test 1: Symmetrische Kopplung (Gleichgewicht) ===")
    print(f"  Potential Game: {'JA' if max_v_s < 1e-10 else 'NEIN'}")
    print(f"  Max Verletzung: {max_v_s:.6f}")
    print(f"  Nash-GG: {len(nash_s)}")
    for n in nash_s:
        print(f"    Cis={n['profile'][0]}, Trans={n['profile'][1]}  "
              f"U={n['u_total']:.1f}")

    # ── Test 2: Asymmetrische Kopplung (Nicht-Gleichgewicht, ATP-getrieben) ──
    # Cis-Hydrolyse → starker Effekt auf Trans
    # Trans → schwacher Rückeffekt auf Cis
    params_asym = {
        "h_cis":  [0.0, 2.0, 5.0],
        "h_trans": [1.0, 1.5, 3.0],   # Trans hat andere Präferenzen
        "J_ct": [                       # Cis → Trans: STARK
            [3.0, -1.0, -2.0],
            [-1.0, 2.0, 0.0],
            [-2.0, 0.0, 1.0],
        ],
        "J_tc": [                       # Trans → Cis: SCHWACH (nicht-reziprok)
            [1.0, -0.3, -0.5],
            [-0.5, 0.8, 0.0],
            [-1.0, 0.0, 0.3],
        ],
    }

    u1_a, u2_a, _ = compute_utilities_groel(params_asym)
    max_v_a, viol_a = test_potential_game_2player(u1_a, u2_a, profiles)
    nash_a = find_nash_2player(u1_a, u2_a, profiles)

    results["asymmetric"] = {
        "description": "Asymmetric coupling (non-equilibrium, ATP-driven, J_ct != J_tc^T)",
        "is_potential_game": bool(max_v_a < 1e-10),
        "max_violation": float(max_v_a),
        "n_violations": len(viol_a),
        "violations_sample": viol_a[:3],
        "nash_equilibria": nash_a,
    }

    print("\n=== Test 2: Asymmetrische Kopplung (ATP-getrieben) ===")
    print(f"  Potential Game: {'JA' if max_v_a < 1e-10 else 'NEIN'}")
    print(f"  Max Verletzung: {max_v_a:.6f}")
    print(f"  Nash-GG: {len(nash_a)}")
    for n in nash_a:
        print(f"    Cis={n['profile'][0]}, Trans={n['profile'][1]}  "
              f"U={n['u_total']:.1f}")

    # ── Test 3: Nash vs. Soziales Optimum ──
    print("\n=== Test 3: Nash vs. Soziales Optimum (asymmetrisch) ===")
    total_u = u1_a + u2_a
    opt_idx = np.argmax(total_u)
    opt_profile = profiles[opt_idx]
    opt_labels = (STATE_NAMES[opt_profile[0]], STATE_NAMES[opt_profile[1]])
    print(f"  Soziales Optimum: Cis={opt_labels[0]}, Trans={opt_labels[1]}  "
          f"U={total_u[opt_idx]:.1f}")

    nash_profiles = [n["profile"] for n in nash_a]
    nash_is_opt = opt_labels in nash_profiles
    print(f"  Nash enthält Optimum: {nash_is_opt}")

    if not nash_is_opt and nash_a:
        best_nash = max(nash_a, key=lambda n: n["u_total"])
        efficiency = best_nash["u_total"] / total_u[opt_idx]
        print(f"  Bestes Nash-GG: {best_nash['profile']}  U={best_nash['u_total']:.1f}")
        print(f"  Effizienz (Nash/Optimum): {efficiency:.2%}")
        results["efficiency_loss"] = {
            "optimum": float(total_u[opt_idx]),
            "best_nash": float(best_nash["u_total"]),
            "efficiency": float(efficiency),
            "description": "Efficiency loss = gap between Nash and social optimum",
        }

    # ── Test 4: Biologische Interpretation ──
    print("\n=== Test 4: Biologische Interpretation ===")
    print("  Gleichgewicht (sym):  Beide Ringe optimieren gemeinsam → Potential Game")
    print("  Nicht-GG (asym):      ATP-Hydrolyse erzwingt gerichtete Kopplung")
    print("  → Spieltheorie sagt vorher, welche Konformation stabil ist,")
    print("    wenn freie-Energie-Minimierung versagt (kein gemeinsames Potential)")

    # Biologischer Zyklus: T,T → R'',T → R'',R → T,R'' → T,T (vereinfacht)
    print("\n  Beobachteter GroEL-Zyklus:")
    cycle = [("T", "T"), ("R''", "T"), ("R''", "R"), ("T", "R''")]
    for i, (c, t) in enumerate(cycle):
        pi = profiles.index((STATES[c], STATES[t]))
        arrow = " →" if i < len(cycle) - 1 else " → (T,T)"
        print(f"    ({c},{t})  U_cis={u1_a[pi]:.1f}  U_trans={u2_a[pi]:.1f}  "
              f"Total={u1_a[pi]+u2_a[pi]:.1f}{arrow}")

    is_cycle_nash = []
    for c, t in cycle:
        is_n = (c, t) in nash_profiles
        is_cycle_nash.append(is_n)
    print(f"\n  Welche Zyklus-Schritte sind Nash-GG: {is_cycle_nash}")
    print("  (Nicht-Nash-Schritte werden durch ATP-Hydrolyse erzwungen)")

    results["cycle_analysis"] = {
        "observed_cycle": [list(x) for x in cycle],
        "cycle_is_nash": is_cycle_nash,
    }

    # Speichern
    out_path = Path(__file__).parent.parent / "results" / "groel_nonequilibrium_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nErgebnisse gespeichert: {out_path}")


if __name__ == "__main__":
    main()
