"""
GroEL/GroES — Kalibriertes Nicht-Gleichgewichts-Spiel
=====================================================
Parameter aus experimentellen Daten:

Allosterische Konstanten (Yifrach & Horovitz, Biochemistry 1995):
- L2  = 2e-9  (trans-Ring T→R ohne GroES) → ΔG = 12.3 kcal/mol
- L2' = 4e-5  (trans-Ring T→R mit GroES auf cis) → ΔG = 6.2 kcal/mol
- GroES-Stabilisierung: 6.1 kcal/mol

ATP-Bindung (Yifrach & Horovitz 1995):
- Erste Transition: ~16 μM (intra-ring, positiv kooperativ, n≈2.8)
- Zweite Transition: ~160 μM (inter-ring, negativ kooperativ)

Zykluszeit (Todd et al., PNAS 2010):
- R'-Zustand: ~8-10 s (längste Phase)
- Gesamtzyklus: ~15-20 s

ATP-Hydrolyse-Energie: ~7.3 kcal/mol unter zellulären Bedingungen

Modell:
- 2 Spieler = 2 Ringe (cis mit GroES, trans ohne)
- 3 Strategien: T, R, R'' (R''= R + GroES-capped, Faltungskammer aktiv)
- Asymmetrische Kopplung aus GroES-vermittelter Signalisierung
"""

import numpy as np
from itertools import product
import json
from pathlib import Path

RT = 0.616  # kcal/mol bei 37°C (310K)
STATE_NAMES = ["T", "R", "R''"]


def dG_from_L(L):
    """Freie Energie aus allosterischer Konstante: ΔG = -RT·ln(L)."""
    return -RT * np.log(L)


def build_calibrated_payoffs():
    """Baue Payoff-Matrizen aus experimentellen Daten.

    Konvention: Höherer Payoff = stabiler (negativere freie Energie).
    Payoff = -ΔG (Stabilität, nicht Energie).
    """
    # ── Experimentelle allosterische Konstanten ──
    L2 = 2e-9    # Trans T→R ohne GroES
    L2_prime = 4e-5  # Trans T→R mit GroES auf Cis

    dG_TR_alone = dG_from_L(L2)       # +12.3 kcal/mol (T→R sehr ungünstig)
    dG_TR_groes = dG_from_L(L2_prime)  # +6.2 kcal/mol (weniger ungünstig)
    dG_ATP = -7.3  # ATP-Hydrolyse unter zellulären Bedingungen

    print(f"Experimentelle Parameter:")
    print(f"  ΔG(T→R, allein):    {dG_TR_alone:.1f} kcal/mol")
    print(f"  ΔG(T→R, mit GroES): {dG_TR_groes:.1f} kcal/mol")
    print(f"  GroES-Effekt:       {dG_TR_alone - dG_TR_groes:.1f} kcal/mol Stabilisierung")
    print(f"  ΔG(ATP-Hydrolyse):  {dG_ATP:.1f} kcal/mol")

    # ── Payoff-Matrizen (3×3: eigene Strategie × Partner-Strategie) ──
    # u[s_own, s_partner] = -ΔG (Stabilität relativ zu TT-Referenz)

    # Cis-Ring (hat GroES gebunden in R''-Zustand)
    # Intrinsische Stabilität + Inter-Ring-Kopplung
    u_cis = np.zeros((3, 3))

    # T-Zustand des Cis: Substrat-Bindung, stabil wenn trans auch T
    u_cis[0, 0] = 0.0    # TT: Referenz
    u_cis[0, 1] = -1.5   # T,R: inter-ring Spannung (trans in R destabilisiert cis-T leicht)
    u_cis[0, 2] = -2.0   # T,R'': noch mehr Spannung

    # R-Zustand des Cis: ATP gebunden, höhere Energie
    u_cis[1, 0] = -dG_TR_alone + dG_ATP  # R allein: -12.3 + (-7.3) = Energie vom ATP
    u_cis[1, 1] = -dG_TR_groes + dG_ATP  # R mit trans-R: GroES-ähnliche Stabilisierung
    u_cis[1, 2] = -dG_TR_groes + dG_ATP - 1.0  # R mit trans-R'': Kompetition um GroES

    # R''-Zustand des Cis: ATP + GroES, Faltungskammer aktiv
    # Am stabilsten wenn trans in T (funktionelle Asymmetrie des Zyklus)
    u_cis[2, 0] = -dG_TR_groes + dG_ATP + 3.0  # R'' mit trans-T: optimaler Funktionszustand
    u_cis[2, 1] = -dG_TR_groes + dG_ATP + 1.0  # R'' mit trans-R: weniger optimal
    u_cis[2, 2] = -dG_TR_groes + dG_ATP - 2.0  # R'' mit trans-R'': Blockade (beide wollen GroES)

    # Trans-Ring (KEIN GroES, empfängt Signal von Cis-Hydrolyse)
    u_trans = np.zeros((3, 3))

    # T-Zustand des Trans: Substrat-Bindung, bevorzugt wenn Cis R'' hat
    u_trans[0, 0] = 0.0    # TT: Referenz
    u_trans[0, 1] = 0.5    # T, cis-R: leicht stabilisiert (cis beginnt Zyklus)
    u_trans[0, 2] = 2.0    # T, cis-R'': stark stabilisiert (trans fängt Substrat ab)

    # R-Zustand des Trans: ATP gebunden, SEHR ungünstig ohne GroES
    u_trans[1, 0] = -dG_TR_alone  # R allein: -12.3 (extrem ungünstig!)
    u_trans[1, 1] = -dG_TR_alone + 2.0  # R mit cis-R: leicht besser
    u_trans[1, 2] = -dG_TR_groes  # R mit cis-R'': GroES-Signal (-6.2, deutlich besser)

    # R''-Zustand des Trans: Nur möglich nach GroES-Transfer von Cis
    u_trans[2, 0] = -dG_TR_alone - 3.0  # R'' ohne cis-Unterstützung: unmöglich
    u_trans[2, 1] = -dG_TR_groes + dG_ATP + 1.0  # R'' mit cis-R: teilweise möglich
    u_trans[2, 2] = -dG_TR_groes + dG_ATP - 5.0  # R'' mit cis-R'': Blockade

    return u_cis, u_trans


def test_potential_game(u1, u2, n=3):
    """Vier-Zyklen-Test für 2-Spieler, n-Strategien-Spiel."""
    profiles = list(product(range(n), repeat=2))
    max_v = 0.0
    violations = []

    for s1 in range(n):
        for s1p in range(n):
            if s1p == s1:
                continue
            for s2 in range(n):
                for s2p in range(n):
                    if s2p == s2:
                        continue
                    lhs = (u1[s1, s2] - u1[s1p, s2]) - (u1[s1, s2p] - u1[s1p, s2p])
                    rhs = (u2[s2, s1] - u2[s2, s1p]) - (u2[s2p, s1] - u2[s2p, s1p])
                    diff = abs(lhs - rhs)
                    if diff > 1e-10:
                        violations.append({
                            "s1": STATE_NAMES[s1], "s1'": STATE_NAMES[s1p],
                            "s2": STATE_NAMES[s2], "s2'": STATE_NAMES[s2p],
                            "violation": round(float(diff), 4),
                        })
                        max_v = max(max_v, diff)

    return max_v, violations


def find_nash(u1, u2, n=3):
    """Finde reine Nash-GG."""
    nash = []
    for s1 in range(n):
        for s2 in range(n):
            is_nash = True
            for s1_alt in range(n):
                if u1[s1_alt, s2] > u1[s1, s2] + 1e-10:
                    is_nash = False
                    break
            if not is_nash:
                continue
            for s2_alt in range(n):
                if u2[s2_alt, s1] > u2[s2, s1] + 1e-10:
                    is_nash = False
                    break
            if is_nash:
                nash.append({
                    "cis": STATE_NAMES[s1],
                    "trans": STATE_NAMES[s2],
                    "u_cis": round(float(u1[s1, s2]), 2),
                    "u_trans": round(float(u2[s2, s1]), 2),
                    "u_total": round(float(u1[s1, s2] + u2[s2, s1]), 2),
                })
    return nash


def analyze_cycle(u_cis, u_trans):
    """Analysiere den biologischen GroEL-Konformationszyklus."""
    # Beobachteter Zyklus (vereinfacht):
    # Phase 1: (T,T) — beide Ringe leer, warten auf Substrat
    # Phase 2: (R'',T) — Cis bindet ATP+GroES+Substrat, faltet
    # Phase 3: (R'',R) — Trans beginnt ATP zu binden (Hydrolyse-Signal)
    # Phase 4: (T,R'') — Cis gibt Substrat frei, trans übernimmt
    cycle = [(0, 0), (2, 0), (2, 1), (0, 2)]
    cycle_names = [("T", "T"), ("R''", "T"), ("R''", "R"), ("T", "R''")]

    print("\n  Biologischer Konformationszyklus:")
    print(f"  {'Phase':<8} {'Cis':<5} {'Trans':<5} {'u_cis':>8} {'u_trans':>8} {'Total':>8} {'Nash?':>6}")
    print("  " + "-" * 52)

    nash_list = find_nash(u_cis, u_trans)
    nash_profiles = [(n["cis"], n["trans"]) for n in nash_list]

    cycle_data = []
    for i, ((sc, st), (nc, nt)) in enumerate(zip(cycle, cycle_names)):
        uc = u_cis[sc, st]
        ut = u_trans[st, sc]
        is_nash = (nc, nt) in nash_profiles
        phase = f"Phase {i+1}"
        marker = "✓" if is_nash else "✗"
        print(f"  {phase:<8} {nc:<5} {nt:<5} {uc:>8.2f} {ut:>8.2f} {uc+ut:>8.2f} {marker:>6}")

        cycle_data.append({
            "phase": i + 1,
            "cis": nc, "trans": nt,
            "u_cis": round(float(uc), 2),
            "u_trans": round(float(ut), 2),
            "is_nash": is_nash,
        })

    # Energiedifferenzen zwischen Phasen
    print("\n  Übergänge im Zyklus:")
    for i in range(len(cycle)):
        j = (i + 1) % len(cycle)
        sc_i, st_i = cycle[i]
        sc_j, st_j = cycle[j]
        du_cis = u_cis[sc_j, st_j] - u_cis[sc_i, st_i]
        du_trans = u_trans[st_j, sc_j] - u_trans[st_i, sc_i]
        du_total = du_cis + du_trans
        driver = "spontan" if du_total > 0 else "ATP-getrieben"
        nc_i, nt_i = cycle_names[i]
        nc_j, nt_j = cycle_names[j]
        print(f"  ({nc_i},{nt_i})→({nc_j},{nt_j}): "
              f"Δu_total={du_total:+.2f} kcal/mol [{driver}]")

    return cycle_data


def main():
    print("=" * 60)
    print("GroEL/GroES — Kalibriertes Nicht-Gleichgewichts-Spiel")
    print("=" * 60)

    u_cis, u_trans = build_calibrated_payoffs()

    # Payoff-Matrizen anzeigen
    print("\nPayoff-Matrix Cis-Ring (u_cis[s_cis, s_trans]):")
    print(f"  {'':>8} {'Trans=T':>10} {'Trans=R':>10} {'Trans=R\"':>10}")
    for i, name in enumerate(STATE_NAMES):
        vals = "  ".join(f"{u_cis[i,j]:>8.2f}" for j in range(3))
        print(f"  {f'Cis={name}':<8} {vals}")

    print("\nPayoff-Matrix Trans-Ring (u_trans[s_trans, s_cis]):")
    print(f"  {'':>8} {'Cis=T':>10} {'Cis=R':>10} {'Cis=R\"':>10}")
    for i, name in enumerate(STATE_NAMES):
        vals = "  ".join(f"{u_trans[i,j]:>8.2f}" for j in range(3))
        print(f"  {f'Trans={name}':<8} {vals}")

    # Potential-Game-Test
    print("\n" + "=" * 60)
    print("Potential-Game-Test (Monderer & Shapley 1996)")
    print("=" * 60)

    max_v, violations = test_potential_game(u_cis, u_trans)
    print(f"  Max Zykelverletzung: {max_v:.4f} kcal/mol")
    print(f"  Potential Game: {'JA' if max_v < 1e-10 else 'NEIN'}")
    print(f"  Anzahl Verletzungen: {len(violations)} von 36 möglichen Zyklen")
    if violations:
        print(f"  Stärkste Verletzung: {violations[0]}")

    # Nash-Gleichgewichte
    print("\n" + "=" * 60)
    print("Nash-Gleichgewichte")
    print("=" * 60)

    nash = find_nash(u_cis, u_trans)
    print(f"  Anzahl reine Nash-GG: {len(nash)}")
    for n in nash:
        print(f"    Cis={n['cis']}, Trans={n['trans']}  "
              f"u_cis={n['u_cis']:.2f}  u_trans={n['u_trans']:.2f}  "
              f"Total={n['u_total']:.2f}")

    # Soziales Optimum
    best_total = -np.inf
    best_profile = None
    for s1 in range(3):
        for s2 in range(3):
            total = u_cis[s1, s2] + u_trans[s2, s1]
            if total > best_total:
                best_total = total
                best_profile = (STATE_NAMES[s1], STATE_NAMES[s2])

    print(f"\n  Soziales Optimum: Cis={best_profile[0]}, Trans={best_profile[1]}  "
          f"Total={best_total:.2f}")
    nash_is_opt = any(
        (n["cis"], n["trans"]) == best_profile for n in nash
    )
    print(f"  Nash = Optimum: {nash_is_opt}")

    # Zyklusanalyse
    print("\n" + "=" * 60)
    print("Biologischer Konformationszyklus")
    print("=" * 60)
    cycle_data = analyze_cycle(u_cis, u_trans)

    # Interpretation
    nash_in_cycle = sum(1 for c in cycle_data if c["is_nash"])
    atp_driven = sum(1 for c in cycle_data if not c["is_nash"])

    print(f"\n  Zusammenfassung:")
    print(f"    Nash-stabile Phasen:  {nash_in_cycle}/4")
    print(f"    ATP-erzwungene Phasen: {atp_driven}/4")
    print(f"    → Spieltheorie identifiziert, welche Übergänge")
    print(f"      energetisch getrieben werden müssen.")

    # Ergebnisse speichern
    results = {
        "experimental_parameters": {
            "L2": 2e-9,
            "L2_prime": 4e-5,
            "dG_TR_alone_kcal": round(float(dG_from_L(2e-9)), 2),
            "dG_TR_groes_kcal": round(float(dG_from_L(4e-5)), 2),
            "dG_ATP_kcal": -7.3,
            "RT_kcal": 0.616,
        },
        "potential_game_test": {
            "is_potential_game": bool(max_v < 1e-10),
            "max_violation_kcal": round(float(max_v), 4),
            "n_violations": len(violations),
        },
        "nash_equilibria": nash,
        "social_optimum": {
            "cis": best_profile[0],
            "trans": best_profile[1],
            "total": round(float(best_total), 2),
            "nash_is_optimum": bool(nash_is_opt),
        },
        "cycle_analysis": {
            "phases": cycle_data,
            "nash_stable_phases": nash_in_cycle,
            "atp_driven_phases": atp_driven,
        },
    }

    out_path = Path(__file__).parent.parent / "results" / "groel_calibrated_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nErgebnisse: {out_path}")


if __name__ == "__main__":
    main()
