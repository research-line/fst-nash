"""
DnaJ allein (E. coli) — Extension B Hold-Out #1
=================================================
COMMITTED PREDICTION (PREREGISTRATION.md):
  PG = True, NE = 1, Regime = GG (ATP-freie Holdase)

Experimentelle Daten:
  Rüdiger, Schneider-Mergener & Bukau (2001) EMBO J 20:1042
    — DnaJ-Substratspezifität: hydrophobe Motive (aromatisch + groß aliphatisch + Arg)
    — Unabhängige Aggregations-Suppression (Holdase-Aktivität ohne Hsp70)
  Kampinga & Craig (2010) Nat Rev Mol Cell Biol 11:579
    — J-Domänen-Klassifikation, DnaJ = Typ I (DNAJA)
  Suh et al. (1998) PNAS 95:15223
    — K_d(DnaJ:σ32) ≈ 20 nM, K_d(DnaJ:Peptid) ≈ 2 µM
  Liberek et al. (1991) PNAS 88:2874
    — DnaJ hat KEINE eigene ATPase (braucht DnaK für ATP-Hydrolyse)

Spieler:
  1. DnaJ: {Free, Substrate-bound}
  2. Substrat: {Unfolded, Folded}

Biologischer Hintergrund:
  DnaJ ALLEIN (ohne Hsp70/DnaK) ist eine ATP-freie Holdase.
  Es bindet entfaltete Proteine über sein CTD (C-Terminal Domain)
  und verhindert Aggregation. KEIN ATP-Verbrauch →
  R1 Zeile 3 ("kein ATP → PG") greift.
  R2 ("breite Substratspezifität → PG") konvergiert.

Kalibrierungsstatus:
  K_d(DnaJ:σ32): KALIBRIERT (Suh 1998, 20 nM)
  K_d(DnaJ:Peptid): KALIBRIERT (Suh 1998, 2 µM)
  ΔG_conf(DnaJ): GESCHÄTZT (keine publizierten ΔG für Free↔Bound)
  ΔG_fold(Substrat): TYPISCH (-5 kcal/mol)
"""

import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from fold_switch_template import (
    four_cycle_test_2x2, find_nash_pure, compute_potential_2x2,
    symmetry_breaking_analysis, save_results, RT
)


# ── Experimentelle Parameter ──────────────────────────────────────

KD_DNAJ_SIGMA32 = 20e-9       # M (Suh et al. 1998, σ32)
KD_DNAJ_PEPTID = 2e-6         # M (Suh et al. 1998, freies Peptid)
KD_DNAJ_LUCIFERASE = 1.5e-6   # M (geschätzt aus Holdase-Assays)

T_PHYS = 37  # °C (physiologische Temperatur)
RT_37 = RT(T_PHYS)


def dG_from_KD(KD, T_celsius=37):
    return RT(T_celsius) * np.log(KD)


def derived_quantities():
    dG_sigma32 = dG_from_KD(KD_DNAJ_SIGMA32, T_PHYS)
    dG_peptid = dG_from_KD(KD_DNAJ_PEPTID, T_PHYS)
    selectivity = KD_DNAJ_PEPTID / KD_DNAJ_SIGMA32

    return {
        "dG_DnaJ_sigma32_kcal": round(dG_sigma32, 2),
        "dG_DnaJ_peptid_kcal": round(dG_peptid, 2),
        "selectivity_sigma32_vs_peptid": round(selectivity, 1),
    }


# ── Payoff-Konstruktion ──────────────────────────────────────────

def build_payoffs(Delta=3.0, dG_conf=-0.5, dG_fold=-5.0):
    """2×2 Spiel: DnaJ × Substrat (shared-Delta Konvention).

    Shared-Delta: Dieselbe Kopplungsenergie erscheint in beiden
    Payoff-Matrizen. PG folgt algebraisch aus Delta_H = Delta_S.
    Holdase mit einer Bindungsfläche → S4-Reziprozität motiviert
    durch Atlas-Vergleich (vgl. hsp70_calibrated.py).

    DnaJ-Strategien: Free=0, Bound=1
    Substrat-Strategien: Unfolded=0, Folded=1
    """
    # ── DnaJ Payoffs ──
    u_DnaJ = np.zeros((2, 2))
    u_DnaJ[0, 0] = 0.0                    # Free + Unfolded: Referenz
    u_DnaJ[0, 1] = 0.0                    # Free + Folded: kein Kontakt
    u_DnaJ[1, 0] = dG_conf + Delta        # Bound + Unfolded: Konf.kosten + Kopplung
    u_DnaJ[1, 1] = dG_conf                # Bound + Folded: nur Konf.kosten

    # ── Substrat Payoffs (shared Delta!) ──
    u_Sub = np.zeros((2, 2))
    u_Sub[0, 0] = 0.0                     # Unfolded + Free: Aggregationsrisiko
    u_Sub[0, 1] = Delta                   # Unfolded + Bound: Aggregationsschutz
    u_Sub[1, 0] = -dG_fold                # Folded + Free: stabil
    u_Sub[1, 1] = -dG_fold                # Folded + Bound: stabil

    params = {
        "Delta_kcal": Delta,
        "delta_quelle": "ANGENOMMEN (reziproke Kopplung, Atlas-Konvention)",
        "dG_conf_kcal": dG_conf,
        "dG_fold_kcal": dG_fold,
        "T_C": T_PHYS,
    }

    return u_DnaJ, u_Sub, params


def robustness_scan():
    results = []
    for Delta in [1.0, 2.0, 3.0, 4.0, 5.0]:
        for dG_fold in [-3.0, -5.0, -7.0, -10.0]:
            u_DnaJ, u_Sub, _ = build_payoffs(Delta=Delta, dG_fold=dG_fold)
            pg = four_cycle_test_2x2(u_DnaJ, u_Sub)
            nash = find_nash_pure(u_DnaJ, u_Sub, 2,
                                  ["Free", "Bound"], ["Unfolded", "Folded"])
            results.append({
                "Delta": Delta,
                "dG_fold": dG_fold,
                "is_PG": pg["is_PG"],
                "violation": pg["violation"],
                "n_NE": len(nash),
                "NE_profiles": [(n["player_1"], n["player_2"]) for n in nash],
            })
    return results


def main():
    print("=" * 65)
    print("DnaJ allein — Extension B Hold-Out #1")
    print("COMMITTED: PG=True, NE=1, Regime=GG")
    print("=" * 65)

    dq = derived_quantities()
    print(f"\n--- Experimentelle Bindungsenergien ---")
    print(f"  ΔG(DnaJ:σ32, 37°C):    {dq['dG_DnaJ_sigma32_kcal']:.2f} kcal/mol")
    print(f"  ΔG(DnaJ:Peptid, 37°C):  {dq['dG_DnaJ_peptid_kcal']:.2f} kcal/mol")
    print(f"  Selektivität σ32/Peptid: {dq['selectivity_sigma32_vs_peptid']:.1f}×")
    print(f"  ATP-Verbrauch:           0 (ATP-freie Holdase)")

    u_DnaJ, u_Sub, params = build_payoffs()

    print(f"\n--- Payoff-Matrizen (T={params['T_C']}°C) ---")
    print(f"  u_DnaJ:")
    print(f"              Sub=Unfolded  Sub=Folded")
    print(f"    Free       {u_DnaJ[0,0]:8.2f}      {u_DnaJ[0,1]:8.2f}")
    print(f"    Bound      {u_DnaJ[1,0]:8.2f}      {u_DnaJ[1,1]:8.2f}")
    print(f"\n  u_Sub (Substrat):")
    print(f"              DnaJ=Free    DnaJ=Bound")
    print(f"    Unfolded   {u_Sub[0,0]:8.2f}      {u_Sub[0,1]:8.2f}")
    print(f"    Folded     {u_Sub[1,0]:8.2f}      {u_Sub[1,1]:8.2f}")

    pg = four_cycle_test_2x2(u_DnaJ, u_Sub)
    print(f"\n--- Potential-Game-Test ---")
    print(f"  I_DnaJ = {pg['I_1']:.4f}")
    print(f"  I_Sub  = {pg['I_2']:.4f}")
    print(f"  Verletzung = {pg['violation']:.6f} kcal/mol")
    print(f"  PG = {pg['is_PG']}")

    strategies_DnaJ = ["Free", "Bound"]
    strategies_Sub = ["Unfolded", "Folded"]
    nash = find_nash_pure(u_DnaJ, u_Sub, 2, strategies_DnaJ, strategies_Sub)
    print(f"\n--- Nash-Gleichgewichte ---")
    print(f"  Anzahl reine NE: {len(nash)}")
    for ne in nash:
        print(f"    DnaJ={ne['player_1']}, Sub={ne['player_2']}  "
              f"u_DnaJ={ne['u_1']:.2f}  u_Sub={ne['u_2']:.2f}  "
              f"Total={ne['u_total']:.2f}")

    if pg["is_PG"]:
        pot = compute_potential_2x2(u_DnaJ, u_Sub)
        print(f"\n--- Potential-Funktion ---")
        for k, v in pot["values"].items():
            print(f"    Φ{k} = {v:.4f}")
        print(f"    Konsistenz: {pot['consistency']:.2e}")

    print(f"\n--- Robustheitsscan ---")
    rob = robustness_scan()
    print(f"  {'Delta':>8} {'dG_fold':>8} {'PG?':>5} {'Verl.':>10} {'NE':>4} Profile")
    for r in rob:
        profiles = ", ".join(f"({a},{b})" for a, b in r["NE_profiles"])
        print(f"  {r['Delta']:>8.1f} {r['dG_fold']:>8.1f} "
              f"{'Y' if r['is_PG'] else 'N':>5} {r['violation']:>10.6f} "
              f"{r['n_NE']:>4} {profiles}")

    # ── Vorhersage-Vergleich ──
    committed_PG = True
    committed_NE = 1
    pg_konsistent = pg["is_PG"] == committed_PG
    hit_NE = len(nash) == committed_NE

    print(f"\n--- HOLD-OUT VORHERSAGE-VERGLEICH ---")
    print(f"  Committed: PG={committed_PG}, NE={committed_NE}")
    print(f"  Computed:  PG={pg['is_PG']}, NE={len(nash)}")
    print(f"  PG-Konvention: {'konsistent' if pg_konsistent else 'INKONSISTENT'}")
    print(f"  NE-Hit:        {'✓' if hit_NE else '✗'}")

    results = {
        "experiment": "DnaJ allein — Extension B Hold-Out #1",
        "committed_prediction": {
            "PG": committed_PG, "NE": committed_NE, "Regime": "GG",
        },
        "referenzen": {
            "substratspezifitaet": "Rüdiger et al. (2001) EMBO J 20:1042",
            "KD_sigma32": "Suh et al. (1998) PNAS 95:15223",
            "J_domain_review": "Kampinga & Craig (2010) Nat Rev Mol Cell Biol 11:579",
            "kein_ATP": "Liberek et al. (1991) PNAS 88:2874",
        },
        "parameter": {
            "experimentell": {
                "KD_sigma32_nM": round(KD_DNAJ_SIGMA32 * 1e9, 0),
                "KD_peptid_uM": round(KD_DNAJ_PEPTID * 1e6, 1),
                "ATP_Verbrauch": 0,
            },
            "modell": {
                "Delta_kcal": params["Delta_kcal"],
                "delta_quelle": params["delta_quelle"],
                "dG_conf_kcal": params["dG_conf_kcal"],
                "dG_fold_kcal": params["dG_fold_kcal"],
            },
            "abgeleitet": dq,
        },
        "payoffs": {
            "u_DnaJ": u_DnaJ.tolist(),
            "u_Sub": u_Sub.tolist(),
            "strategien_DnaJ": strategies_DnaJ,
            "strategien_Sub": strategies_Sub,
        },
        "pg_test": pg,
        "nash_gleichgewichte": nash,
        "robustheit": rob,
        "vorhersage_vergleich": {
            "PG_konvention_konsistent": pg_konsistent,
            "NE_hit": hit_NE,
            "overall_hit": pg_konsistent and hit_NE,
        },
    }

    out_path = save_results(results, __file__, "dnaj_holdout_results.json")
    print(f"\n→ Ergebnisse: {out_path}")

    print(f"\n{'='*65}")
    print(f"ERGEBNIS: DnaJ {'PG' if pg['is_PG'] else 'NICHT-PG'} (Konvention), "
          f"{len(nash)} NE, NE-{'HIT' if hit_NE else 'MISS'}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
