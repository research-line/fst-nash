"""
Hsp33 (E. coli) — Extension B Hold-Out #3
===========================================
COMMITTED PREDICTION (PREREGISTRATION.md):
  PG = True, NE = 1, Regime = GG (ATP-freie, redox-regulierte Holdase)

Experimentelle Daten:
  Graumann et al. (2001) Structure 9:377
    — Zwei-Schritt-Aktivierung: (1) Zn²⁺-Release (4 Cys), (2) 2 Disulfide → Dimerisierung
    — Aktives Dimer: exponiert hydrophobe Fläche (~25% der Gesamtoberfläche)
  Xu, Schmitt, Tang, Jakob & Fitzgerald (2010) Biochemistry 49:1346
    — SUPREX: K_d = 3-300 nM (substratabhängig, 4 Substrate getestet)
    — Citrat-Synthase K_d ≈ 3 nM, Luciferase K_d ≈ 300 nM
  Reichmann et al. (2012) Cell 148:947
    — Arbeitszyklus: Hsp33 schützt → Übergabe an DnaK-System
    — Nicht eigenständig faltend, reine Holdase
  Winter et al. (2008) Mol Cell 29:545
    — Hsp33 vs Hsp70 unter oxidativem Stress: Hsp33 übernimmt wenn DnaK versagt

Spieler:
  1. Hsp33: {Reduced (inaktiv, Zn²⁺-gebunden), Oxidized (aktiv, Dimer)}
  2. Substrat: {Unfolded, Folded}

Biologischer Hintergrund:
  Hsp33 ist ein redox-reguliertes Chaperon das OHNE ATP arbeitet.
  Aktivierung durch oxidativen Stress + Hitze (Zwei-Schritt-Schalter).
  Im reduzierten Zustand: Zn²⁺ koordiniert 4 Cysteines → inaktiv.
  Bei Oxidation: Zn²⁺ freigesetzt, 2 Disulfidbrücken → Dimerisierung → aktive Holdase.
  Redox-Schalter ist ORTHOGONAL zur Substratbindung (R5).
  KEIN ATP → R1 Zeile 3 ("kein ATP → PG") greift.

Kalibrierungsstatus:
  K_d(Hsp33:Substrate): KALIBRIERT (Xu et al. 2010, SUPREX)
  Aktivierungsmechanismus: KALIBRIERT (Graumann 2001)
  ΔG_activation(Redox-Switch): GESCHÄTZT
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

KD_HSP33_CS = 3e-9         # M (Citrat-Synthase, Xu et al. 2010)
KD_HSP33_LUC = 300e-9      # M (Luciferase, Xu et al. 2010)
KD_HSP33_MDH = 30e-9       # M (MDH, Xu et al. 2010, interpoliert)
KD_HSP33_INACTIVE = 500e-6 # M (geschätzt: reduzierte Form bindet kaum)

T_PHYS = 43.0  # °C (Hitzeschock + oxidativer Stress)
RT_43 = RT(T_PHYS)


def dG_from_KD(KD, T_celsius=43):
    return RT(T_celsius) * np.log(KD)


def derived_quantities():
    dG_cs = dG_from_KD(KD_HSP33_CS, T_PHYS)
    dG_luc = dG_from_KD(KD_HSP33_LUC, T_PHYS)
    dG_mdh = dG_from_KD(KD_HSP33_MDH, T_PHYS)
    selectivity = KD_HSP33_LUC / KD_HSP33_CS

    return {
        "dG_Hsp33_CS_kcal": round(dG_cs, 2),
        "dG_Hsp33_Luc_kcal": round(dG_luc, 2),
        "dG_Hsp33_MDH_kcal": round(dG_mdh, 2),
        "selectivity_CS_vs_Luc": round(selectivity, 0),
        "KD_range_nM": "3-300",
    }


# ── Payoff-Konstruktion ──────────────────────────────────────────

def build_payoffs(Delta=3.0, dG_redox_switch=-2.0, dG_fold=-5.0):
    """2×2 Spiel: Hsp33 × Substrat (shared-Delta Konvention).

    Shared-Delta: Dieselbe Kopplungsenergie erscheint in beiden
    Payoff-Matrizen. PG folgt algebraisch aus Delta_H = Delta_S.
    Redox-Holdase mit einer Bindungsfläche → S4-Reziprozität motiviert
    durch Atlas-Vergleich (vgl. hsp70_calibrated.py).

    Hsp33-Strategien: Reduced=0, Oxidized=1
    Substrat-Strategien: Unfolded=0, Folded=1
    """
    # ── Hsp33 Payoffs ──
    u_Hsp33 = np.zeros((2, 2))
    u_Hsp33[0, 0] = 0.0                       # Reduced + Unfolded: Referenz
    u_Hsp33[0, 1] = 0.0                       # Reduced + Folded: kein Kontakt
    u_Hsp33[1, 0] = dG_redox_switch + Delta   # Oxidized + Unfolded: Schalter + Kopplung
    u_Hsp33[1, 1] = dG_redox_switch           # Oxidized + Folded: nur Schalterkosten

    # ── Substrat Payoffs (shared Delta!) ──
    u_Sub = np.zeros((2, 2))
    u_Sub[0, 0] = 0.0                         # Unfolded + Reduced: Aggregationsrisiko
    u_Sub[0, 1] = Delta                       # Unfolded + Oxidized: Aggregationsschutz
    u_Sub[1, 0] = -dG_fold                    # Folded + Reduced: stabil
    u_Sub[1, 1] = -dG_fold                    # Folded + Oxidized: stabil

    params = {
        "Delta_kcal": Delta,
        "delta_quelle": "ANGENOMMEN (reziproke Kopplung, Atlas-Konvention)",
        "dG_redox_switch_kcal": dG_redox_switch,
        "dG_fold_kcal": dG_fold,
        "T_C": T_PHYS,
    }

    return u_Hsp33, u_Sub, params


def robustness_scan():
    results = []
    for Delta in [1.0, 2.0, 3.0, 4.0, 5.0]:
        for dG_fold in [-3.0, -5.0, -7.0, -10.0]:
            u_Hsp33, u_Sub, _ = build_payoffs(Delta=Delta, dG_fold=dG_fold)
            pg = four_cycle_test_2x2(u_Hsp33, u_Sub)
            nash = find_nash_pure(u_Hsp33, u_Sub, 2,
                                  ["Reduced", "Oxidized"],
                                  ["Unfolded", "Folded"])
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
    print("Hsp33 (Redox-Holdase) — Extension B Hold-Out #3")
    print("COMMITTED: PG=True, NE=1, Regime=GG")
    print("=" * 65)

    dq = derived_quantities()
    print(f"\n--- Experimentelle Bindungsenergien (SUPREX) ---")
    print(f"  ΔG(Hsp33:CS, 43°C):     {dq['dG_Hsp33_CS_kcal']:.2f} kcal/mol")
    print(f"  ΔG(Hsp33:Luc, 43°C):    {dq['dG_Hsp33_Luc_kcal']:.2f} kcal/mol")
    print(f"  ΔG(Hsp33:MDH, 43°C):    {dq['dG_Hsp33_MDH_kcal']:.2f} kcal/mol")
    print(f"  K_d-Bereich:             {dq['KD_range_nM']} nM")
    print(f"  Selektivität CS/Luc:     {dq['selectivity_CS_vs_Luc']:.0f}×")
    print(f"  ATP-Verbrauch:           0 (ATP-freie Redox-Holdase)")

    u_Hsp33, u_Sub, params = build_payoffs()

    print(f"\n--- Payoff-Matrizen (T={params['T_C']}°C) ---")
    print(f"  u_Hsp33:")
    print(f"              Sub=Unfolded  Sub=Folded")
    print(f"    Reduced    {u_Hsp33[0,0]:8.2f}      {u_Hsp33[0,1]:8.2f}")
    print(f"    Oxidized   {u_Hsp33[1,0]:8.2f}      {u_Hsp33[1,1]:8.2f}")
    print(f"\n  u_Sub (Substrat):")
    print(f"              Hsp33=Reduced  Hsp33=Oxidized")
    print(f"    Unfolded   {u_Sub[0,0]:8.2f}        {u_Sub[0,1]:8.2f}")
    print(f"    Folded     {u_Sub[1,0]:8.2f}        {u_Sub[1,1]:8.2f}")

    pg = four_cycle_test_2x2(u_Hsp33, u_Sub)
    print(f"\n--- Potential-Game-Test ---")
    print(f"  I_Hsp33 = {pg['I_1']:.4f}")
    print(f"  I_Sub   = {pg['I_2']:.4f}")
    print(f"  Verletzung = {pg['violation']:.6f} kcal/mol")
    print(f"  PG = {pg['is_PG']}")

    strategies_H = ["Reduced", "Oxidized"]
    strategies_S = ["Unfolded", "Folded"]
    nash = find_nash_pure(u_Hsp33, u_Sub, 2, strategies_H, strategies_S)
    print(f"\n--- Nash-Gleichgewichte ---")
    print(f"  Anzahl reine NE: {len(nash)}")
    for ne in nash:
        print(f"    Hsp33={ne['player_1']}, Sub={ne['player_2']}  "
              f"u_Hsp33={ne['u_1']:.2f}  u_Sub={ne['u_2']:.2f}  "
              f"Total={ne['u_total']:.2f}")

    if pg["is_PG"]:
        pot = compute_potential_2x2(u_Hsp33, u_Sub)
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
        "experiment": "Hsp33 (Redox-Holdase) — Extension B Hold-Out #3",
        "committed_prediction": {
            "PG": committed_PG, "NE": committed_NE, "Regime": "GG",
        },
        "referenzen": {
            "zwei_schritt_aktivierung": "Graumann et al. (2001) Structure 9:377",
            "SUPREX_KD": "Xu et al. (2010) Biochemistry 49:1346",
            "arbeitszyklus": "Reichmann et al. (2012) Cell 148:947",
            "oxidativer_stress": "Winter et al. (2008) Mol Cell 29:545",
        },
        "parameter": {
            "experimentell": {
                "KD_CS_nM": round(KD_HSP33_CS * 1e9, 0),
                "KD_Luc_nM": round(KD_HSP33_LUC * 1e9, 0),
                "KD_MDH_nM": round(KD_HSP33_MDH * 1e9, 0),
                "KD_inactive_uM": round(KD_HSP33_INACTIVE * 1e6, 0),
                "ATP_Verbrauch": 0,
            },
            "modell": {
                "Delta_kcal": params["Delta_kcal"],
                "delta_quelle": params["delta_quelle"],
                "dG_redox_switch_kcal": params["dG_redox_switch_kcal"],
                "dG_fold_kcal": params["dG_fold_kcal"],
            },
            "abgeleitet": dq,
        },
        "payoffs": {
            "u_Hsp33": u_Hsp33.tolist(),
            "u_Sub": u_Sub.tolist(),
            "strategien_Hsp33": strategies_H,
            "strategien_Sub": strategies_S,
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

    out_path = save_results(results, __file__, "hsp33_holdout_results.json")
    print(f"\n→ Ergebnisse: {out_path}")

    print(f"\n{'='*65}")
    print(f"ERGEBNIS: Hsp33 {'PG' if pg['is_PG'] else 'NICHT-PG'} (Konvention), "
          f"{len(nash)} NE, NE-{'HIT' if hit_NE else 'MISS'}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
