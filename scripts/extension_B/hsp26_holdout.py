"""
Hsp26 (S. cerevisiae) — Extension B Hold-Out #2
=================================================
COMMITTED PREDICTION (PREREGISTRATION.md):
  PG = True, NE = 1, Regime = GG (ATP-freie Holdase)

Experimentelle Daten:
  Haslbeck et al. (1999) EMBO J 18:6744
    — Temperaturaktivierung: >38°C → Oligomer-Reorganisation
    — Hsp26 = kleines Hitzeschock-Protein (sHsp), 24-mer
    — Monomer: 23,748 Da
  Franzmann et al. (2005) J Mol Biol 350:1083
    — Oligomer-Dissoziation NICHT nötig für Aktivierung
    — Temperatur ändert Bindungskapazität innerhalb des Oligomers
  Haslbeck et al. (2021) Nat Commun 12:6768
    — Cryo-EM Struktur, Phosphorylierung als zweiter Aktivierungsweg
  Cashikar et al. (2005) Mol Cell 20:739
    — Substrat:Monomer-Stöchiometrie: ~1 Substrat pro Dimer

Spieler:
  1. Hsp26: {Inactive (Oligomer), Active (reorganisiert)}
  2. Substrat: {Unfolded, Folded}

Biologischer Hintergrund:
  Hsp26 ist ein ATP-freies, temperatur-aktiviertes sHsp.
  Im Ruhezustand bildet es 24-mer-Oligomere. Bei Hitzeschock (>38°C)
  reorganisiert das Oligomer und exponiert hydrophobe Bindungsstellen.
  Die Holdase-Funktion verhindert irreversible Aggregation.
  KEIN ATP → R1 Zeile 3 ("kein ATP → PG") greift.
  R4 (sHsp: Holdase, keine Substrat-Stimulation der ATPase) konvergiert.

Kalibrierungsstatus:
  Temperaturaktivierung: KALIBRIERT (Haslbeck 1999)
  Stöchiometrie: KALIBRIERT (Cashikar 2005, ~1/Dimer)
  ΔG_activation: GESCHÄTZT (keine publizierten ΔG-Werte für Oligomer-Switch)
  ΔG_bind(Substrat): GESCHÄTZT aus Holdase-Effizienz
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

T_ACTIVATION = 38.0     # °C (Haslbeck 1999: Übergang >38°C)
T_PHYS_STRESS = 42.0    # °C (typische Hitzeschock-Bedingung)
MONOMER_DA = 23748       # Da (Monomer-Masse)
OLIGOMER_N = 24          # 24-mer (Haslbeck 1999)
SUBSTRATE_PER_DIMER = 1  # Cashikar 2005

RT_42 = RT(T_PHYS_STRESS)

KD_HSP26_SUBSTRATE_EST = 5e-6   # M, geschätzt aus Holdase-Assay-Effizienz
KD_HSP26_INACTIVE = 200e-6      # M, geschätzt (inaktiv bindet kaum)


def dG_from_KD(KD, T_celsius=42):
    return RT(T_celsius) * np.log(KD)


def derived_quantities():
    dG_active = dG_from_KD(KD_HSP26_SUBSTRATE_EST, T_PHYS_STRESS)
    dG_inactive = dG_from_KD(KD_HSP26_INACTIVE, T_PHYS_STRESS)

    return {
        "dG_Hsp26_active_bind_kcal": round(dG_active, 2),
        "dG_Hsp26_inactive_bind_kcal": round(dG_inactive, 2),
        "T_activation_C": T_ACTIVATION,
        "T_stress_C": T_PHYS_STRESS,
        "oligomer_size": OLIGOMER_N,
        "substrate_per_dimer": SUBSTRATE_PER_DIMER,
    }


# ── Payoff-Konstruktion ──────────────────────────────────────────

def build_payoffs(Delta=3.0, dG_activation=-1.0, dG_fold=-5.0):
    """2×2 Spiel: Hsp26 × Substrat (shared-Delta Konvention).

    Shared-Delta: Dieselbe Kopplungsenergie erscheint in beiden
    Payoff-Matrizen. PG folgt algebraisch aus Delta_H = Delta_S.
    sHsp-Holdase mit einer Bindungsfläche → S4-Reziprozität motiviert
    durch Atlas-Vergleich (vgl. hsp70_calibrated.py).

    Hsp26-Strategien: Inactive=0, Active=1
    Substrat-Strategien: Unfolded=0, Folded=1
    """
    # ── Hsp26 Payoffs ──
    u_Hsp26 = np.zeros((2, 2))
    u_Hsp26[0, 0] = 0.0                       # Inactive + Unfolded: Referenz
    u_Hsp26[0, 1] = 0.0                       # Inactive + Folded: kein Kontakt
    u_Hsp26[1, 0] = dG_activation + Delta     # Active + Unfolded: Reorganisation + Kopplung
    u_Hsp26[1, 1] = dG_activation             # Active + Folded: nur Reorganisationskosten

    # ── Substrat Payoffs (shared Delta!) ──
    u_Sub = np.zeros((2, 2))
    u_Sub[0, 0] = 0.0                         # Unfolded + Inactive: Aggregationsrisiko
    u_Sub[0, 1] = Delta                       # Unfolded + Active: Aggregationsschutz
    u_Sub[1, 0] = -dG_fold                    # Folded + Inactive: stabil
    u_Sub[1, 1] = -dG_fold                    # Folded + Active: stabil

    params = {
        "Delta_kcal": Delta,
        "delta_quelle": "ANGENOMMEN (reziproke Kopplung, Atlas-Konvention)",
        "dG_activation_kcal": dG_activation,
        "dG_fold_kcal": dG_fold,
        "T_C": T_PHYS_STRESS,
    }

    return u_Hsp26, u_Sub, params


def robustness_scan():
    results = []
    for Delta in [1.0, 2.0, 3.0, 4.0, 5.0]:
        for dG_fold in [-3.0, -5.0, -7.0, -10.0]:
            u_Hsp26, u_Sub, _ = build_payoffs(Delta=Delta, dG_fold=dG_fold)
            pg = four_cycle_test_2x2(u_Hsp26, u_Sub)
            nash = find_nash_pure(u_Hsp26, u_Sub, 2,
                                  ["Inactive", "Active"],
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
    print("Hsp26 (sHsp) — Extension B Hold-Out #2")
    print("COMMITTED: PG=True, NE=1, Regime=GG")
    print("=" * 65)

    dq = derived_quantities()
    print(f"\n--- Experimentelle Parameter ---")
    print(f"  ΔG(Hsp26:Sub, aktiv):   {dq['dG_Hsp26_active_bind_kcal']:.2f} kcal/mol")
    print(f"  ΔG(Hsp26:Sub, inaktiv): {dq['dG_Hsp26_inactive_bind_kcal']:.2f} kcal/mol")
    print(f"  T_Aktivierung:          {dq['T_activation_C']:.0f}°C")
    print(f"  T_Stress:               {dq['T_stress_C']:.0f}°C")
    print(f"  Oligomer-Größe:         {dq['oligomer_size']}-mer")
    print(f"  Substrat/Dimer:         {dq['substrate_per_dimer']}")
    print(f"  ATP-Verbrauch:          0 (ATP-freies sHsp)")

    u_Hsp26, u_Sub, params = build_payoffs()

    print(f"\n--- Payoff-Matrizen (T={params['T_C']}°C) ---")
    print(f"  u_Hsp26:")
    print(f"              Sub=Unfolded  Sub=Folded")
    print(f"    Inactive   {u_Hsp26[0,0]:8.2f}      {u_Hsp26[0,1]:8.2f}")
    print(f"    Active     {u_Hsp26[1,0]:8.2f}      {u_Hsp26[1,1]:8.2f}")
    print(f"\n  u_Sub (Substrat):")
    print(f"              Hsp26=Inactive  Hsp26=Active")
    print(f"    Unfolded   {u_Sub[0,0]:8.2f}         {u_Sub[0,1]:8.2f}")
    print(f"    Folded     {u_Sub[1,0]:8.2f}         {u_Sub[1,1]:8.2f}")

    pg = four_cycle_test_2x2(u_Hsp26, u_Sub)
    print(f"\n--- Potential-Game-Test ---")
    print(f"  I_Hsp26 = {pg['I_1']:.4f}")
    print(f"  I_Sub   = {pg['I_2']:.4f}")
    print(f"  Verletzung = {pg['violation']:.6f} kcal/mol")
    print(f"  PG = {pg['is_PG']}")

    strategies_H = ["Inactive", "Active"]
    strategies_S = ["Unfolded", "Folded"]
    nash = find_nash_pure(u_Hsp26, u_Sub, 2, strategies_H, strategies_S)
    print(f"\n--- Nash-Gleichgewichte ---")
    print(f"  Anzahl reine NE: {len(nash)}")
    for ne in nash:
        print(f"    Hsp26={ne['player_1']}, Sub={ne['player_2']}  "
              f"u_Hsp26={ne['u_1']:.2f}  u_Sub={ne['u_2']:.2f}  "
              f"Total={ne['u_total']:.2f}")

    if pg["is_PG"]:
        pot = compute_potential_2x2(u_Hsp26, u_Sub)
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
        "experiment": "Hsp26 (sHsp) — Extension B Hold-Out #2",
        "committed_prediction": {
            "PG": committed_PG, "NE": committed_NE, "Regime": "GG",
        },
        "referenzen": {
            "temperaturaktivierung": "Haslbeck et al. (1999) EMBO J 18:6744",
            "oligomer_mechanismus": "Franzmann et al. (2005) J Mol Biol 350:1083",
            "cryo_em": "Haslbeck et al. (2021) Nat Commun 12:6768",
            "stoechiometrie": "Cashikar et al. (2005) Mol Cell 20:739",
        },
        "parameter": {
            "experimentell": {
                "T_activation_C": T_ACTIVATION,
                "oligomer_size": OLIGOMER_N,
                "monomer_Da": MONOMER_DA,
                "substrate_per_dimer": SUBSTRATE_PER_DIMER,
                "ATP_Verbrauch": 0,
                "KD_active_uM": round(KD_HSP26_SUBSTRATE_EST * 1e6, 1),
                "KD_inactive_uM": round(KD_HSP26_INACTIVE * 1e6, 0),
            },
            "modell": {
                "Delta_kcal": params["Delta_kcal"],
                "delta_quelle": params["delta_quelle"],
                "dG_activation_kcal": params["dG_activation_kcal"],
                "dG_fold_kcal": params["dG_fold_kcal"],
            },
            "abgeleitet": dq,
        },
        "payoffs": {
            "u_Hsp26": u_Hsp26.tolist(),
            "u_Sub": u_Sub.tolist(),
            "strategien_Hsp26": strategies_H,
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

    out_path = save_results(results, __file__, "hsp26_holdout_results.json")
    print(f"\n→ Ergebnisse: {out_path}")

    print(f"\n{'='*65}")
    print(f"ERGEBNIS: Hsp26 {'PG' if pg['is_PG'] else 'NICHT-PG'} (Konvention), "
          f"{len(nash)} NE, NE-{'HIT' if hit_NE else 'MISS'}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
