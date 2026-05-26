"""
Trigger Factor (TF, E. coli) — ATP-freier Chaperon-Kontrollfall
================================================================
Experimentelle Daten:
  Tetter et al. (2024) Sci Adv — K_D, Thermodynamik (DOI:10.1126/sciadv.adn4824)
  Patzelt et al. (2002) Biol Chem — K_D(TF:Ribosom) = 1.2 µM
  Naqvi et al. (2022) Nat Commun — K_D(TF:GAPDH) = 0.32 µM, Holdase+Foldase
  Haldar et al. (2017) Nat Commun — Mechanische Foldase unter Kraft
  Kaiser et al. (2006) Nature — Residenzzeit ~10 s, offene Konformation
  Deuerling et al. (1999) Nature — TF+DnaK synthetisch letal >30°C

Spieler:
  1. TF: {Compact (frei/Dimer), Open (ribosomengebunden)}
  2. Substrat: {Unfolded (naszierend), Folded (nativ)}

Biologischer Hintergrund:
  TF ist ein ribosomen-assoziiertes Chaperon das OHNE ATP arbeitet.
  Es ist der erste Chaperon-Kontakt für naszierende Polypeptide.
  KEIN ATP → Gleichgewichtssystem? Oder passiv-mechanische Asymmetrie?

  Kritischer Kontrolltest (Q10 in QUESTIONS.md):
  Wenn NE=1 → ATP ist NICHT die Ursache der NE-Eindeutigkeit
  Wenn NE≥2 → ATP-Hypothese gestärkt

Kalibrierungsstatus:
  K_D(TF:Ribosom): KALIBRIERT (Patzelt 2002, Tetter 2024)
  K_D(TF:Substrat): KALIBRIERT (Naqvi 2022, Tetter 2024)
  ΔH/ΔS(TF:70S): KALIBRIERT (Tetter 2024)
  Konformationsenergie TF: GESCHÄTZT (keine ΔG-Werte publiziert)
  Faltungsenergie Substrat: TYPISCH (-5 kcal/mol)
"""

import numpy as np
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from fold_switch_template import (
    four_cycle_test_2x2, find_nash_pure, compute_potential_2x2,
    symmetry_breaking_analysis, save_results, RT
)


# ── Experimentelle Parameter ──────────────────────────────────────

# Tetter et al. (2024) Sci Adv — Thermodynamik
KD_TF_RIBOSOM_22C = 2.71e-6   # M (leeres 70S, 22°C)
KD_TF_RIBOSOM_37C = 11.1e-6   # M (leeres 70S, 37°C)
KD_TF_RNC_22C = 482e-9        # M (α-Synuclein RNC, 22°C)
DH_TF_70S = -16.7              # kcal/mol (= -69.8 kJ/mol, enthalpie-getrieben)
DS_TF_70S = -0.0316            # kcal/(mol·K) (= -132 J/(mol·K))
DDG_RNC_VS_70S = -1.1          # kcal/mol (= -4 bis -5 kJ/mol, entropisch)

# Naqvi et al. (2022) Nat Commun
KD_TF_GAPDH = 0.32e-6         # M (entfaltetes GAPDH, 4°C)
KD_TF_LUCIFERASE = 4.5e-6     # M (entfaltete Luciferase, 25°C)
KON_TF_GAPDH = 6.6e6          # M⁻¹·s⁻¹
KOFF_TF_GAPDH = 8.7            # s⁻¹

# Patzelt et al. (2001, 2002)
KD_TF_PEPTID = 120e-6         # M (freies Peptid)

RT_37C = RT(37)  # 0.616 kcal/mol


def dG_from_KD(KD, T_celsius=37):
    """ΔG_bind = RT·ln(K_D)."""
    return RT(T_celsius) * np.log(KD)


def derived_quantities():
    """Abgeleitete thermodynamische Größen."""
    dG_rib_22 = dG_from_KD(KD_TF_RIBOSOM_22C, 22)
    dG_rib_37 = dG_from_KD(KD_TF_RIBOSOM_37C, 37)
    dG_rnc_22 = dG_from_KD(KD_TF_RNC_22C, 22)
    dG_gapdh = dG_from_KD(KD_TF_GAPDH, 4)
    dG_peptid = dG_from_KD(KD_TF_PEPTID, 37)

    selectivity = KD_TF_RIBOSOM_22C / KD_TF_RNC_22C

    return {
        "dG_TF_70S_22C_kcal": round(dG_rib_22, 2),
        "dG_TF_70S_37C_kcal": round(dG_rib_37, 2),
        "dG_TF_RNC_22C_kcal": round(dG_rnc_22, 2),
        "dG_TF_GAPDH_4C_kcal": round(dG_gapdh, 2),
        "dG_TF_peptid_37C_kcal": round(dG_peptid, 2),
        "selectivity_RNC_vs_70S": round(selectivity, 1),
        "ddG_RNC_vs_70S_kcal": round(DDG_RNC_VS_70S, 2),
    }


# ── Payoff-Konstruktion ──────────────────────────────────────────

def build_payoffs(dG_open=-1.5, dG_fold=-5.0, dG_bind_open_unfolded=-8.6,
                  dG_bind_compact_unfolded=-5.2, T=37):
    """2×2 Spiel: TF × Substrat.

    Parameter:
      dG_open: Energiekosten für TF Compact→Open (kcal/mol).
        Geschätzt: Ribosom-Bindung stabilisiert offene Form.
        Kein direkter ΔG publiziert.
      dG_fold: Faltungsenergie Substrat (kcal/mol, typisch -5).
      dG_bind_open_unfolded: Bindungsenergie TF(Open)+Substrat(Unfolded).
        Abgeleitet aus K_D(RNC) ≈ 480 nM bei 22°C → ΔG ≈ -8.6 kcal/mol.
      dG_bind_compact_unfolded: Bindungsenergie TF(Compact)+Substrat(Unfolded).
        Schwächer — TF im Kompakt-Zustand hat reduzierten Substrat-Zugang.
        Abgeleitet aus K_D(freies Protein) ≈ 0.32-4.5 µM → ΔG ≈ -5 bis -7.

    TF-Strategien: Compact=0, Open=1
    Substrat-Strategien: Unfolded=0, Folded=1
    """
    RT_T = RT(T)

    # ── TF Payoffs ──
    # u_TF[s_TF, s_Substrat]
    u_TF = np.zeros((2, 2))

    # Compact + Unfolded: Schwache Bindung (Dimer-Form, kein Ribosom)
    u_TF[0, 0] = dG_bind_compact_unfolded  # negativ = stabilisierend
    # Compact + Folded: Kein Substrat-Kontakt (Substrat gefaltet)
    u_TF[0, 1] = 0.0

    # Open + Unfolded: Starke Bindung (Ribosom-assoziiert, molecular cradle)
    u_TF[1, 0] = dG_open + dG_bind_open_unfolded
    # Open + Folded: TF offen aber Substrat weg (Ribosom gebunden, Substrat gefaltet)
    u_TF[1, 1] = dG_open

    # ── Substrat Payoffs ──
    # u_S[s_Substrat, s_TF]
    u_S = np.zeros((2, 2))

    # Unfolded + Compact: Schwache Protektion
    u_S[0, 0] = dG_bind_compact_unfolded * 0.5  # Substrat profitiert weniger
    # Unfolded + Open: Starke Protektion (Aggregationsschutz)
    u_S[0, 1] = dG_bind_open_unfolded * 0.3  # Substrat-Anteil am Bindungsgewinn

    # Folded + Compact: Substrat gefaltet, stabil, TF irrelevant
    # Konvention: höherer Payoff = stabiler → -ΔG_fold (positiv für stabile Faltung)
    u_S[1, 0] = -dG_fold
    # Folded + Open: Substrat gefaltet, TF immer noch da → leichte Behinderung
    u_S[1, 1] = -dG_fold - 0.5

    params = {
        "dG_open_kcal": dG_open,
        "dG_fold_kcal": dG_fold,
        "dG_bind_open_unfolded_kcal": dG_bind_open_unfolded,
        "dG_bind_compact_unfolded_kcal": dG_bind_compact_unfolded,
        "T_C": T,
    }

    return u_TF, u_S, params


def robustness_scan():
    """Scanne über dG_open und dG_fold."""
    results = []
    for dG_open in [-0.5, -1.0, -1.5, -2.0, -3.0]:
        for dG_fold in [-3.0, -5.0, -7.0, -10.0]:
            u_TF, u_S, params = build_payoffs(dG_open=dG_open, dG_fold=dG_fold)
            pg = four_cycle_test_2x2(u_TF, u_S)
            nash = find_nash_pure(u_TF, u_S, 2,
                                  ["Compact", "Open"], ["Unfolded", "Folded"])
            results.append({
                "dG_open": dG_open,
                "dG_fold": dG_fold,
                "is_PG": pg["is_PG"],
                "violation": pg["violation"],
                "n_NE": len(nash),
                "NE_profiles": [(n["player_1"], n["player_2"]) for n in nash],
            })
    return results


def main():
    print("=" * 65)
    print("Trigger Factor — ATP-freier Chaperon-Kontrollfall")
    print("=" * 65)

    # 1. Abgeleitete Größen
    dq = derived_quantities()
    print(f"\n--- Experimentelle Bindungsenergien ---")
    print(f"  ΔG(TF:70S, 22°C):   {dq['dG_TF_70S_22C_kcal']:.2f} kcal/mol")
    print(f"  ΔG(TF:70S, 37°C):   {dq['dG_TF_70S_37C_kcal']:.2f} kcal/mol")
    print(f"  ΔG(TF:RNC, 22°C):   {dq['dG_TF_RNC_22C_kcal']:.2f} kcal/mol")
    print(f"  ΔG(TF:GAPDH, 4°C):  {dq['dG_TF_GAPDH_4C_kcal']:.2f} kcal/mol")
    print(f"  ΔG(TF:Peptid, 37°C):{dq['dG_TF_peptid_37C_kcal']:.2f} kcal/mol")
    print(f"  Selektivität RNC/70S: {dq['selectivity_RNC_vs_70S']:.1f}×")
    print(f"  ΔΔG(RNC vs 70S):     {dq['ddG_RNC_vs_70S_kcal']:.2f} kcal/mol")
    print(f"  ATP-Verbrauch:        0 (ATP-freies Chaperon)")

    # 2. Payoff-Matrizen
    u_TF, u_S, params = build_payoffs()

    print(f"\n--- Payoff-Matrizen (T={params['T_C']}°C) ---")
    print(f"  u_TF (Trigger Factor):")
    print(f"              Sub=Unfolded  Sub=Folded")
    print(f"    Compact    {u_TF[0,0]:8.2f}      {u_TF[0,1]:8.2f}")
    print(f"    Open       {u_TF[1,0]:8.2f}      {u_TF[1,1]:8.2f}")
    print(f"\n  u_S (Substrat):")
    print(f"              TF=Compact   TF=Open")
    print(f"    Unfolded   {u_S[0,0]:8.2f}      {u_S[0,1]:8.2f}")
    print(f"    Folded     {u_S[1,0]:8.2f}      {u_S[1,1]:8.2f}")

    # 3. PG-Test
    pg = four_cycle_test_2x2(u_TF, u_S)
    print(f"\n--- Potential-Game-Test ---")
    print(f"  I_TF = {pg['I_1']:.4f}")
    print(f"  I_S  = {pg['I_2']:.4f}")
    print(f"  Verletzung = {pg['violation']:.4f} kcal/mol")
    print(f"  PG = {pg['is_PG']}")

    # 4. Nash-GG
    strategies_TF = ["Compact", "Open"]
    strategies_S = ["Unfolded", "Folded"]
    nash = find_nash_pure(u_TF, u_S, 2, strategies_TF, strategies_S)
    print(f"\n--- Nash-Gleichgewichte ---")
    print(f"  Anzahl reine NE: {len(nash)}")
    for ne in nash:
        print(f"    TF={ne['player_1']}, Sub={ne['player_2']}  "
              f"u_TF={ne['u_1']:.2f}  u_S={ne['u_2']:.2f}  "
              f"Total={ne['u_total']:.2f}")

    # 5. Kritischer Q10-Test
    print(f"\n--- KRITISCHER KONTROLLTEST (Q10 in QUESTIONS.md) ---")
    if len(nash) == 1:
        print(f"  NE = 1 → ATP ist NICHT die Ursache der NE-Eindeutigkeit!")
        print(f"  Auch das ATP-freie Chaperon TF hat 1 NE.")
        print(f"  → Die NE-Eindeutigkeit folgt aus der Chaperon-FUNKTION,")
        print(f"    nicht aus der ATP-Kopplung.")
        print(f"  → Die ATP-Hypothese für Q10 ist WIDERLEGT.")
    elif len(nash) >= 2:
        print(f"  NE = {len(nash)} → ATP-freies Chaperon ist BISTABIL!")
        print(f"  → ATP-Hypothese GESTÄRKT: nur ATP-getriebene Systeme")
        print(f"    haben 1 NE, ATP-freie haben ≥2.")

    # 6. Vergleich mit Atlas
    print(f"\n--- Vergleich mit Chaperon-Atlas ---")
    print(f"  System          PG?    NE  Regime       ATP?")
    print(f"  XCL1            True    2  GG           Nein")
    print(f"  Hsp70/DnaJ      True    1  kin.NGG      Ja")
    print(f"  Hsp90/Cdc37     False   1  therm.NGG    Ja")
    print(f"  GroEL/GroES     False   1  therm.NGG    Ja")
    print(f"  KaiB/KaiC       False   1  therm.NGG    Indirekt")
    print(f"  Trigger Factor  {pg['is_PG']!s:<7} {len(nash)}  "
          f"{'GG' if pg['is_PG'] else 'non-GG':12} Nein")

    # 7. Robustheit
    print(f"\n--- Robustheitsscan ---")
    rob = robustness_scan()
    print(f"  {'dG_open':>8} {'dG_fold':>8} {'PG?':>5} {'Verl.':>8} {'NE':>4} Profile")
    for r in rob:
        profiles = ", ".join(f"({a},{b})" for a, b in r["NE_profiles"])
        print(f"  {r['dG_open']:>8.1f} {r['dG_fold']:>8.1f} "
              f"{'Y' if r['is_PG'] else 'N':>5} {r['violation']:>8.4f} "
              f"{r['n_NE']:>4} {profiles}")

    # ── JSON Output ──
    results = {
        "experiment": "Trigger Factor — ATP-freier Chaperon-Kontrollfall",
        "referenzen": {
            "thermodynamik": "Tetter et al. (2024) Sci Adv DOI:10.1126/sciadv.adn4824",
            "KD_ribosom": "Patzelt et al. (2002) Biol Chem 383:1611",
            "holdase_foldase": "Naqvi et al. (2022) Nat Commun 13:4051",
            "mech_foldase": "Haldar et al. (2017) Nat Commun 8:668",
            "realtime": "Kaiser et al. (2006) Nature 444:455",
            "synthetische_letalitat": "Deuerling et al. (1999) Nature 400:693",
        },
        "parameter": {
            "experimentell": {
                "KD_TF_70S_22C_uM": round(KD_TF_RIBOSOM_22C * 1e6, 2),
                "KD_TF_70S_37C_uM": round(KD_TF_RIBOSOM_37C * 1e6, 1),
                "KD_TF_RNC_22C_nM": round(KD_TF_RNC_22C * 1e9, 0),
                "KD_TF_GAPDH_4C_uM": round(KD_TF_GAPDH * 1e6, 2),
                "DH_TF_70S_kcal": DH_TF_70S,
                "DS_TF_70S_kcal_K": DS_TF_70S,
                "ATP_Verbrauch": 0,
            },
            "geschaetzt": {
                "dG_open_kcal": params["dG_open_kcal"],
                "dG_fold_kcal": params["dG_fold_kcal"],
            },
            "abgeleitet": dq,
        },
        "payoffs": {
            "u_TF": u_TF.tolist(),
            "u_S": u_S.tolist(),
            "strategien_TF": strategies_TF,
            "strategien_S": strategies_S,
        },
        "pg_test": pg,
        "nash_gleichgewichte": nash,
        "robustheit": rob,
        "Q10_kontrolltest": {
            "n_NE": len(nash),
            "ATP_frei": True,
            "interpretation": (
                "NE=1: ATP nicht ursächlich für NE-Eindeutigkeit"
                if len(nash) == 1
                else f"NE={len(nash)}: ATP-Hypothese gestärkt"
            ),
        },
    }

    out_path = save_results(results, __file__,
                            "trigger_factor_calibrated_results.json")
    print(f"\n→ Ergebnisse: {out_path}")

    print(f"\n{'='*65}")
    print(f"ERGEBNIS: Trigger Factor {'PG' if pg['is_PG'] else 'NICHT-PG'}, "
          f"{len(nash)} NE, KEIN ATP")
    if len(nash) == 1:
        print(f"  → NE-Eindeutigkeit ist Chaperon-Eigenschaft, NICHT ATP-Eigenschaft!")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
