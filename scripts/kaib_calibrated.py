"""
KaiB/KaiC — Kalibriertes Fold-Switching/Circadiane-Uhr-Spiel
=============================================================
Experimentelle Daten:
  Zhang et al. (2024) PNAS 121:e2412327121 — Thermodynamik gs↔fs
  Wayment-Steele et al. (2024) PNAS 121:e2412293121 — Kinetik (R. sphaeroides)
  Terauchi et al. (2007) PNAS 104:16377 — KaiC ATPase
  Snijder et al. (2017) Science 355:1181 — 6:6 Komplex
  Chang et al. (2015) Science 349:324 — Fold-Switch-Entdeckung
  Tseng et al. (2017) Science 355:1174 — fsKaiB-CI Struktur (PDB 5JWR)

Spieler:
  1. KaiB: {Ground-State (GS), Fold-Switched (FS)}
  2. KaiC-CI: {Post-Hydrolysis/ADP-bound (Bindungskompetent),
               Pre-Hydrolysis/ATP-bound (Nicht-Bindungskompetent)}

Biologischer Hintergrund:
  KaiB's Fold-Switch (gs↔fs, Stunden-Zeitskala) ist intrinsisch — KEIN ATP
  in KaiB selbst. Aber KaiC's CI-Domäne hydrolysiert ATP (14.5 ATP/Tag),
  und NUR der post-Hydrolyse-Zustand (ADP-bound) ist bindungskompetent
  für fsKaiB. → ATP-Kopplung ist INDIREKT über KaiC.

  Die Asymmetrie: KaiC's Zustand beeinflusst KaiB's Bindungsgewinn stark
  (nur ADP-KaiC bindet fsKaiB), aber KaiB's Zustand beeinflusst KaiC's
  ATPase nur schwach (moduliert Dephosphorylierungsrate).
  → Erwartung: NICHT-PG (asymmetrische Kopplung, ATP-getrieben).

Kalibrierungsstatus:
  ΔG(gs→fs): KALIBRIERT (Zhang 2024, +0.4 kcal/mol bei 30°C)
  Kinetik: KALIBRIERT (Wayment-Steele 2024, R. sphaeroides)
  KaiC-Bindung: GESCHÄTZT (kein K_D publiziert, kooperativ)
  KaiC-ATPase: KALIBRIERT (Terauchi 2007)
  Delta_KaiC (Rückwirkung): GESCHÄTZT
"""

import numpy as np
import json
from pathlib import Path

# Importiere Template-Funktionen
import sys
sys.path.insert(0, str(Path(__file__).parent))
from fold_switch_template import (
    four_cycle_test_2x2, find_nash_pure, compute_potential_2x2,
    symmetry_breaking_analysis, save_results, RT
)


# ── Experimentelle Parameter ──────────────────────────────────────

# Zhang et al. (2024) PNAS — T. elongatus KaiB_te* Konstrukt
DELTA_H_GS_FS = -19.6    # kcal/mol (= -82 kJ/mol)
DELTA_S_GS_FS = -0.066   # kcal/(mol·K) (= -276 J/(mol·K))

# Wayment-Steele et al. (2024) — R. sphaeroides KaiB bei 20°C
K_GS_TO_FS = 0.09   # h⁻¹ (= 2.5e-5 s⁻¹)
K_FS_TO_GS = 0.36   # h⁻¹ (= 1.0e-4 s⁻¹)
KEQ_RSPHAEROIDES_20C = K_GS_TO_FS / K_FS_TO_GS  # = 0.25

# Terauchi et al. (2007) — KaiC ATPase
KAIC_ATPASE_PER_DAY = 14.5  # ATP/Tag pro Monomer
DG_ATP = -7.3  # kcal/mol unter zellulären Bedingungen

# Garcia-Pino et al. (2022) — Simulation
DG_MONOMER_GS_FS = -1.0  # kcal/mol (als Monomer: fs leicht bevorzugt!)
DG_DIMER_DISSOC = 30.0   # kcal/mol pro Monomer (Tetramer hält gs fest)

# Tseng et al. (2017) — Interface fsKaiB-KaiC
INTERFACE_AREA_A2 = 1000  # Å²

RT_30C = RT(30)  # 0.602 kcal/mol


def dG_fold_switch(T_celsius):
    """ΔG(gs→fs) bei gegebener Temperatur (Zhang 2024)."""
    T_K = T_celsius + 273.15
    return DELTA_H_GS_FS - T_K * DELTA_S_GS_FS


def derived_quantities():
    """Thermodynamische Größen aus experimentellen Daten."""
    dG_30 = dG_fold_switch(30)
    dG_20 = dG_fold_switch(20)
    dG_37 = dG_fold_switch(37)

    keq_30 = np.exp(-dG_30 / RT(30))
    keq_20 = np.exp(-dG_20 / RT(20))

    f_fs_20 = KEQ_RSPHAEROIDES_20C / (1 + KEQ_RSPHAEROIDES_20C)

    return {
        "dG_gs_fs_20C_kcal": round(dG_20, 3),
        "dG_gs_fs_30C_kcal": round(dG_30, 3),
        "dG_gs_fs_37C_kcal": round(dG_37, 3),
        "Keq_Rsphaeroides_20C": round(KEQ_RSPHAEROIDES_20C, 3),
        "f_fs_Rsphaeroides_20C": round(f_fs_20, 3),
        "Keq_from_dG_30C": round(keq_30, 4),
        "tau_gs_h": round(1 / K_GS_TO_FS, 1),
        "tau_fs_h": round(1 / K_FS_TO_GS, 1),
        "KaiC_ATP_per_hour": round(KAIC_ATPASE_PER_DAY / 24, 2),
    }


# ── Payoff-Konstruktion ──────────────────────────────────────────

def build_payoffs(dG_bind_fsKaiB_KaiC=-6.0, delta_KaiC=1.0, T=30):
    """Asymmetrisches 2×2 Spiel: KaiB × KaiC-CI.

    Parameter:
      dG_bind_fsKaiB_KaiC: Bindungsenergie fsKaiB + ADP-KaiC (kcal/mol).
        Geschätzt aus Interface-Fläche (~1000 Å², typisch -6 bis -10 kcal/mol).
        Kein K_D publiziert (kooperative 6:6-Bindung).
      delta_KaiC: Rückwirkung von KaiB auf KaiC (kcal/mol).
        fsKaiB-Bindung stabilisiert KaiC im ADP-Zustand (verlangsamt
        ADP-Release), was KaiC-Dephosphorylierung fördert.
      T: Temperatur (°C).

    Payoff-Konvention: Höher = stabiler (negativere freie Energie).

    KaiB-Strategien: GS=0, FS=1
    KaiC-Strategien: ADP(post-hydrolysis)=0, ATP(pre-hydrolysis)=1
    """
    dG_gs_fs = dG_fold_switch(T)
    RT_T = RT(T)

    # ── KaiB Payoffs ──
    # u_KaiB[s_KaiB, s_KaiC]
    u_KaiB = np.zeros((2, 2))

    # GS + ADP-KaiC: Referenz (gs-KaiB bindet nicht an KaiC)
    u_KaiB[0, 0] = 0.0
    # GS + ATP-KaiC: gs-KaiB bindet auch nicht → gleich
    u_KaiB[0, 1] = 0.0

    # FS + ADP-KaiC: Fold-Switch-Kosten + Bindungsgewinn
    u_KaiB[1, 0] = -dG_gs_fs + dG_bind_fsKaiB_KaiC
    # FS + ATP-KaiC: Fold-Switch-Kosten, KEINE Bindung (ATP-KaiC nicht kompetent)
    u_KaiB[1, 1] = -dG_gs_fs

    # ── KaiC Payoffs ──
    # u_KaiC[s_KaiC, s_KaiB]  (Beachte: Transposition!)
    u_KaiC = np.zeros((2, 2))

    # ADP + GS-KaiB: Post-Hydrolyse, keine KaiB-Bindung
    u_KaiC[0, 0] = 0.0
    # ADP + FS-KaiB: Post-Hydrolyse + fsKaiB bindet → Stabilisierung
    u_KaiC[0, 1] = delta_KaiC

    # ATP + GS-KaiB: Pre-Hydrolyse, energetisch höher (ATP gebunden)
    u_KaiC[1, 0] = DG_ATP  # negativ: ATP-Hydrolyse ist exergon
    # ATP + FS-KaiB: Pre-Hydrolyse, fsKaiB kann nicht binden
    u_KaiC[1, 1] = DG_ATP

    params = {
        "dG_gs_fs_kcal": round(dG_gs_fs, 3),
        "dG_bind_kcal": dG_bind_fsKaiB_KaiC,
        "delta_KaiC_kcal": delta_KaiC,
        "T_C": T,
        "RT_kcal": round(RT_T, 4),
    }

    return u_KaiB, u_KaiC, params


def analyze_asymmetry(u_KaiB, u_KaiC):
    """Analysiere die Kopplungsasymmetrie (S2-Brechung).

    Interaktionskontrast:
    I_KaiB = u_KaiB[GS,ADP] - u_KaiB[GS,ATP] - u_KaiB[FS,ADP] + u_KaiB[FS,ATP]
    I_KaiC = u_KaiC[ADP,GS] - u_KaiC[ADP,FS] - u_KaiC[ATP,GS] + u_KaiC[ATP,FS]
    """
    I_KaiB = u_KaiB[0, 0] - u_KaiB[0, 1] - u_KaiB[1, 0] + u_KaiB[1, 1]
    I_KaiC = u_KaiC[0, 0] - u_KaiC[0, 1] - u_KaiC[1, 0] + u_KaiC[1, 1]

    return {
        "I_KaiB": round(float(I_KaiB), 4),
        "I_KaiC": round(float(I_KaiC), 4),
        "Delta_I": round(float(abs(I_KaiB - I_KaiC)), 4),
        "interpretation": (
            f"KaiB spürt KaiC-Zustand mit {abs(I_KaiB):.1f} kcal/mol "
            f"(Bindung nur an ADP-KaiC). "
            f"KaiC spürt KaiB-Zustand mit {abs(I_KaiC):.1f} kcal/mol "
            f"(Rückwirkung). "
            f"Asymmetrie = {abs(I_KaiB - I_KaiC):.1f} kcal/mol."
        ),
    }


def robustness_scan():
    """Scanne über Bindungsenergie und delta_KaiC."""
    results = []
    for dG_bind in [-4.0, -6.0, -8.0, -10.0]:
        for delta_KaiC in [0.0, 0.5, 1.0, 2.0, 3.0]:
            u_KaiB, u_KaiC, params = build_payoffs(
                dG_bind_fsKaiB_KaiC=dG_bind,
                delta_KaiC=delta_KaiC,
            )
            pg = four_cycle_test_2x2(u_KaiB, u_KaiC)
            nash = find_nash_pure(u_KaiB, u_KaiC, 2,
                                  ["GS", "FS"], ["ADP", "ATP"])
            results.append({
                "dG_bind": dG_bind,
                "delta_KaiC": delta_KaiC,
                "is_PG": pg["is_PG"],
                "violation": pg["violation"],
                "n_NE": len(nash),
                "NE_profiles": [(n["player_1"], n["player_2"]) for n in nash],
            })
    return results


def temperature_scan():
    """Temperaturabhängigkeit (Temperaturkompensation der Uhr)."""
    results = []
    for T in [15, 20, 25, 30, 35, 40]:
        u_KaiB, u_KaiC, params = build_payoffs(T=T)
        pg = four_cycle_test_2x2(u_KaiB, u_KaiC)
        nash = find_nash_pure(u_KaiB, u_KaiC, 2,
                              ["GS", "FS"], ["ADP", "ATP"])
        results.append({
            "T_C": T,
            "dG_gs_fs": params["dG_gs_fs_kcal"],
            "is_PG": pg["is_PG"],
            "violation": pg["violation"],
            "n_NE": len(nash),
            "NE_profiles": [(n["player_1"], n["player_2"]) for n in nash],
        })
    return results


def main():
    print("=" * 65)
    print("KaiB/KaiC — Kalibriertes Fold-Switching/Circadiane-Uhr-Spiel")
    print("=" * 65)

    # 1. Abgeleitete Größen
    dq = derived_quantities()
    print(f"\n--- Experimentelle Größen ---")
    print(f"  ΔG(gs→fs, 20°C): {dq['dG_gs_fs_20C_kcal']:.3f} kcal/mol")
    print(f"  ΔG(gs→fs, 30°C): {dq['dG_gs_fs_30C_kcal']:.3f} kcal/mol")
    print(f"  ΔG(gs→fs, 37°C): {dq['dG_gs_fs_37C_kcal']:.3f} kcal/mol")
    print(f"  K_eq(R.sph., 20°C): {dq['Keq_Rsphaeroides_20C']:.3f}")
    print(f"  f(fs, R.sph., 20°C): {dq['f_fs_Rsphaeroides_20C']:.1%}")
    print(f"  τ(gs): {dq['tau_gs_h']:.1f} h, τ(fs): {dq['tau_fs_h']:.1f} h")
    print(f"  KaiC ATPase: {dq['KaiC_ATP_per_hour']:.2f} ATP/h pro Monomer")

    # 2. Payoff-Matrizen (Referenz: 30°C)
    dG_bind = -6.0
    delta_KaiC = 1.0
    u_KaiB, u_KaiC, params = build_payoffs(
        dG_bind_fsKaiB_KaiC=dG_bind,
        delta_KaiC=delta_KaiC,
    )

    print(f"\n--- Payoff-Matrizen (T={params['T_C']}°C) ---")
    print(f"  Parameter: ΔG(gs→fs)={params['dG_gs_fs_kcal']:.3f}, "
          f"ΔG_bind={dG_bind}, δ_KaiC={delta_KaiC}")
    print(f"\n  u_KaiB (KaiB-Payoffs):")
    print(f"              KaiC=ADP    KaiC=ATP")
    print(f"    GS       {u_KaiB[0,0]:8.3f}    {u_KaiB[0,1]:8.3f}")
    print(f"    FS       {u_KaiB[1,0]:8.3f}    {u_KaiB[1,1]:8.3f}")
    print(f"\n  u_KaiC (KaiC-Payoffs):")
    print(f"              KaiB=GS    KaiB=FS")
    print(f"    ADP      {u_KaiC[0,0]:8.3f}    {u_KaiC[0,1]:8.3f}")
    print(f"    ATP      {u_KaiC[1,0]:8.3f}    {u_KaiC[1,1]:8.3f}")

    # 3. PG-Test
    pg = four_cycle_test_2x2(u_KaiB, u_KaiC)
    print(f"\n--- Potential-Game-Test (Monderer & Shapley 1996) ---")
    print(f"  I_KaiB = {pg['I_1']:.4f}")
    print(f"  I_KaiC = {pg['I_2']:.4f}")
    print(f"  Verletzung = {pg['violation']:.4f} kcal/mol")
    print(f"  PG = {pg['is_PG']}")

    # 4. Asymmetrie-Analyse
    asym = analyze_asymmetry(u_KaiB, u_KaiC)
    print(f"\n--- Kopplungsasymmetrie (S2-Analyse) ---")
    print(f"  {asym['interpretation']}")

    # 5. Nash-GG
    strategies_KaiB = ["GS", "FS"]
    strategies_KaiC = ["ADP", "ATP"]
    nash = find_nash_pure(u_KaiB, u_KaiC, 2, strategies_KaiB, strategies_KaiC)
    print(f"\n--- Nash-Gleichgewichte ---")
    print(f"  Anzahl reine NE: {len(nash)}")
    for ne in nash:
        print(f"    KaiB={ne['player_1']}, KaiC={ne['player_2']}  "
              f"u_KaiB={ne['u_1']:.3f}  u_KaiC={ne['u_2']:.3f}  "
              f"Total={ne['u_total']:.3f}")

    # 6. Biologische Interpretation
    print(f"\n--- Biologische Interpretation ---")
    if not pg["is_PG"]:
        print(f"  KaiB/KaiC ist NICHT-PG (Verletzung {pg['violation']:.1f} kcal/mol).")
        print(f"  Asymmetrie: KaiC's ADP/ATP-Zustand bestimmt KaiB-Bindung")
        print(f"  (I_KaiB={asym['I_KaiB']:.1f}), aber KaiB's Rückwirkung auf KaiC")
        print(f"  ist schwächer (I_KaiC={asym['I_KaiC']:.1f}).")
        print(f"  → Gerichtete Kopplung, konsistent mit ATP-getriebenem Zyklus.")
    else:
        print(f"  ÜBERRASCHUNG: KaiB/KaiC wäre PG → Gleichgewichtssystem")

    if len(nash) == 1:
        print(f"\n  1 NE: {nash[0]['player_1']},{nash[0]['player_2']}")
        print(f"  → Wie bei Chaperon-Systemen: eindeutiger Zielzustand.")
        print(f"  → ATP-getriebene Kopplung eliminiert alternatives NE.")
    elif len(nash) == 2:
        print(f"\n  2 NE: Bistabilität wie bei XCL1!")
        print(f"  → Trotz ATP-Kopplung: zwei stabile Zustände koexistieren.")

    # 7. Vergleich mit XCL1
    print(f"\n--- Vergleich: KaiB/KaiC vs. XCL1 ---")
    print(f"  XCL1:     PG=True,  NE=2, Regime=Gleichgewicht, kein ATP")
    print(f"  KaiB/KaiC: PG={pg['is_PG']}, NE={len(nash)}, "
          f"Regime={'GG' if pg['is_PG'] else 'Nicht-GG'}, "
          f"ATP={'indirekt (KaiC)' if not pg['is_PG'] else 'nein'}")

    # 8. Robustheit
    print(f"\n--- Robustheitsscan ---")
    rob = robustness_scan()
    print(f"  {'dG_bind':>8} {'δ_KaiC':>8} {'PG?':>5} {'Verl.':>8} {'NE':>4} {'Profile'}")
    for r in rob:
        profiles = ", ".join(f"({a},{b})" for a, b in r["NE_profiles"])
        print(f"  {r['dG_bind']:>8.1f} {r['delta_KaiC']:>8.1f} "
              f"{'Y' if r['is_PG'] else 'N':>5} {r['violation']:>8.4f} "
              f"{r['n_NE']:>4} {profiles}")

    # 9. Temperaturabhängigkeit
    print(f"\n--- Temperaturabhängigkeit ---")
    temp = temperature_scan()
    for t in temp:
        profiles = ", ".join(f"({a},{b})" for a, b in t["NE_profiles"])
        print(f"  T={t['T_C']:2d}°C: ΔG(gs→fs)={t['dG_gs_fs']:+.3f}, "
              f"PG={t['is_PG']}, NE={t['n_NE']}, {profiles}")

    # ── JSON Output ──
    results = {
        "experiment": "KaiB/KaiC kalibriertes Fold-Switching/Circadiane-Uhr-Spiel",
        "referenzen": {
            "thermodynamik": "Zhang et al. (2024) PNAS 121:e2412327121",
            "kinetik": "Wayment-Steele et al. (2024) PNAS 121:e2412293121",
            "ATPase": "Terauchi et al. (2007) PNAS 104:16377",
            "struktur_komplex": "Snijder et al. (2017) Science 355:1181",
            "fold_switch": "Chang et al. (2015) Science 349:324",
            "CI_struktur": "Tseng et al. (2017) Science 355:1174",
        },
        "parameter": {
            "experimentell": {
                "DeltaH_gs_fs_kcal": DELTA_H_GS_FS,
                "DeltaS_gs_fs_kcal_K": DELTA_S_GS_FS,
                "k_gs_to_fs_per_h": K_GS_TO_FS,
                "k_fs_to_gs_per_h": K_FS_TO_GS,
                "Keq_Rsphaeroides_20C": KEQ_RSPHAEROIDES_20C,
                "KaiC_ATPase_per_day": KAIC_ATPASE_PER_DAY,
            },
            "geschaetzt": {
                "dG_bind_fsKaiB_KaiC": dG_bind,
                "delta_KaiC": delta_KaiC,
            },
            "abgeleitet": dq,
        },
        "payoffs": {
            "u_KaiB": u_KaiB.tolist(),
            "u_KaiC": u_KaiC.tolist(),
            "strategien_KaiB": strategies_KaiB,
            "strategien_KaiC": strategies_KaiC,
        },
        "pg_test": pg,
        "asymmetrie": asym,
        "nash_gleichgewichte": nash,
        "robustheit": rob,
        "temperatur": temp,
        "schlussfolgerung": {
            "is_PG": pg["is_PG"],
            "n_NE": len(nash),
            "regime": "Nicht-GG (ATP-indirekt)" if not pg["is_PG"] else "GG",
            "vergleich_XCL1": (
                "XCL1=PG/2NE/GG vs KaiB/KaiC="
                f"{'PG' if pg['is_PG'] else 'non-PG'}/{len(nash)}NE/"
                f"{'GG' if pg['is_PG'] else 'non-GG'}"
            ),
        },
    }

    out_path = save_results(results, __file__, "kaib_calibrated_results.json")
    print(f"\n→ Ergebnisse: {out_path}")

    print(f"\n{'='*65}")
    print(f"ERGEBNIS: KaiB/KaiC {'NICHT-PG' if not pg['is_PG'] else 'PG'}, "
          f"{len(nash)} NE")
    print(f"  PG-Verletzung: {pg['violation']:.4f} kcal/mol")
    print(f"  Regime: {'Nicht-Gleichgewicht (ATP via KaiC)' if not pg['is_PG'] else 'Gleichgewicht'}")
    print(f"  Vergleich: XCL1 (PG, 2 NE) vs KaiB/KaiC "
          f"({'non-PG' if not pg['is_PG'] else 'PG'}, {len(nash)} NE)")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
