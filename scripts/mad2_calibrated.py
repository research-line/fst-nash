"""
MAD2 Spindle-Checkpoint — Kalibriertes Nash-Modell
====================================================
Metamorphes Protein: Open-MAD2 (O-MAD2) vs. Closed-MAD2 (C-MAD2)
Template-katalysierte Konversion + TRIP13-abhaengiges Recycling

Spieler:
  MAD2:  {Open, Closed}
  CDC20: {Free, Bound}

Experimentelle Parameter:
  - Luo et al. 2004, Nat Struct Mol Biol 11:338
  - Sironi et al. 2002, EMBO J 21:2496
  - Simonetta et al. 2009, PLoS Biol 7:e10
  - Skinner et al. 2008, J Cell Biol 183:761
  - Ye et al. 2015, eLife 4:e07367
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fold_switch_template import (
    four_cycle_test_2x2, find_nash_pure, compute_potential_2x2,
    symmetry_breaking_analysis, save_results, RT
)

# ================================================================
# Experimentelle Parameter
# ================================================================

T = 30.0  # Celsius (Luo 2004)
RT_val = RT(T)  # 0.602 kcal/mol

# --- Konformationsstabilitaet ---
# C-MAD2 : O-MAD2 = 8:1 bei 30 Grad C (Luo 2004)
K_eq_CO = 8.0
dG_O_to_C = -RT_val * np.log(K_eq_CO)  # -1.25 kcal/mol
dG_conf = -dG_O_to_C  # +1.25 (C stabiler, positiv = guenstig)

# --- Bindungsaffinitaet C-MAD2:CDC20 ---
# K_D = 100 nM, ITC: dH=-13.9, TdS=-4.4, dG=-9.5 kcal/mol (Sironi 2002)
K_D_cdc20 = 100e-9  # M
dG_bind_total = -9.5  # kcal/mol
dH_bind = -13.9  # kcal/mol (Safety Belt H-Bonds -> MAD2-Seite)
TdS_bind = -4.4  # kcal/mol (Entropieverlust -> beide Seiten)

# --- O-MAD2 bindet CDC20 NICHT ---
# Safety Belt kann in Open-Konformation nicht geschlossen werden

# --- Kinetik ---
t_half_spontan_h = 9.0  # Stunden O->C (Luo 2004, 30 Grad C)
t_half_rueck_h = t_half_spontan_h * 6.0  # C->O 6x langsamer (Skinner 2008)
acceleration_template = 50  # >50x durch Template (Luo 2025, EMBO Rep)

# --- TRIP13 Recycling ---
atp_per_recycling = 9  # 8-10 ATP pro C->O Konversion (Ye 2015)
trip13_rate = 1.9  # MAD2/min pro TRIP13-Hexamer (Ye 2015, 37 Grad C)


# ================================================================
# Payoff-Konstruktion
# ================================================================

def build_payoffs(dG_conformational=None, f_mad2=0.55):
    """2x2 Payoff-Matrizen fuer MAD2 x CDC20.

    Konvention: hoeher = stabiler (= -dG)

    MAD2:  {0=Open, 1=Closed}
    CDC20: {0=Free, 1=Bound}

    Bindungsenergie-Split: f_mad2 fuer MAD2, (1-f_mad2) fuer CDC20.
    Begruendung: Safety Belt (H-Bonds) stabilisiert hauptsaechlich MAD2.
    """
    if dG_conformational is None:
        dG_conformational = dG_conf

    bind_total = -dG_bind_total  # 9.5 kcal/mol (positiv = guenstig)
    bind_mad2 = bind_total * f_mad2
    bind_cdc20 = bind_total * (1 - f_mad2)

    u_M = np.zeros((2, 2))
    u_M[0, 0] = 0.0                          # Open, Free (Referenz)
    u_M[0, 1] = 0.0                          # Open, Bound (kein Komplex moeglich)
    u_M[1, 0] = dG_conformational             # Closed, Free (intrinsisch stabiler)
    u_M[1, 1] = dG_conformational + bind_mad2  # Closed, Bound

    u_C = np.zeros((2, 2))
    u_C[0, 0] = 0.0          # Free, Open (Referenz)
    u_C[0, 1] = 0.0          # Free, Closed (CDC20 frei, MAD2 irrelevant)
    u_C[1, 0] = 0.0          # Bound, Open (kein Komplex moeglich)
    u_C[1, 1] = bind_cdc20   # Bound, Closed (stabiler Safety-Belt-Komplex)

    return u_M, u_C


def main():
    print("=" * 65)
    print("MAD2 Spindle-Checkpoint -- Kalibriertes Nash-Modell")
    print("=" * 65)

    # --- Parameter ---
    print(f"\n--- Experimentelle Parameter (T = {T} Grad C, RT = {RT_val:.3f} kcal/mol) ---")
    print(f"  K_eq(C:O) = {K_eq_CO}:1 (Luo 2004)")
    print(f"  dG(O->C) = {dG_O_to_C:.2f} kcal/mol")
    print(f"  K_D(C-MAD2:CDC20) = {K_D_cdc20*1e9:.0f} nM (Sironi 2002)")
    print(f"  dG(Bind) = {dG_bind_total:.1f} kcal/mol (ITC: dH={dH_bind}, TdS={TdS_bind})")
    print(f"  O-MAD2:CDC20 = KEIN Komplex (Safety Belt fehlt)")
    print(f"  t_1/2(O->C) = {t_half_spontan_h:.0f} h spontan, "
          f"~{t_half_spontan_h/acceleration_template*60:.0f} min katalysiert")
    print(f"  TRIP13: {atp_per_recycling} ATP/Recycling (Ye 2015)")

    # --- Payoffs ---
    u_M, u_C = build_payoffs()
    st_M = ["Open", "Closed"]
    st_C = ["Free", "Bound"]

    print(f"\n--- Payoff-Matrizen ---")
    print(f"\n  u_MAD2[mad2, cdc20]:")
    print(f"            {'Free':>10s}  {'Bound':>10s}")
    for i, s in enumerate(st_M):
        print(f"    {s:<10s}  {u_M[i,0]:10.2f}  {u_M[i,1]:10.2f}")

    print(f"\n  u_CDC20[cdc20, mad2]:")
    print(f"            {'Open':>10s}  {'Closed':>10s}")
    for j, s in enumerate(st_C):
        print(f"    {s:<10s}  {u_C[j,0]:10.2f}  {u_C[j,1]:10.2f}")

    # --- PG-Test ---
    pg = four_cycle_test_2x2(u_M, u_C)
    print(f"\n--- Vier-Zyklen-Test ---")
    print(f"  I_MAD2 = {pg['I_1']:.4f}")
    print(f"  I_CDC20 = {pg['I_2']:.4f}")
    print(f"  Verletzung = {pg['violation']:.4f} kcal/mol")
    print(f"  PG = {pg['is_PG']}")

    # --- Nash-GG ---
    nash = find_nash_pure(u_M, u_C, 2, st_M, st_C)
    print(f"\n--- Nash-Gleichgewichte ---")
    print(f"  Anzahl reine NE: {len(nash)}")
    for ne in nash:
        print(f"    MAD2={ne['player_1']}, CDC20={ne['player_2']}")
        print(f"      u_M={ne['u_1']:.2f}, u_C={ne['u_2']:.2f}, Total={ne['u_total']:.2f}")

    # --- Potential ---
    if pg['is_PG']:
        pot = compute_potential_2x2(u_M, u_C)
        print(f"\n--- Potential-Funktion ---")
        for k, v in pot['values'].items():
            print(f"    Phi{k} = {v:.4f}")

    # --- Goloubinoff S1-S4 ---
    sb = symmetry_breaking_analysis(u_M, u_C, 2)
    print(f"\n--- Goloubinoff S1-S4 ---")
    print(f"  S1 (Spieler-Symmetrie): gebrochen={sb['S1_player_symmetry']['broken']}")
    print(f"  S4 (Potential Game): {sb['S4_potential_game']['is_PG']}")

    # --- Biologische Interpretation ---
    print(f"\n--- Biologische Interpretation ---")
    if len(nash) == 1 and nash[0]['player_1'] == 'Closed':
        ne = nash[0]
        print(f"  Eindeutiges NE: ({ne['player_1']}, {ne['player_2']})")
        print(f"  = C-MAD2:CDC20 Komplex = Checkpoint-aktiver Zustand")
        print(f"  O-MAD2 Population in vivo wird aktiv aufrechterhalten (TRIP13)")
        print(f"  Checkpoint-Silencing = TRIP13 treibt System WEG vom NE")
    elif len(nash) == 2:
        print(f"  Zwei NE: Bistabilitaet")
        for ne in nash:
            print(f"    ({ne['player_1']}, {ne['player_2']})")

    # --- Vergleich XCL1 vs MAD2 ---
    print(f"\n--- Vergleich: MAD2 vs. XCL1 ---")
    pct_C = 100 * K_eq_CO / (1 + K_eq_CO)
    print(f"  XCL1: dG ~0 (50:50), unreguliert -> 2 NE")
    print(f"  MAD2: dG = {dG_O_to_C:.2f} ({pct_C:.0f}:{100-pct_C:.0f} C:O), "
          f"TRIP13-reguliert -> {len(nash)} NE")
    print(f"  Metamorphe Struktur allein bestimmt NE-Zahl NICHT.")
    print(f"  Entscheidend: Stabilitaets-Asymmetrie + funktionelle Regulierung.")

    # --- Stabilitaetsscan ---
    print(f"\n--- Stabilitaetsscan: Wann entstehen 2 NE? ---")
    print(f"  {'dG(O->C)':>10s}  {'K_eq(C:O)':>10s}  {'NE':>4s}  {'PG?':>4s}  Profil")

    scan_results = []
    for dg_oc in np.arange(2.0, -2.6, -0.5):
        dg_c = -dg_oc
        u_Ms, u_Cs = build_payoffs(dG_conformational=dg_c)
        pg_s = four_cycle_test_2x2(u_Ms, u_Cs)
        nash_s = find_nash_pure(u_Ms, u_Cs, 2, st_M, st_C)

        k_eq = np.exp(-dg_oc / RT_val) if abs(dg_oc) < 20 else float('inf')
        profil = ", ".join(f"({n['player_1']},{n['player_2']})" for n in nash_s)

        print(f"  {dg_oc:10.1f}  {k_eq:10.1f}  {len(nash_s):4d}  "
              f"{'J' if pg_s['is_PG'] else 'N':>4s}  {profil}")

        scan_results.append({
            "dG_O_to_C": round(float(dg_oc), 1),
            "K_eq_CO": round(float(k_eq), 2),
            "n_NE": len(nash_s),
            "is_PG": pg_s["is_PG"],
            "profiles": profil,
        })

    # --- Bindungsenergiesplit-Scan ---
    print(f"\n--- Split-Scan: Einfluss der Bindungsenergieverteilung ---")
    print(f"  {'f_MAD2':>8s}  {'Verl.':>8s}  {'NE':>4s}  {'PG?':>4s}")
    split_results = []
    for f in np.arange(0.3, 0.71, 0.05):
        u_Ms, u_Cs = build_payoffs(f_mad2=f)
        pg_s = four_cycle_test_2x2(u_Ms, u_Cs)
        nash_s = find_nash_pure(u_Ms, u_Cs, 2, st_M, st_C)
        print(f"  {f:8.2f}  {pg_s['violation']:8.4f}  {len(nash_s):4d}  "
              f"{'J' if pg_s['is_PG'] else 'N':>4s}")
        split_results.append({
            "f_mad2": round(float(f), 2),
            "violation": pg_s["violation"],
            "n_NE": len(nash_s),
            "is_PG": pg_s["is_PG"],
        })

    # --- Ergebnisse ---
    results = {
        "system": "MAD2 (Spindle Assembly Checkpoint)",
        "type": "metamorphic_regulated",
        "temperature_C": T,
        "RT_kcal_mol": round(float(RT_val), 4),
        "experimental_parameters": {
            "K_eq_CO": K_eq_CO,
            "dG_O_to_C_kcal_mol": round(float(dG_O_to_C), 4),
            "K_D_CDC20_nM": K_D_cdc20 * 1e9,
            "dG_bind_total_kcal_mol": dG_bind_total,
            "dH_bind_kcal_mol": dH_bind,
            "TdS_bind_kcal_mol": TdS_bind,
            "O_MAD2_CDC20_binding": "none (Safety Belt requires Closed)",
            "ATP_per_recycling": atp_per_recycling,
            "TRIP13_rate_per_min": trip13_rate,
            "t_half_spontaneous_h": t_half_spontan_h,
            "t_half_catalyzed_min": round(t_half_spontan_h / acceleration_template * 60, 1),
        },
        "references": {
            "Luo_2004": "Luo et al. 2004, Nat Struct Mol Biol 11:338",
            "Sironi_2002": "Sironi et al. 2002, EMBO J 21:2496",
            "Simonetta_2009": "Simonetta et al. 2009, PLoS Biol 7:e10",
            "Skinner_2008": "Skinner et al. 2008, J Cell Biol 183:761",
            "Ye_2015": "Ye et al. 2015, eLife 4:e07367",
        },
        "payoff_matrices": {
            "u_MAD2": u_M.tolist(),
            "u_CDC20": u_C.tolist(),
            "states_MAD2": st_M,
            "states_CDC20": st_C,
        },
        "four_cycle_test": pg,
        "nash_equilibria": nash,
        "n_NE": len(nash),
        "goloubinoff_symmetry": sb,
        "stability_scan": scan_results,
        "split_scan": split_results,
        "interpretation": {
            "is_metamorphic": True,
            "is_regulated": True,
            "regulator": "TRIP13 + p31comet (ATP-driven C->O recycling, 8-10 ATP)",
            "NE_count": len(nash),
            "NE_profile": f"({nash[0]['player_1']}, {nash[0]['player_2']})" if nash else "none",
            "comparison_XCL1": (
                "XCL1: dG~0 (50:50, unreguliert) -> 2 NE. "
                "MAD2: dG=-1.25 (89:11, TRIP13-reguliert) -> 1 NE. "
                "Intrinsische Stabilitaets-Asymmetrie eliminiert 2. NE."
            ),
            "key_insight": (
                "MAD2 ist strukturell metamorph (zwei stabile Folds), aber "
                "funktionell reguliert (TRIP13 recycelt C-MAD2 aktiv). "
                "C-MAD2 dominiert intrinsisch (8:1) -> nur 1 NE. "
                "Stabilitaetsscan zeigt: 2 NE erst bei dG(O->C) > 0 "
                "(O muesste stabiler sein). "
                "Checkpoint-Silencing = TRIP13 treibt System WEG vom NE."
            ),
            "hypothesis_update": (
                "Revidierte NE-Hypothese konsistent, aber NICHT bestaetigt: "
                "1 NE folgt trivial aus Konstruktion (Closed strikt dominant, "
                "da u_M[Closed,*] > u_M[Open,*] in jeder Spalte). "
                "Der Test ist nur fuer Systeme mit dG ~ 0 nicht-trivial. "
                "N=1 auf 2-NE-Seite (nur XCL1) — ununterscheidbar von Ausreisser."
            ),
        },
    }

    out = save_results(results, __file__, "mad2_calibrated_results.json")
    print(f"\n-> Ergebnisse: {out}")

    print(f"\n{'=' * 65}")
    print(f"ERGEBNIS: MAD2 (metamorph, reguliert) = Non-PG, {len(nash)} NE")
    if len(nash) == 1:
        ne = nash[0]
        print(f"  NE: ({ne['player_1']}, {ne['player_2']})")
    print(f"  Konsistent mit Hypothese v3, aber 1 NE trivial aus Konstruktion")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
