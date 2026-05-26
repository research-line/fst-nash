"""
Prefoldin (PFD) — Kalibriertes Nash-Modell
=============================================
ATP-freier Holdase: fängt ungefaltete Substrate, übergibt an TRiC/CCT.
Zweiter ATP-freier Kontrollfall nach Trigger Factor (C5 im Atlas).

Spieler:
  PFD:      {Open (Tentakel relaxiert), Closed (Tentakel kontrahiert)}
  Substrat: {Free (in Lösung), Bound (in PFD-Kavität)}

ACHTUNG: Mehrere Parameter sind GESCHÄTZT (kein publizierter Wert).
Alle Schätzungen sind markiert. Ergebnis abhängig von Schätzungen —
Stabilitätsscan zeigt Sensitivität.

Experimentelle Parameter:
  - Zako et al. 2005, FEBS Lett 579:3718 (k_off PFD:Substrat)
  - Lundin et al. 2004, PNAS 101:15524 (Tentakelrotation ~12 Grad)
  - Sahlan et al. 2010, J Mol Biol 399:628 (K_D PFD:CPN)
  - Gestaut et al. 2019, Science 366:1190 (Cryo-EM PFD-TRiC)
  - Siegert et al. 2000, Cell 103:621 (PFD-Struktur)
  - Vainberg et al. 1998, Cell 93:863 (PFD-Funktion)
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
# Experimentelle + geschätzte Parameter
# ================================================================

T = 25.0  # Celsius (SPR-Messungen bei 25 Grad C, Sahlan 2010)
RT_val = RT(T)  # 0.593 kcal/mol

# --- PFD:Substrat Bindung ---
# GESCHÄTZT: Kein publizierter K_D(PFD:Actin/Tubulin)!
# k_off(PFD:CS) = 0.028 s^-1, k_off(PFD:GFP) = 0.011 s^-1 (Zako 2005)
# Typischer k_on ~ 10^5 M^-1 s^-1 -> K_D ~ 100-300 nM
K_D_substrate_nM = 100  # nM (GESCHÄTZT, Bereich 10-300 nM)
K_D_substrate = K_D_substrate_nM * 1e-9  # M
dG_bind_total = RT_val * np.log(K_D_substrate)  # -9.56 kcal/mol

# --- Konformationsänderung: Open -> Closed ---
# GESCHÄTZT: Kein publizierter dG(open->closed)!
# Tentakelrotation ~12 Grad (Lundin 2004 SAXS)
# Typisch für kleine Domänenbewegungen: 1-3 kcal/mol
# Open ist intrinsisch stabil (Tentakel natürlich ausgestreckt)
dG_open_to_closed = 2.0  # kcal/mol (GESCHÄTZT, positiv = Closing kostet Energie)
dG_conf = -dG_open_to_closed  # -2.0 (Closed weniger stabil ohne Substrat)

# --- PFD:CPN Bindung (Referenz, nicht im Spiel) ---
# K_D(PFD:CPN) = 50 nM (Sahlan 2010, archaeal SPR)
K_D_cpn_nM = 50
dG_bind_cpn = RT_val * np.log(K_D_cpn_nM * 1e-9)  # -9.95 kcal/mol

# --- Strukturdaten ---
# Hexamer: 2alpha + 4beta, ~84 Angstrom breit (Siegert 2000)
# Kein ATP/NTP involviert — rein thermodynamisch getrieben


# ================================================================
# Payoff-Konstruktion
# ================================================================

def build_payoffs(dG_conformational=None, f_pfd=0.50):
    """2x2 Payoff-Matrizen für PFD x Substrat.

    Konvention: höher = stabiler (= -dG)

    PFD:      {0=Open, 1=Closed}
    Substrat: {0=Free, 1=Bound}

    Bindungsenergie-Split: f_pfd für PFD, (1-f_pfd) für Substrat.
    Bei f=0.5: symmetrischer hydrophober Kontakt.
    """
    if dG_conformational is None:
        dG_conformational = dG_conf

    bind_total = -dG_bind_total  # positiv = günstig
    bind_pfd = bind_total * f_pfd
    bind_sub = bind_total * (1 - f_pfd)

    u_P = np.zeros((2, 2))
    u_P[0, 0] = 0.0                              # Open, Free (Referenz)
    u_P[0, 1] = 0.0                              # Open, Bound (kein Griff möglich)
    u_P[1, 0] = dG_conformational                  # Closed, Free (Closing-Kosten)
    u_P[1, 1] = dG_conformational + bind_pfd       # Closed, Bound

    u_S = np.zeros((2, 2))
    u_S[0, 0] = 0.0          # Free, Open (Referenz)
    u_S[0, 1] = 0.0          # Free, Closed (PFD-Zustand irrelevant)
    u_S[1, 0] = 0.0          # Bound, Open (kein stabiler Komplex)
    u_S[1, 1] = bind_sub     # Bound, Closed (vor Aggregation geschützt)

    return u_P, u_S


def main():
    print("=" * 65)
    print("Prefoldin (PFD) -- Kalibriertes Nash-Modell (ATP-frei)")
    print("=" * 65)

    # --- Parameter ---
    print(f"\n--- Parameter (T = {T} Grad C, RT = {RT_val:.3f} kcal/mol) ---")
    print(f"  K_D(PFD:Substrat) = {K_D_substrate_nM} nM (GESCHAETZT)")
    print(f"  dG(Bind) = {dG_bind_total:.2f} kcal/mol")
    print(f"  dG(Open->Closed) = +{dG_open_to_closed:.1f} kcal/mol (GESCHAETZT)")
    print(f"  Open ist intrinsisch stabiler als Closed")
    print(f"  ATP-Verbrauch: 0 (rein thermodynamisch)")
    print(f"  [Referenz: K_D(PFD:CPN) = {K_D_cpn_nM} nM, dG = {dG_bind_cpn:.2f}]")

    # --- Payoffs ---
    u_P, u_S = build_payoffs()
    st_P = ["Open", "Closed"]
    st_S = ["Free", "Bound"]

    print(f"\n--- Payoff-Matrizen ---")
    print(f"\n  u_PFD[pfd, sub]:")
    print(f"            {'Free':>10s}  {'Bound':>10s}")
    for i, s in enumerate(st_P):
        print(f"    {s:<10s}  {u_P[i,0]:10.2f}  {u_P[i,1]:10.2f}")

    print(f"\n  u_Sub[sub, pfd]:")
    print(f"            {'Open':>10s}  {'Closed':>10s}")
    for j, s in enumerate(st_S):
        print(f"    {s:<10s}  {u_S[j,0]:10.2f}  {u_S[j,1]:10.2f}")

    # --- PG-Test ---
    pg = four_cycle_test_2x2(u_P, u_S)
    print(f"\n--- Vier-Zyklen-Test ---")
    print(f"  I_PFD = {pg['I_1']:.4f}")
    print(f"  I_Sub = {pg['I_2']:.4f}")
    print(f"  Verletzung = {pg['violation']:.4f} kcal/mol")
    print(f"  PG = {pg['is_PG']}")

    # --- Nash-GG ---
    nash = find_nash_pure(u_P, u_S, 2, st_P, st_S)
    print(f"\n--- Nash-Gleichgewichte ---")
    print(f"  Anzahl reine NE: {len(nash)}")
    for ne in nash:
        print(f"    PFD={ne['player_1']}, Sub={ne['player_2']}")
        print(f"      u_P={ne['u_1']:.2f}, u_S={ne['u_2']:.2f}, Total={ne['u_total']:.2f}")

    # --- Potential ---
    if pg['is_PG']:
        pot = compute_potential_2x2(u_P, u_S)
        print(f"\n--- Potential-Funktion ---")
        for k, v in pot['values'].items():
            print(f"    Phi{k} = {v:.4f}")
        print(f"    Potential-Maximum: Phi{list(pot['values'].keys())[pot['argmax_idx']]}")

    # --- Goloubinoff S1-S4 ---
    sb = symmetry_breaking_analysis(u_P, u_S, 2)
    print(f"\n--- Goloubinoff S1-S4 ---")
    print(f"  S1 (Spieler-Symmetrie): gebrochen={sb['S1_player_symmetry']['broken']}")
    print(f"  S4 (Potential Game): {sb['S4_potential_game']['is_PG']}")

    # --- Biologische Interpretation ---
    print(f"\n--- Biologische Interpretation ---")
    if len(nash) == 2:
        print(f"  ZWEI NE: funktionelle Bistabilitaet")
        for ne in nash:
            label = ""
            if ne['player_1'] == 'Open' and ne['player_2'] == 'Free':
                label = " = Warte-Zustand (PFD bereit fuer neues Substrat)"
            elif ne['player_1'] == 'Closed' and ne['player_2'] == 'Bound':
                label = " = Holdase-Komplex (Substrat vor Aggregation geschuetzt)"
            print(f"    ({ne['player_1']}, {ne['player_2']}){label}")
        print(f"  Beide NE sind funktionell relevant im PFD-Zyklus:")
        print(f"    Open/Free -> Substrat-Einfang -> Closed/Bound -> TRiC-Transfer -> Open/Free")
        print(f"  KEIN ATP noetig fuer Uebergang zwischen NE!")
    elif len(nash) == 1:
        ne = nash[0]
        print(f"  Eindeutiges NE: ({ne['player_1']}, {ne['player_2']})")

    # --- Stabilitätsscan ---
    print(f"\n--- Stabilitaetsscan: dG(Open->Closed) variieren ---")
    print(f"  {'dG(O->C)':>10s}  {'dG_conf':>10s}  {'NE':>4s}  {'PG?':>4s}  Profil")

    scan_results = []
    for dg_oc in np.arange(-2.0, 5.1, 0.5):
        dg_c = -dg_oc
        u_Ps, u_Ss = build_payoffs(dG_conformational=dg_c)
        pg_s = four_cycle_test_2x2(u_Ps, u_Ss)
        nash_s = find_nash_pure(u_Ps, u_Ss, 2, st_P, st_S)

        profil = ", ".join(f"({n['player_1']},{n['player_2']})" for n in nash_s)

        marker = ""
        if abs(dg_oc - dG_open_to_closed) < 0.01:
            marker = " <-- aktuell"

        print(f"  {dg_oc:10.1f}  {dg_c:10.1f}  {len(nash_s):4d}  "
              f"{'J' if pg_s['is_PG'] else 'N':>4s}  {profil}{marker}")

        scan_results.append({
            "dG_open_to_closed": round(float(dg_oc), 1),
            "dG_conf": round(float(dg_c), 1),
            "n_NE": len(nash_s),
            "is_PG": pg_s["is_PG"],
            "profiles": profil,
        })

    # --- Split-Scan ---
    print(f"\n--- Split-Scan: Bindungsenergieverteilung ---")
    print(f"  {'f_PFD':>8s}  {'Verl.':>8s}  {'NE':>4s}  {'PG?':>4s}")
    split_results = []
    for f in np.arange(0.3, 0.71, 0.05):
        u_Ps, u_Ss = build_payoffs(f_pfd=f)
        pg_s = four_cycle_test_2x2(u_Ps, u_Ss)
        nash_s = find_nash_pure(u_Ps, u_Ss, 2, st_P, st_S)
        print(f"  {f:8.2f}  {pg_s['violation']:8.4f}  {len(nash_s):4d}  "
              f"{'J' if pg_s['is_PG'] else 'N':>4s}")
        split_results.append({
            "f_pfd": round(float(f), 2),
            "violation": pg_s["violation"],
            "n_NE": len(nash_s),
            "is_PG": pg_s["is_PG"],
        })

    # --- Vergleich ---
    print(f"\n--- Vergleich: PFD vs. andere ATP-freie Systeme ---")
    print(f"  TF (C4):   1 NE, non-PG, kein ATP, Holdase (ribosomgebunden)")
    print(f"  PFD (C5):  {len(nash)} NE, {'PG' if pg['is_PG'] else 'non-PG'}, kein ATP, Holdase (frei)")
    print(f"  XCL1:      2 NE, PG, kein ATP, Metamorphes Protein")
    print(f"  MAD2:      1 NE, non-PG, kein ATP (TRIP13: ja), Metamorph")

    # --- Ehrlichkeitswarnung ---
    print(f"\n--- EHRLICHKEITSWARNUNG ---")
    print(f"  K_D(PFD:Substrat): GESCHAETZT (kein publizierter Wert)")
    print(f"  dG(Open->Closed): GESCHAETZT (kein publizierter Wert)")
    print(f"  NE-Ergebnis abhaengig von Vorzeichen von dG_conf:")
    print(f"    dG_conf < 0 (Open stabiler) -> 2 NE")
    print(f"    dG_conf > 0 (Closed stabiler) -> 1 NE")
    print(f"  PG/non-PG abhaengig von f_pfd Split (Tautologie)")
    print(f"  Dieses Modell FORMALISIERT, sagt aber nichts VORHER.")

    # --- Ergebnisse ---
    results = {
        "system": "Prefoldin (PFD, ATP-freier Holdase)",
        "type": "holdase_atp_free",
        "temperature_C": T,
        "RT_kcal_mol": round(float(RT_val), 4),
        "experimental_parameters": {
            "K_D_substrate_nM": K_D_substrate_nM,
            "K_D_substrate_nM_status": "GESCHAETZT (kein publizierter Wert)",
            "K_D_substrate_nM_range": "10-300 nM (aus k_off Zako 2005 + typischem k_on)",
            "dG_bind_total_kcal_mol": round(float(dG_bind_total), 2),
            "dG_open_to_closed_kcal_mol": dG_open_to_closed,
            "dG_open_to_closed_status": "GESCHAETZT (kein publizierter Wert)",
            "dG_open_to_closed_basis": "12 Grad Tentakelrotation (Lundin 2004), typisch 1-3 kcal/mol",
            "ATP_per_cycle": 0,
            "K_D_cpn_nM": K_D_cpn_nM,
            "dG_bind_cpn_kcal_mol": round(float(dG_bind_cpn), 2),
        },
        "references": {
            "Zako_2005": "Zako et al. 2005, FEBS Lett 579:3718 (k_off PFD:CS/GFP)",
            "Lundin_2004": "Lundin et al. 2004, PNAS 101:15524 (Tentakelrotation SAXS)",
            "Sahlan_2010": "Sahlan et al. 2010, J Mol Biol 399:628 (K_D PFD:CPN SPR)",
            "Gestaut_2019": "Gestaut et al. 2019, Science 366:1190 (Cryo-EM PFD-TRiC)",
            "Siegert_2000": "Siegert et al. 2000, Cell 103:621 (PFD-Struktur)",
            "Vainberg_1998": "Vainberg et al. 1998, Cell 93:863 (PFD-Funktion)",
        },
        "payoff_matrices": {
            "u_PFD": u_P.tolist(),
            "u_Sub": u_S.tolist(),
            "states_PFD": st_P,
            "states_Sub": st_S,
        },
        "four_cycle_test": pg,
        "nash_equilibria": nash,
        "n_NE": len(nash),
        "goloubinoff_symmetry": sb,
        "stability_scan": scan_results,
        "split_scan": split_results,
        "interpretation": {
            "is_holdase": True,
            "has_ATP": False,
            "NE_count": len(nash),
            "NE_profiles": [f"({n['player_1']}, {n['player_2']})" for n in nash],
            "key_finding": (
                "PFD zeigt 2 NE bei geschaetztem dG_conf < 0 (Open intrinsisch stabil). "
                "Dies waere das ZWEITE 2-NE-System neben XCL1 — aber haengt "
                "vollstaendig vom Vorzeichen eines NICHT gemessenen Parameters ab. "
                "Der Stabilitaetsscan zeigt die 2->1 NE Transition bei dG_conf = 0."
            ),
            "honesty_caveat": (
                "BEIDE zentralen Parameter (K_D, dG_conf) sind geschaetzt. "
                "Das Modell formalisiert die Holdase-Biologie spieltheoretisch, "
                "sagt aber nichts vorher, was nicht bereits aus den Inputs folgt. "
                "2 NE iff Open intrinsisch stabil — das ist die Input-Annahme, "
                "nicht ein emergentes Ergebnis."
            ),
        },
    }

    out = save_results(results, __file__, "prefoldin_calibrated_results.json")
    print(f"\n-> Ergebnisse: {out}")

    print(f"\n{'=' * 65}")
    print(f"ERGEBNIS: Prefoldin (ATP-frei, Holdase) = "
          f"{'PG' if pg['is_PG'] else 'non-PG'}, {len(nash)} NE")
    for ne in nash:
        print(f"  NE: ({ne['player_1']}, {ne['player_2']})")
    print(f"  WARNUNG: Ergebnis basiert auf geschaetzten Parametern")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
