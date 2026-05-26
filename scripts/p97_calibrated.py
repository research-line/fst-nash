"""
p97/VCP Kalibriertes Spieltheoretisches Modell
================================================
Kalibrierung aus Chou et al. (2014) JMB und Bhatt et al. (2015) PNAS.

Testfrage: Welche Goloubinoff-Symmetrien bricht p97/VCP?
  - Cofaktor-gesteuerte Substratselektion (Ufd1-Npl4, p47) => S2 gebrochen
  - Starke D1-D2 allosterische Kreuzkopplung (280x) => S4 wahrscheinlich gebrochen
  - => Regime 3 (Thermodynamisches Nicht-Gleichgewicht), wie Hsp90/Cdc37

Spieler:
  1. p97-Hexamer (P): {D1-dominant (D1), D2-dominant (D2)}
  2. Ubiquitiniertes Substrat (U): {Gebunden (Bound), Frei (Free)}

Payoff-Ableitung:
  p97 intrinsisch: D2-Ring katalytisch dominant (kcat ~7.5 min-1).
    D1-Ring stark von D2 reguliert (280x kcat/KM-Erhoehung durch D2-Nukleotid).
  Substrat intrinsisch: Ubiquitiniertes Protein am ER/Membran.
  Kopplung: Cofaktor-abhaengig — Ufd1-Npl4 vs p47 selektieren verschiedene
    Substrate. Diese Selektion ist NICHT reziprok (p97 profitiert anders
    von Substrat als Substrat von p97) => Delta_P != Delta_U => PG verletzt.

Status: Teilweise kalibriert.
  D2 ATPase: KALIBRIERT (Chou 2014, kcat = 7.5 min-1).
  D1-D2 Kreuzkopplung (280x): KALIBRIERT (Chou 2014).
  Cofaktor-Modulation: KALIBRIERT (p47 inhibiert, Ufd1-Npl4 nicht).
  S2-Brechung: GESCHAETZT (Cofaktor-Selektivitaet als Proxy).
  PG-Verletzung: FOLGT AUS KONSTRUKTION (Delta_P != Delta_U).

Referenzen:
  Chou et al. (2014) J Mol Biol 426:2886 — D1-D2 Kinetik
  Bhatt et al. (2015) PNAS — Cofaktor-Regulation
  Meyer & Bhatt (2023) Nat Rev Mol Cell Biol — p97 Review
  van den Boom & Meyer (2018) Front Mol Biosci 5:21 — VCP/p97 Cofaktoren
  Goloubinoff et al. (2022) Biomolecules 12:832 — S1-S4 Symmetrien
"""

import numpy as np
import json
from pathlib import Path


RT = 0.593  # kcal/mol bei 25 C

# ── Experimentelle Parameter ──────────────────────────────────────

# p97 ATPase (Chou et al. 2014, J Mol Biol 426:2886)
KCAT_D2 = 7.5               # min-1 (D2-Ring, katalytisch dominant)
KCAT_D2_WALKER_MUTANT = 0.29 # min-1 (K524A Walker-A Mutante, 22x reduziert)
KM_D2 = 0.5e-3              # ~0.5 mM (geschaetzt)

# D1-D2 Kreuzkopplung (Chou 2014)
# D2-Nukleotidbindung erhoeht D1 kcat/KM um 280x
# (kcat 7x hoch, KM 40x runter)
D1_D2_COUPLING = 280        # kcat/KM Verhaeltnis
D1_KCAT_FACTOR = 7           # kcat-Erhoehung durch D2
D1_KM_FACTOR = 40            # KM-Reduktion durch D2

# Cofaktor-Modulation (Bhatt et al. 2015, van den Boom 2018)
# p47: inhibiert ATPase signifikant
# Ufd1-Npl4: hemmt NICHT
# => Cofaktor bestimmt, welches Substrat erkannt wird (S2-analog)
P47_INHIBITION = 0.3         # Faktor (reduziert ATPase auf ~30%)
UFD1_NPL4_EFFECT = 1.0       # kein Effekt auf ATPase

# Substrat-Selektivitaet
# p47: ERAD-Substrate
# Ufd1-Npl4: ubiquitinierte Substrate (breiteres Spektrum)
COFACTOR_SELECTIVITY = P47_INHIBITION / UFD1_NPL4_EFFECT


def print_header():
    print("=" * 70)
    print("p97/VCP Kalibriertes Unfoldase-Modell")
    print("Experimentelle Daten: Chou (2014), Bhatt (2015)")
    print("=" * 70)


def derived_quantities():
    """Thermodynamische Groessen aus den Kinetik-Daten."""
    ddG_coupling = RT * np.log(D1_D2_COUPLING)
    ddG_cofactor = RT * np.log(1.0 / P47_INHIBITION)

    return {
        "D1_D2_coupling_factor": D1_D2_COUPLING,
        "ddG_D1_D2_coupling_kcal": round(ddG_coupling, 2),
        "cofactor_selectivity": round(COFACTOR_SELECTIVITY, 2),
        "ddG_cofactor_kcal": round(ddG_cofactor, 2),
        "D2_kcat_per_min": KCAT_D2,
    }


def print_derived():
    dq = derived_quantities()
    print("\n--- Abgeleitete thermodynamische Groessen ---")
    print(f"  p97 D2-Ring kcat: {KCAT_D2} min-1 (katalytisch dominant)")
    print(f"  D2 Walker-A Mutante: {KCAT_D2_WALKER_MUTANT} min-1 (22x reduziert)")
    print(f"  D1-D2 Kreuzkopplung: {D1_D2_COUPLING}x (kcat/KM)")
    print(f"    davon kcat {D1_KCAT_FACTOR}x, KM {D1_KM_FACTOR}x")
    print(f"  ddG(D1-D2): {dq['ddG_D1_D2_coupling_kcal']:.2f} kcal/mol")
    print(f"  Cofaktor-Selektivitaet: p47 = {P47_INHIBITION:.1f}x, "
          f"Ufd1-Npl4 = {UFD1_NPL4_EFFECT:.1f}x")
    print(f"  ddG(Cofaktor): {dq['ddG_cofactor_kcal']:.2f} kcal/mol")


# ── Payoff-Konstruktion ──────────────────────────────────────────


def build_payoffs(delta_P=1.5, delta_U=3.5):
    """Payoff-Matrizen fuer p97 vs. ubiquitiniertes Substrat.

    Strategien:
      p97: D1-dominant (D1), D2-dominant (D2)
      Substrat: Bound (B), Free (F)

    Asymmetrie-Logik (S2-Brechung via Cofaktoren):
      - p97 im D2-dominanten Modus profitiert moderat von gebundenem
        Substrat (delta_P) — ATPase wird nicht substrat-stimuliert
      - Substrat profitiert STARK von D2-dominantem p97 (delta_U)
        — wird entfaltet/extrahiert
      - Cofaktor-Selektion macht delta_P != delta_U
    """
    dq = derived_quantities()
    dG_coupling = dq["ddG_D1_D2_coupling_kcal"]

    u_P = np.zeros((2, 2))
    u_P[0, 0] = 0.0                         # (D1, B)
    u_P[0, 1] = 0.0                         # (D1, F)
    u_P[1, 0] = dG_coupling + delta_P       # (D2, B) — Cofaktor-Substrat
    u_P[1, 1] = dG_coupling                 # (D2, F) — kein Substrat

    u_U = np.zeros((2, 2))
    u_U[0, 0] = 0.0                         # (B, D1)
    u_U[0, 1] = delta_U                     # (B, D2) — Entfaltung/Extraktion
    u_U[1, 0] = 2.0                         # (F, D1) — frei, stabil
    u_U[1, 1] = 2.0                         # (F, D2) — frei, unabhaengig

    return u_P, u_U


# ── Potential-Game-Test ─────────────────────────────────────────────


def interaction_contrast(u):
    return u[0, 0] - u[0, 1] - u[1, 0] + u[1, 1]


def four_cycle_test(u_P, u_U):
    I_P = interaction_contrast(u_P)
    I_U = interaction_contrast(u_U)
    violation = abs(I_P - I_U)
    return I_P, I_U, violation, violation < 1e-10


# ── Nash-Gleichgewichte ──────────────────────────────────────────


def find_nash_equilibria(u_P, u_U):
    strategies_P = ["D1-dominant", "D2-dominant"]
    strategies_U = ["Bound", "Free"]
    nash = []
    for p in range(2):
        for u in range(2):
            p_best = (u_P[p, u] >= u_P[1 - p, u])
            u_best = (u_U[u, p] >= u_U[1 - u, p])
            if p_best and u_best:
                nash.append((strategies_P[p], strategies_U[u],
                             float(u_P[p, u]), float(u_U[u, p])))
    return nash


# ── Goloubinoff-Klassifikation ───────────────────────────────────


def goloubinoff_table(is_pg):
    print("\n--- Goloubinoff-Symmetrie-Klassifikation ---")
    print(f"  S1 (Binding):     Erhalten (p97 bindet diverse Ub-Substrate)")
    print(f"  S2 (Unbinding):   GEBROCHEN (Cofaktor-Selektion: p47 vs Ufd1-Npl4)")
    print(f"  S3 (Transition):  Erhalten (ATPase nicht substrat-stimuliert)")
    print(f"  S4 (PG/Reversi.): {'Erhalten' if is_pg else 'GEBROCHEN'} "
          f"(D1-D2-Kreuzkopplung {D1_D2_COUPLING}x => asymmetrische Wirkung)")
    print(f"  Regime: 3 — Thermodynamisches Nicht-Gleichgewicht (wie Hsp90)")


# ── Robustheit ───────────────────────────────────────────────────


def robustness_test():
    print("\n--- Robustheit: PG-Status vs. delta_P (delta_U = 3.5 fixiert) ---")
    for dp in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
        u_P, u_U = build_payoffs(delta_P=dp, delta_U=3.5)
        I_P, I_U, viol, is_pg = four_cycle_test(u_P, u_U)
        nash = find_nash_equilibria(u_P, u_U)
        tag = " <-- delta_P = delta_U (reziprok)" if abs(dp - 3.5) < 0.01 else ""
        print(f"  delta_P = {dp:4.1f}: I_P = {I_P:6.2f}, I_U = {I_U:6.2f}, "
              f"Verletzung = {viol:.2f}, PG={is_pg}, Nash={len(nash)} NE{tag}")


# ── Hauptprogramm ────────────────────────────────────────────────


def main():
    print_header()
    print_derived()

    delta_P = 1.5
    delta_U = 3.5
    u_P, u_U = build_payoffs(delta_P=delta_P, delta_U=delta_U)

    print(f"\n--- Payoff-Matrizen (delta_P = {delta_P}, delta_U = {delta_U}) ---")
    print(f"  u_P (p97):")
    print(f"  {'':15s} {'U=Bound':>12s} {'U=Free':>12s}")
    for i, s in enumerate(["D1-dominant", "D2-dominant"]):
        print(f"    {f'P={s}':<15s} {u_P[i,0]:12.4f} {u_P[i,1]:12.4f}")
    print(f"  u_U (Substrat):")
    print(f"  {'':15s} {'P=D1-dom':>12s} {'P=D2-dom':>12s}")
    for i, s in enumerate(["Bound", "Free"]):
        print(f"    {f'U={s}':<15s} {u_U[i,0]:12.4f} {u_U[i,1]:12.4f}")

    I_P, I_U, violation, is_pg = four_cycle_test(u_P, u_U)
    print(f"\n--- Potential-Game-Test ---")
    print(f"  I_P = {I_P:.4f}")
    print(f"  I_U = {I_U:.4f}")
    print(f"  Verletzung = {violation:.4f}")
    print(f"  PG = {is_pg}")
    if not is_pg:
        print(f"  => Kein Potential Game (Cofaktor-Selektion + D1-D2-Asymmetrie)")

    nash = find_nash_equilibria(u_P, u_U)
    print(f"\n--- Nash-Gleichgewichte ---")
    for sp, su, up, uu in nash:
        label = ""
        if sp == "D2-dominant" and su == "Bound":
            label = " -> Aktive Substrat-Verarbeitung"
        elif sp == "D1-dominant" and su == "Free":
            label = " -> Ruhezustand"
        print(f"  ({sp}, {su}): u_P={up:.4f}, u_U={uu:.4f}{label}")

    goloubinoff_table(is_pg)
    robustness_test()

    # ── JSON-Export ───────────────────────────────────────────────
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    outfile = results_dir / "p97_calibrated_results.json"

    dq = derived_quantities()
    results = {
        "modell": "p97/VCP AAA+ ATPase",
        "regime": 3,
        "regime_name": "Thermodynamisches Nicht-Gleichgewicht",
        "experimentelle_parameter": {
            "kcat_D2_per_min": KCAT_D2,
            "kcat_D2_mutant_per_min": KCAT_D2_WALKER_MUTANT,
            "D1_D2_coupling_factor": D1_D2_COUPLING,
            "D1_kcat_factor": D1_KCAT_FACTOR,
            "D1_KM_factor": D1_KM_FACTOR,
            "p47_inhibition": P47_INHIBITION,
            "ufd1_npl4_effect": UFD1_NPL4_EFFECT,
        },
        "abgeleitete_groessen": dq,
        "payoff_parameter": {
            "delta_P": delta_P,
            "delta_U": delta_U,
            "asymmetrie_motivation": (
                "Cofaktor-Selektion (p47 vs Ufd1-Npl4) macht Substratbindung "
                "konformationsabhaengig (S2-Brechung). p97 profitiert weniger "
                "von Substrat als Substrat von p97 => delta_P < delta_U."
            ),
        },
        "ergebnisse": {
            "I_P": float(I_P),
            "I_U": float(I_U),
            "four_cycle_violation": float(violation),
            "potential_game": True if is_pg else False,
            "nash_equilibria": [
                {"p97": sp, "substrat": su, "u_P": up, "u_U": uu}
                for sp, su, up, uu in nash
            ],
        },
        "kalibrierungsstatus": {
            "p97_D2_atpase": {"status": "KALIBRIERT", "quelle": "Chou 2014"},
            "D1_D2_coupling": {"status": "KALIBRIERT", "quelle": "Chou 2014 (280x)"},
            "cofaktor_modulation": {"status": "KALIBRIERT", "quelle": "Bhatt 2015"},
            "s2_brechung": {"status": "GESCHAETZT", "quelle": "Cofaktor-Selektivitaet als Proxy"},
            "delta_P": {"status": "GESCHAETZT", "quelle": "1.5 kcal/mol"},
            "delta_U": {"status": "GESCHAETZT", "quelle": "3.5 kcal/mol"},
            "pg_verletzung": {"status": "FOLGT AUS KONSTRUKTION",
                              "erklaerung": "PG=False folgt aus delta_P != delta_U"},
        },
        "schlussfolgerung": {
            "s2_gebrochen": True,
            "s4_pg_gebrochen": True if not is_pg else False,
            "regime": 3,
            "vergleich_hsp90": (
                "Aehnlich Hsp90/Cdc37: Cofaktor-gesteuerte S2-Brechung. "
                "Aber: bei Hsp90 ist es ein einzelner Cofaktor (Cdc37), "
                "bei p97 sind es multiple Cofaktoren mit verschiedenen Selektivitaeten."
            ),
            "einschraenkung": (
                "delta_P und delta_U sind geschaetzt. Die Cofaktor-Selektivitaet "
                "ist experimentell belegt, aber die genauen Kopplungsenergien "
                "zum Substrat sind nicht direkt gemessen."
            ),
        },
    }

    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n-> Ergebnisse: {outfile}")

    print(f"""
======================================================================
ERGEBNIS: p97/VCP teilweise kalibriert (Regime 3)
  PG: {is_pg} (Verletzung = {violation:.2f} kcal/mol)
  S2-Brechung: Cofaktor-Selektion (p47 vs Ufd1-Npl4) [GESCHAETZT]
  D1-D2-Kopplung: {D1_D2_COUPLING}x [KALIBRIERT]
  delta_P = {delta_P:.1f}, delta_U = {delta_U:.1f} (GESCHAETZT)
  Nash: {len(nash)} NE
  [EINSCHRAENKUNG] PG=False folgt aus delta_P != delta_U (Konstruktion)
======================================================================""")


if __name__ == "__main__":
    main()
