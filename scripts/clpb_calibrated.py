"""
ClpB/Hsp104 Kalibriertes Spieltheoretisches Modell
====================================================
Kalibrierung aus Werbeck et al. (2011), Mogk-Gruppe und weiteren Quellen.

Testfrage: Welche Goloubinoff-Symmetrien bricht ClpB?
  - ClpB hat moderate S3-Brechung (~1.6x Casein-Stimulation, T. thermophilus)
  - Asymmetrische NBD1/NBD2-Ringe (eine Groessenordnung verschieden) => S4 gebrochen
  - DnaK-Abhaengigkeit: ClpB allein disaggregationsinaktiv
  - => Regime 3 (Thermodynamisches Nicht-Gleichgewicht), wie GroEL

Spieler:
  1. ClpB-Hexamer (C): {Symmetric (S, beide Ringe gleich), Asymmetric (A, NBD1!=NBD2)}
  2. Aggregat-Substrat (G): {Aggregated (Agg), Dissolved (Dis)}

Payoff-Ableitung:
  ClpB intrinsisch: Hexamere ATPase mit zwei Ringen (NBD1, NBD2).
    NBD1 reguliert NBD2 trans (Werbeck 2011). Asymmetrische Dezeleration
    zwischen den Ringen aktiviert Disaggregation (Mogk-Gruppe).
  Substrat intrinsisch: Aggregierter Zustand ist kinetisch gefangen.
  Kopplung: Asymmetrisch — ClpB profitiert von ungefaltetem/aggregiertem
    Substrat (Translokationskopplung), aber Substrat profitiert
    UEBERPROPORTIONAL von asymmetrischem ClpB (disaggregiert).
    => Delta_H != Delta_S => PG verletzt.

Status: Teilweise kalibriert.
  ATPase-Raten (NBD1, NBD2): KALIBRIERT (Werbeck 2011, Mogk-Gruppe).
  S3-Brechung (moderat): KALIBRIERT (~1.6x Casein, T. thermophilus).
  Asymmetrie NBD1/NBD2: KALIBRIERT (~10x Unterschied, Mogk-Gruppe).
  PG-Verletzung: FOLGT AUS KONSTRUKTION (Delta_H != Delta_S).

Referenzen:
  Werbeck et al. (2011) Biochemistry 50:1523 — ClpB Oligomerisierung/Nukleotid
  Mogk, Bukau et al. (2003) J Biol Chem 278:17615 — Asymmetric deceleration
  Shorter & Lindquist (2004) Science 304:1793 — Hsp104 Disaggregation
  Doyle et al. (2007) Nat Struct Mol Biol 14:114 — ClpB Translokation
  Goloubinoff et al. (2022) Biomolecules 12:832 — S1-S4 Symmetrien
"""

import numpy as np
import json
from pathlib import Path


RT = 0.593  # kcal/mol bei 25 C

# ── Experimentelle Parameter ──────────────────────────────────────

# ClpB ATPase (Werbeck et al. 2011, T. thermophilus)
KCAT_BASAL = 3.9            # min-1 (Hexamer, basal)
KCAT_CASEIN = 6.2           # min-1 (+ Casein-Substrat)
STIMULATION_SUBSTRATE = KCAT_CASEIN / KCAT_BASAL  # ~1.6x (moderat)

# Hsp104 ATPase (Shorter Lab, Doyle et al. 2007)
KCAT_HSP104 = 72.0          # min-1 pro Protomer (= 1.2 s-1)
KM_ATP_HSP104 = 4.5e-3      # ~4.5 mM

# NBD1/NBD2 Asymmetrie (Mogk-Gruppe)
# NBD2 ist katalytisch dominant; NBD1 ~10x langsamer
KCAT_NBD2 = 3.0             # min-1 (T. therm., isoliert)
KCAT_NBD1 = 0.3             # min-1 (T. therm., geschaetzt aus trans-Regulation)
NBD_ASYMMETRY = KCAT_NBD2 / KCAT_NBD1  # ~10x

# Translokationsrate (Doyle et al. 2007)
TRANSLOCATION_RATE = 0.9    # Aminosaeuren/s
STEP_SIZE = 60              # Aminosaeuren pro ATP (sub-saturierend)

# DnaK-Abhaengigkeit
# ClpB allein: keine Disaggregation
# ClpB + DnaK/DnaJ/GrpE: volle Disaggregation
DNAK_REQUIRED = True


def print_header():
    print("=" * 70)
    print("ClpB/Hsp104 Kalibriertes Disaggregations-Modell")
    print("Experimentelle Daten: Werbeck (2011), Mogk/Bukau, Doyle (2007)")
    print("=" * 70)


def derived_quantities():
    """Thermodynamische Groessen aus den Kinetik-Daten."""
    s3_ratio = STIMULATION_SUBSTRATE
    ddG_S3 = RT * np.log(s3_ratio) if s3_ratio > 1 else 0.0
    ddG_asymm = RT * np.log(NBD_ASYMMETRY)

    return {
        "S3_ratio": round(s3_ratio, 2),
        "ddG_S3_kcal": round(ddG_S3, 2),
        "NBD_asymmetry": round(NBD_ASYMMETRY, 1),
        "ddG_asymmetry_kcal": round(ddG_asymm, 2),
        "translocation_aa_per_s": TRANSLOCATION_RATE,
    }


def print_derived():
    dq = derived_quantities()
    print("\n--- Abgeleitete thermodynamische Groessen ---")
    print(f"  ClpB ATPase kcat (basal): {KCAT_BASAL} min-1")
    print(f"  ClpB ATPase kcat (+Casein): {KCAT_CASEIN} min-1")
    print(f"  S3-Stimulation: {dq['S3_ratio']:.2f}x (moderat)")
    print(f"  ddG(S3): {dq['ddG_S3_kcal']:.2f} kcal/mol")
    print(f"  NBD2/NBD1 Asymmetrie: {dq['NBD_asymmetry']:.0f}x")
    print(f"  ddG(Asymmetrie): {dq['ddG_asymmetry_kcal']:.2f} kcal/mol")
    print(f"  Translokation: {TRANSLOCATION_RATE} AA/s")
    print(f"  DnaK-Abhaengigkeit: {'Ja (obligat)' if DNAK_REQUIRED else 'Nein'}")


# ── Payoff-Konstruktion ──────────────────────────────────────────


def build_payoffs(delta_C=2.0, delta_G=4.0, dG_agg=-3.0):
    """Payoff-Matrizen fuer ClpB vs. Aggregat.

    Strategien:
      ClpB: Symmetric (S), Asymmetric (A)
      Substrat: Aggregated (Agg), Dissolved (Dis)

    Asymmetrie-Logik:
      - ClpB im asymmetrischen Modus profitiert von Aggregat-Substrat
        um delta_C (Translokationskopplung, moderate S3)
      - Aggregat profitiert UEBERPROPORTIONAL vom asymmetrischen ClpB
        um delta_G (Disaggregation, starke Wirkung)
      - delta_C != delta_G => PG verletzt (asymmetrische Kopplung)
    """
    dq = derived_quantities()
    ddG_asymm = dq["ddG_asymmetry_kcal"]

    u_C = np.zeros((2, 2))
    u_C[0, 0] = 0.0                        # (S, Agg)
    u_C[0, 1] = 0.0                        # (S, Dis)
    u_C[1, 0] = ddG_asymm + delta_C        # (A, Agg) — Translokation aktiv
    u_C[1, 1] = ddG_asymm                  # (A, Dis) — kein Substrat-Bonus

    u_G = np.zeros((2, 2))
    u_G[0, 0] = 0.0                        # (Agg, S)
    u_G[0, 1] = delta_G                    # (Agg, A) — Disaggregation!
    u_G[1, 0] = -dG_agg                    # (Dis, S) — geloest, stabil
    u_G[1, 1] = -dG_agg                    # (Dis, A) — geloest, unabhaengig

    return u_C, u_G


# ── Potential-Game-Test ─────────────────────────────────────────────


def interaction_contrast(u):
    return u[0, 0] - u[0, 1] - u[1, 0] + u[1, 1]


def four_cycle_test(u_C, u_G):
    I_C = interaction_contrast(u_C)
    I_G = interaction_contrast(u_G)
    violation = abs(I_C - I_G)
    return I_C, I_G, violation, violation < 1e-10


# ── Nash-Gleichgewichte ──────────────────────────────────────────


def find_nash_equilibria(u_C, u_G):
    strategies_C = ["Symmetric", "Asymmetric"]
    strategies_G = ["Aggregated", "Dissolved"]
    nash = []
    for c in range(2):
        for g in range(2):
            c_best = (u_C[c, g] >= u_C[1 - c, g])
            g_best = (u_G[g, c] >= u_G[1 - g, c])
            if c_best and g_best:
                nash.append((strategies_C[c], strategies_G[g],
                             float(u_C[c, g]), float(u_G[g, c])))
    return nash


# ── Goloubinoff-Klassifikation ───────────────────────────────────


def goloubinoff_table(is_pg):
    print("\n--- Goloubinoff-Symmetrie-Klassifikation ---")
    print(f"  S1 (Binding):     Erhalten (ClpB bindet diverse Aggregate)")
    print(f"  S2 (Unbinding):   Erhalten (Freisetzung substratunabhaengig)")
    print(f"  S3 (Transition):  GEBROCHEN (moderat, {STIMULATION_SUBSTRATE:.1f}x Casein)")
    print(f"  S4 (PG/Reversi.): {'Erhalten' if is_pg else 'GEBROCHEN'} "
          f"(NBD1/NBD2-Asymmetrie {NBD_ASYMMETRY:.0f}x => delta_C != delta_G)")
    print(f"  Regime: 3 — Thermodynamisches Nicht-Gleichgewicht")
    print(f"  Besonderheit: DnaK/DnaJ/GrpE obligat fuer Disaggregation")


# ── Robustheit ───────────────────────────────────────────────────


def robustness_test():
    print("\n--- Robustheit: PG-Status vs. delta_C (delta_G = 4.0 fixiert) ---")
    for dc in [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]:
        u_C, u_G = build_payoffs(delta_C=dc, delta_G=4.0)
        I_C, I_G, viol, is_pg = four_cycle_test(u_C, u_G)
        nash = find_nash_equilibria(u_C, u_G)
        tag = " <-- delta_C = delta_G (reziprok)" if abs(dc - 4.0) < 0.01 else ""
        print(f"  delta_C = {dc:4.1f}: I_C = {I_C:6.2f}, I_G = {I_G:6.2f}, "
              f"Verletzung = {viol:.2f}, PG={is_pg}, Nash={len(nash)} NE{tag}")


# ── Hauptprogramm ────────────────────────────────────────────────


def main():
    print_header()
    print_derived()

    delta_C = 2.0
    delta_G = 4.0
    u_C, u_G = build_payoffs(delta_C=delta_C, delta_G=delta_G)

    print(f"\n--- Payoff-Matrizen (delta_C = {delta_C}, delta_G = {delta_G}) ---")
    print(f"  u_C (ClpB):")
    print(f"  {'':15s} {'G=Aggregated':>14s} {'G=Dissolved':>14s}")
    for i, s in enumerate(["Symmetric", "Asymmetric"]):
        print(f"    {f'C={s}':<15s} {u_C[i,0]:14.4f} {u_C[i,1]:14.4f}")
    print(f"  u_G (Aggregat):")
    print(f"  {'':15s} {'C=Symmetric':>14s} {'C=Asymmetric':>14s}")
    for i, s in enumerate(["Aggregated", "Dissolved"]):
        print(f"    {f'G={s}':<15s} {u_G[i,0]:14.4f} {u_G[i,1]:14.4f}")

    I_C, I_G, violation, is_pg = four_cycle_test(u_C, u_G)
    print(f"\n--- Potential-Game-Test ---")
    print(f"  I_C = {I_C:.4f}")
    print(f"  I_G = {I_G:.4f}")
    print(f"  Verletzung = {violation:.4f}")
    print(f"  PG = {is_pg}")
    if not is_pg:
        print(f"  => Kein Potential Game (asymmetrische NBD1/NBD2-Kopplung)")

    nash = find_nash_equilibria(u_C, u_G)
    print(f"\n--- Nash-Gleichgewichte ---")
    for sc, sg, uc, ug in nash:
        label = ""
        if sc == "Asymmetric" and sg == "Aggregated":
            label = " -> Aktive Disaggregation"
        elif sc == "Symmetric" and sg == "Dissolved":
            label = " -> Ruhezustand"
        print(f"  ({sc}, {sg}): u_C={uc:.4f}, u_G={ug:.4f}{label}")

    goloubinoff_table(is_pg)
    robustness_test()

    # ── JSON-Export ───────────────────────────────────────────────
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    outfile = results_dir / "clpb_calibrated_results.json"

    dq = derived_quantities()
    results = {
        "modell": "ClpB/Hsp104 Disaggregase",
        "regime": 3,
        "regime_name": "Thermodynamisches Nicht-Gleichgewicht",
        "experimentelle_parameter": {
            "kcat_basal_per_min": KCAT_BASAL,
            "kcat_casein_per_min": KCAT_CASEIN,
            "stimulation_factor": float(STIMULATION_SUBSTRATE),
            "kcat_hsp104_per_min": KCAT_HSP104,
            "nbd_asymmetry_factor": float(NBD_ASYMMETRY),
            "translocation_aa_per_s": TRANSLOCATION_RATE,
            "dnak_required": DNAK_REQUIRED,
        },
        "abgeleitete_groessen": dq,
        "payoff_parameter": {
            "delta_C": delta_C,
            "delta_G": delta_G,
            "asymmetrie_motivation": (
                "delta_C < delta_G weil Substrat ueberproportional von "
                "Disaggregation profitiert (kinetisch gefangener Zustand befreit)"
            ),
        },
        "ergebnisse": {
            "I_C": float(I_C),
            "I_G": float(I_G),
            "four_cycle_violation": float(violation),
            "potential_game": True if is_pg else False,
            "nash_equilibria": [
                {"clpb": sc, "aggregat": sg, "u_C": uc, "u_G": ug}
                for sc, sg, uc, ug in nash
            ],
        },
        "kalibrierungsstatus": {
            "clpb_atpase": {"status": "KALIBRIERT", "quelle": "Werbeck 2011"},
            "nbd_asymmetrie": {"status": "KALIBRIERT", "quelle": "Mogk-Gruppe (~10x)"},
            "s3_brechung": {"status": "KALIBRIERT", "quelle": "1.6x Casein (moderat)"},
            "delta_C": {"status": "GESCHAETZT", "quelle": "2.0 kcal/mol (Translokation)"},
            "delta_G": {"status": "GESCHAETZT", "quelle": "4.0 kcal/mol (Disaggregation)"},
            "pg_verletzung": {"status": "FOLGT AUS KONSTRUKTION",
                              "erklaerung": "PG=False folgt aus delta_C != delta_G"},
        },
        "schlussfolgerung": {
            "s3_gebrochen": True,
            "s4_pg_gebrochen": True if not is_pg else False,
            "regime": 3,
            "besonderheit": "DnaK/DnaJ/GrpE obligat; ClpB allein inaktiv",
            "vergleich_groel": (
                "Aehnlich GroEL: asymmetrische Kopplung bricht PG. "
                "Aber: GroEL hat cis/trans-Ring-Asymmetrie, "
                "ClpB hat NBD1/NBD2-Ring-Asymmetrie."
            ),
            "einschraenkung": (
                "delta_C und delta_G sind geschaetzt. Die Asymmetrie ist "
                "experimentell belegt (NBD1/NBD2 ~10x), aber die genauen "
                "Kopplungsenergien zum Substrat sind nicht gemessen."
            ),
        },
    }

    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n-> Ergebnisse: {outfile}")

    print(f"""
======================================================================
ERGEBNIS: ClpB/Hsp104 teilweise kalibriert (Regime 3)
  PG: {is_pg} (Verletzung = {violation:.2f} kcal/mol)
  S3-Brechung: {STIMULATION_SUBSTRATE:.1f}x (moderat) [KALIBRIERT]
  NBD1/NBD2-Asymmetrie: {NBD_ASYMMETRY:.0f}x [KALIBRIERT]
  delta_C = {delta_C:.1f}, delta_G = {delta_G:.1f} (GESCHAETZT)
  Nash: {len(nash)} NE
  Besonderheit: DnaK/DnaJ/GrpE obligat fuer Disaggregation
  [EINSCHRAENKUNG] PG=False folgt aus delta_C != delta_G (Konstruktion)
======================================================================""")


if __name__ == "__main__":
    main()
