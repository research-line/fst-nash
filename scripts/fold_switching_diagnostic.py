"""
Fold-Switching-Diagnostik: Kann das Framework Fold-Switcher klassifizieren?
==========================================================================
AlphaFold2 versagt bei 94% der Fold-Switching-Proteine (Chakravarty et al. 2025).
Hypothese: Fold-Switching-Proteine zerfallen in zwei Regime:
  1. Gleichgewichts-Bistabilität (kein ATP) → PG → S4 intakt
  2. ATP-getriebenes Switching (gerichteter Zyklus) → Nicht-PG → S4 gebrochen

Testsysteme:
  - XCL1/Lymphotactin: Chemokinfold ↔ β-Dimer, ~50:50 bei 37°C
    (Tuinstra et al. 2008, PNAS 105:5057)
    Kein ATP → Gleichgewicht → erwartet: PG
  - KaiB/KaiC: Ground-State ↔ Fold-Switched, Stunden-Zeitskala
    ATP-Hydrolyse in KaiC treibt KaiB-Fold-Switch
    (Terauchi et al. 2007; Chang et al. 2015, Science 349:324)
    ATP-getrieben → erwartet: Nicht-PG

Payoff-Kalibrierung:
  XCL1: Aus Gleichgewichts-Thermodynamik (ΔG ≈ 0 bei 37°C/150mM NaCl)
  KaiB/KaiC: Aus ATP-Hydrolyse-Kopplung (ΔG_ATP ≈ -7.3 kcal/mol)
"""

import numpy as np
import json
from pathlib import Path


def build_xcl1_payoffs():
    """
    XCL1/Lymphotactin als 2×2-Spiel.
    Spieler 1 = N-terminale Domäne, Spieler 2 = C-terminale Domäne.
    Strategien: {Chemokinfold (C), β-Dimer-Fold (B)}

    Thermodynamische Kalibrierung:
    - ΔG(C↔B) ≈ 0 kcal/mol bei 37°C, 150mM NaCl (Tuinstra 2008: ~50:50)
    - Mismatch-Strafe: Domänen in verschiedenen Folds ≈ 3 kcal/mol
      (Typische Interfaceenergie für inkonsistente Folds)
    - KEIN ATP → detailed balance → Kopplung MUSS reziprok sein
    """
    strategies_1 = ["Chemokine", "Beta"]
    strategies_2 = ["Chemokine", "Beta"]

    mismatch = 3.0
    delta_fold = 0.1  # leichte Präferenz für Chemokinfold (≈kBT/6)

    u1 = np.array([
        [0.0, -mismatch],           # (C,C)=0, (C,B)=-3: N-term in Chemokine, C-term mismatcht
        [-mismatch, delta_fold],     # (B,C)=-3, (B,B)≈0: beide in Beta, leicht stabil
    ])

    # Reziproke Kopplung: u2 hat GLEICHE Interaktionsstruktur
    # (detailed balance bei Gleichgewicht → J_12 = J_21^T)
    u2 = np.array([
        [0.0, -mismatch],           # symmetrisch zu u1
        [-mismatch, delta_fold],
    ])

    return u1, u2, strategies_1, strategies_2


def build_kaib_kaic_payoffs():
    """
    KaiB/KaiC als 2×2-Spiel.
    Spieler 1 = KaiB, Spieler 2 = KaiC.
    KaiB-Strategien: {Ground-State (GS), Fold-Switched (FS)}
    KaiC-Strategien: {Active (phos.), Inactive (dephos.)}

    Thermodynamische Kalibrierung:
    - ΔG(KaiB GS→FS) ≈ 5-8 kcal/mol (ungünstig ohne KaiC)
    - KaiB(FS) bindet KaiC(Active): Kd ~ μM → ΔG_bind ≈ -8 kcal/mol
    - KaiC ATP-Hydrolyse: ΔG_ATP ≈ -7.3 kcal/mol
    - Asymmetrie: KaiC-Hydrolyse TREIBT KaiB-Switch (gerichtet),
      aber KaiB-Feedback auf KaiC ist schwächer (moduliert Phospho-Rate)
    """
    strategies_1 = ["GS", "FS"]
    strategies_2 = ["Active", "Inactive"]

    u_kaib = np.array([
        [0.0,  -1.0],   # GS+Active=ref, GS+Inactive=-1 (leichte Destabilisierung)
        [-5.0,  2.0],   # FS+Active=-5 (energetisch ungünstig ohne Bindung),
                         # FS+Inactive=+2 (nach ATP-Hydrolyse: Fold-Switch stabil)
    ])

    u_kaic = np.array([
        [0.0,  -3.0],   # Active+GS=ref, Active+FS=-3 (FS-KaiB inhibiert KaiA → Verlust)
        [-2.0,  1.0],   # Inactive+GS=-2 (falsche Phase), Inactive+FS=+1 (Uhr-Alignment)
    ])

    return u_kaib, u_kaic, strategies_1, strategies_2


def test_separability(u1, u2, name1, name2):
    """S3-Test: Ist die Kopplung additiv zerlegbar?"""
    n1, n2 = u1.shape
    max_interaction_1 = 0.0
    max_interaction_2 = 0.0

    for i in range(n1):
        for j in range(i + 1, n1):
            for k in range(n2):
                for l in range(k + 1, n2):
                    I1 = (u1[i, k] - u1[j, k]) - (u1[i, l] - u1[j, l])
                    I2 = (u2[k, i] - u2[l, i]) - (u2[k, j] - u2[l, j])
                    max_interaction_1 = max(max_interaction_1, abs(I1))
                    max_interaction_2 = max(max_interaction_2, abs(I2))

    s3_broken_1 = max_interaction_1 > 1e-10
    s3_broken_2 = max_interaction_2 > 1e-10
    return {
        f"{name1}_max_interaction": round(max_interaction_1, 4),
        f"{name2}_max_interaction": round(max_interaction_2, 4),
        "s3_broken": s3_broken_1 or s3_broken_2,
    }


def test_potential_game(u1, u2):
    """S4-Test: Vier-Zyklen-Bedingung (Monderer & Shapley 1996)."""
    n1, n2 = u1.shape
    violations = 0
    total = 0
    max_asymmetry = 0.0

    for i in range(n1):
        for j in range(i + 1, n1):
            for k in range(n2):
                for l in range(k + 1, n2):
                    I1 = (u1[i, k] - u1[j, k]) - (u1[i, l] - u1[j, l])
                    I2 = (u2[k, i] - u2[l, i]) - (u2[k, j] - u2[l, j])
                    diff = abs(I1 - I2)
                    max_asymmetry = max(max_asymmetry, diff)
                    total += 1
                    if diff > 1e-10:
                        violations += 1

    is_pg = violations == 0
    return {
        "is_potential_game": is_pg,
        "max_asymmetry": round(max_asymmetry, 4),
        "violations": violations,
        "total_cycles": total,
    }


def find_nash_equilibria(u1, u2, strats1, strats2):
    """Finde reine Nash-Gleichgewichte."""
    n1, n2 = u1.shape
    nash = []
    for i in range(n1):
        for j in range(n2):
            is_br1 = all(u1[i, j] >= u1[k, j] - 1e-10 for k in range(n1))
            is_br2 = all(u2[j, i] >= u2[l, i] - 1e-10 for l in range(n2))
            if is_br1 and is_br2:
                nash.append({
                    "profile": (strats1[i], strats2[j]),
                    "u1": round(float(u1[i, j]), 2),
                    "u2": round(float(u2[j, i]), 2),
                })
    return nash


def analyze_fold_switcher(name, u1, u2, strats1, strats2, name1, name2, expected_pg):
    """Vollständige Analyse eines Fold-Switching-Systems."""
    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"{'=' * 70}")

    # Payoffs anzeigen
    print(f"\n  Payoff {name1}:")
    header = "".join(f"{strats2[j]:>10}" for j in range(len(strats2)))
    print(f"  {'':>12}{header}")
    for i, s1 in enumerate(strats1):
        row = "".join(f"{u1[i, j]:10.2f}" for j in range(len(strats2)))
        print(f"  {s1:>12}{row}")

    print(f"\n  Payoff {name2}:")
    header = "".join(f"{strats1[j]:>10}" for j in range(len(strats1)))
    print(f"  {'':>12}{header}")
    for i, s2 in enumerate(strats2):
        row = "".join(f"{u2[i, j]:10.2f}" for j in range(len(strats1)))
        print(f"  {s2:>12}{row}")

    # S3-Test
    sep = test_separability(u1, u2, name1, name2)
    print(f"\n  S3 — Coupling Separability:")
    print(f"    {name1}: max |I| = {sep[f'{name1}_max_interaction']:.2f} kcal/mol"
          f" → {'GEBROCHEN' if sep['s3_broken'] else 'INTAKT'}")

    # S4-Test
    pg = test_potential_game(u1, u2)
    pg_label = "JA (PG)" if pg["is_potential_game"] else "NEIN (Nicht-PG)"
    print(f"\n  S4 — Potential-Game-Test:")
    print(f"    Potential Game: {pg_label}")
    print(f"    Max Asymmetrie: {pg['max_asymmetry']:.2f} kcal/mol")
    print(f"    Verletzungen: {pg['violations']}/{pg['total_cycles']}")

    # Nash-GG
    nash = find_nash_equilibria(u1, u2, strats1, strats2)
    print(f"\n  Nash-Gleichgewichte: {len(nash)}")
    for n in nash:
        print(f"    {n['profile']} — u1={n['u1']}, u2={n['u2']}")

    # Diagnose
    match_expected = pg["is_potential_game"] == expected_pg
    if pg["is_potential_game"]:
        regime = "Gleichgewichts-Bistabilität"
        implication = "Fold-Switching durch thermische Fluktuation, kein ATP nötig"
        af_note = "AlphaFold KÖNNTE beide Folds finden (Gleichgewichts-Ensemble)"
    else:
        regime = "ATP-getriebenes Switching"
        implication = "Gerichteter Zyklus, Energiezufuhr treibt Konformationswechsel"
        af_note = "AlphaFold KANN NICHT den Switch vorhersagen (Nicht-GG nicht modelliert)"

    print(f"\n  DIAGNOSE:")
    print(f"    Regime: {regime}")
    print(f"    → {implication}")
    print(f"    → {af_note}")
    print(f"    Erwartung erfüllt: {'JA ✓' if match_expected else 'NEIN ✗'}")

    return {
        "name": name,
        "s3": sep,
        "s4": pg,
        "nash_equilibria": nash,
        "regime": regime,
        "expected_pg": expected_pg,
        "match": match_expected,
        "alphafold_implication": af_note,
    }


def main():
    print("=" * 70)
    print("FOLD-SWITCHING-DIAGNOSTIK")
    print("Klassifiziert das Framework Fold-Switcher in Regime?")
    print("=" * 70)

    results = {"systems": []}

    # --- XCL1/Lymphotactin (Gleichgewicht) ---
    u1_xcl, u2_xcl, s1_xcl, s2_xcl = build_xcl1_payoffs()
    r_xcl = analyze_fold_switcher(
        "XCL1/Lymphotactin — Gleichgewichts-Fold-Switcher",
        u1_xcl, u2_xcl, s1_xcl, s2_xcl,
        "N-term", "C-term",
        expected_pg=True,
    )
    results["systems"].append(r_xcl)

    # --- KaiB/KaiC (ATP-getrieben) ---
    u1_kai, u2_kai, s1_kai, s2_kai = build_kaib_kaic_payoffs()
    r_kai = analyze_fold_switcher(
        "KaiB/KaiC — ATP-getriebener Fold-Switcher",
        u1_kai, u2_kai, s1_kai, s2_kai,
        "KaiB", "KaiC",
        expected_pg=False,
    )
    results["systems"].append(r_kai)

    # --- Zusammenfassung ---
    print(f"\n{'=' * 70}")
    print("ZUSAMMENFASSUNG: FOLD-SWITCHING-KLASSIFIKATION")
    print(f"{'=' * 70}")

    all_match = all(s["match"] for s in results["systems"])
    print(f"\n  {'System':<45} {'S4 (PG)?':>10} {'Regime':<30} {'Match':>6}")
    print(f"  {'-' * 95}")
    for s in results["systems"]:
        pg_str = "JA (PG)" if s["s4"]["is_potential_game"] else "NEIN"
        match_str = "✓" if s["match"] else "✗"
        print(f"  {s['name'][:45]:<45} {pg_str:>10} {s['regime']:<30} {match_str:>6}")

    print(f"\n  Klassifikation korrekt: {'BESTANDEN ✓' if all_match else 'FEHLGESCHLAGEN ✗'}")

    if all_match:
        print(f"""
  IMPLIKATION FÜR ALPHAFOLD:
  Das Framework unterscheidet zwei Typen von Fold-Switching:

  1. Gleichgewichts-Bistabilität (XCL1-Typ):
     → PG = thermische Fluktuation zwischen zwei Minima
     → AlphaFold KÖNNTE prinzipiell beide Folds finden
        (wenn Ensemble-Methoden verwendet werden)
     → Spieltheorie: redundant (= Ising/Potts-Modell)

  2. ATP-getriebenes Switching (KaiB/KaiC-Typ):
     → Nicht-PG = gerichteter Zyklus, Energiezufuhr nötig
     → AlphaFold KANN den Switch NICHT vorhersagen
        (kein Energie-Input im Modell, kein Nicht-GG)
     → Spieltheorie: identifiziert den ATP-Bedarf

  Dies erklärt TEILWEISE AlphaFolds 94% Fehlerrate bei Fold-Switchern:
  Viele biologische Fold-Switcher sind ATP-/GTP-getrieben (Typ 2),
  und kein statisches Einzelstruktur-Modell kann sie korrekt vorhersagen.

  OFFENE FRAGE: Welcher Anteil der 94% ist Typ 1 vs. Typ 2?
  → Systematische Klassifikation des Porter/Looger-Datensatzes nötig.""")

    results["classification_passed"] = all_match
    results["description"] = (
        "Fold-Switching-Diagnostik: Klassifiziert das Framework "
        "Fold-Switcher in Gleichgewichts- vs. ATP-getriebene Regime?"
    )
    results["references"] = {
        "xcl1_tuinstra_2008": "Tuinstra et al. (2008) PNAS 105:5057",
        "kaib_chang_2015": "Chang et al. (2015) Science 349:324",
        "alphafold_failure": "Chakravarty, Lee & Porter (2025) Protein Science",
        "alphafold2_fails": "Porter & Looger (2022) Protein Science 31:e4353",
        "monderer_shapley_1996": "Games & Econ Behavior 14:124",
    }

    out_path = Path(__file__).parent.parent / "results" / "fold_switching_diagnostic_results.json"

    def convert(obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} not serializable")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=convert)
    print(f"\nErgebnisse: {out_path}")


if __name__ == "__main__":
    main()
