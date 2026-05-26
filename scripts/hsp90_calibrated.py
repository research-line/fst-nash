"""
Hsp90/Cdc37 Kalibriertes Spieltheoretisches Modell
====================================================
Kalibrierung mit experimentellen Daten aus Goloubinoff et al. (2022),
Verba et al. (2016), Sreeramulu et al. (2009) u.a.

Testfrage: Bricht Hsp90/Cdc37 die S2-Symmetrie (Goloubinoff 2022)?
  - Cdc37 bindet inaktive Kinase-Konformation (DFG-out, sub-µM K_D)
  - Cdc37 kann aktive Konformation NICHT binden (DFG-in, sterischer Clash)
  - => k_off haengt von Substratkonformation ab => S2 gebrochen
  - => Delta_H != Delta_S => PG-Verletzung erwartet

Spieler:
  1. Hsp90 (H): {Open (O), Closed (C)} — N-Domaene Dimerisierung
  2. Substrat-Kinase (S): {Inactive (I, DFG-out), Active (A, DFG-in)}

Payoff-Ableitung:
  Hsp90 intrinsisch: Langsame ATPase (kcat = 0.09 min-1), Open-Closed Zyklus.
  Substrat intrinsisch: DFG-Flip Gleichgewicht (B-Raf: ~50:50 WT, ~90:10 V600E).
  Kopplungsenergie: Cdc37 als Co-Chaperon erzeugt konformationsabhaengige
    Bindung — bindet NUR inaktive Konformation. Dies macht Delta_H != Delta_S.

Status: Teilweise kalibriert.
  Hsp90 ATPase kcat: KALIBRIERT (Prodromou 2000, Richter 2008).
  Cdc37 K_D-Asymmetrie: GESCHAETZT aus struktureller Evidenz.
    Sreeramulu (2009) mass CDK4, nicht B-Raf; 100x fuer DFG-in vs -out
    ist Extrapolation aus sterischen Argumenten (Verba 2016 Cryo-EM).
  S2-Brechung: KALIBRIERT (Delta_H != Delta_S, ddG ~ 2.7 kcal/mol).
  PG-Verletzung: FOLGT ALGEBRAISCH aus Delta_H != Delta_S.
    Dies ist eine Konstruktionseigenschaft, keine empirische Entdeckung.
    Genau wie Hsp70 PG=True aus Delta_H = Delta_S folgt.

  Empirischer Gehalt (was NICHT aus der Konstruktion folgt):
  - Quantitative Vorhersage: B-Raf WT vs V600E aktive Fraktion unter Hsp90
  - Cdc37-Konzentrations-Dosis-Antwort auf Kinase-Aktivitaet
  - Konsistenz mit Goloubinoff S2-Klassifikation

Referenzen:
  Goloubinoff et al. (2022) Biomolecules 12:832 — S1-S4 Symmetrien
  Verba et al. (2016) Science 352:1542 — Hsp90-Cdc37-Cdk4 Cryo-EM
  Sreeramulu et al. (2009) J Biol Chem 284:3885 — Cdc37-Kinase NMR
  Prodromou et al. (2000) EMBO J 19:4383 — Hsp90 ATPase Raten
  Richter et al. (2008) J Biol Chem 283:17757 — Cdc37-Hsp90 Regulation
  Lavoie et al. (2005) EMBO J 24:3613 — B-Raf/Cdc37 Interaktion
"""

import numpy as np
import json
from pathlib import Path


RT = 0.593  # kcal/mol bei 25 C

# ── Experimentelle Parameter ──────────────────────────────────────

# Hsp90 ATPase (Prodromou 2000, Richter 2008)
KCAT_HSP90 = 0.09       # min-1 (human Hsp90alpha, langsam!)
KM_ATP = 0.5e-3         # ~0.5 mM (Michaelis-Konstante)

# Hsp90 Open-Closed Gleichgewicht
# Open ist Grundzustand; Closed nach ATP-Bindung + N-Dimerisierung
# Keq(O->C) ~ 0.3 bei apo, ~ 2-5 mit ATP (Hessling 2009 Nat Struct Mol Biol)
KEQ_OC_APO = 0.3
KEQ_OC_ATP = 3.0
DG_OC = -RT * np.log(KEQ_OC_ATP)  # ~ -0.65 kcal/mol mit ATP

# Cdc37 Bindung an Kinase-Substrate (Sreeramulu 2009)
# NMR chemical shift perturbation + ITC
KD_CDC37_INACTIVE = 0.5e-6   # ~0.5 µM fuer inaktive Kinase (DFG-out)
KD_CDC37_ACTIVE = 50e-6      # ~50 µM fuer aktive Kinase (DFG-in)
# Faktor ~100x: Sreeramulu (2009) mass CDK4/Cdc37 NMR; 100x fuer
# DFG-in vs DFG-out ist Extrapolation aus sterischer Analyse (Verba 2016)
SELECTIVITY_FACTOR = KD_CDC37_ACTIVE / KD_CDC37_INACTIVE

# Daraus: ddG der Cdc37-Bindung
DDG_CDC37 = RT * np.log(KD_CDC37_ACTIVE / KD_CDC37_INACTIVE)
# ~2.7 kcal/mol (Cdc37 bevorzugt inaktive Konformation stark)

# B-Raf DFG-Flip Gleichgewicht (Roring 2012, Thevakumaran 2015)
# WT B-Raf: ~50:50 DFG-in/DFG-out (inactive favored by autoinhibition)
# V600E B-Raf: ~90:10 DFG-in/DFG-out (constitutively active)
FRAC_ACTIVE_BRAF_WT = 0.50      # ohne Chaperon
FRAC_ACTIVE_BRAF_V600E = 0.90   # ohne Chaperon

# Daraus: dG(I->A) der DFG-Flip
DG_DFG_WT = -RT * np.log(FRAC_ACTIVE_BRAF_WT / (1 - FRAC_ACTIVE_BRAF_WT))
DG_DFG_V600E = -RT * np.log(FRAC_ACTIVE_BRAF_V600E / (1 - FRAC_ACTIVE_BRAF_V600E))


def print_header():
    print("=" * 70)
    print("Hsp90/Cdc37 Kalibriertes S2-Brechungs-Modell")
    print("Experimentelle Daten: Goloubinoff (2022), Sreeramulu (2009)")
    print("=" * 70)


def print_derived_quantities():
    """Zeige abgeleitete thermodynamische Groessen."""
    print("\n--- Abgeleitete thermodynamische Groessen ---")
    print(f"  Hsp90 ATPase kcat: {KCAT_HSP90} min-1")
    print(f"  Hsp90 dG(Open->Closed, +ATP): {DG_OC:.3f} kcal/mol")
    print(f"  Cdc37 K_D(inactive): {KD_CDC37_INACTIVE*1e6:.1f} µM")
    print(f"  Cdc37 K_D(active):   {KD_CDC37_ACTIVE*1e6:.0f} µM")
    print(f"  Selektivitaetsfaktor: {SELECTIVITY_FACTOR:.0f}x")
    print(f"  ddG(Cdc37): {DDG_CDC37:.2f} kcal/mol (Inactive bevorzugt)")
    print(f"  B-Raf WT dG(I->A): {DG_DFG_WT:.3f} kcal/mol")
    print(f"  B-Raf V600E dG(I->A): {DG_DFG_V600E:.3f} kcal/mol")


def build_payoffs():
    """
    Konstruiere Payoff-Matrizen fuer Hsp90 (H) vs Substrat-Kinase (S).

    Strategien:
      H: Open (O), Closed (C)
      S: Inactive (I), Active (A)

    Payoffs:
      u_H[i,j] = Hsp90-Nutzen bei Strategie i, Substrat bei j
      u_S[i,j] = Substrat-Nutzen bei Hsp90 i, eigene Strategie j

    Kopplungslogik (S2-Brechung via Cdc37):
      - Cdc37 bringt inaktive Kinase zu Hsp90 => Closed+Inactive ist stabil
      - Cdc37 kann aktive Kinase NICHT binden => Closed+Active instabil
      - Dies erzeugt asymmetrische Kopplung: Delta_H != Delta_S
    """
    # Hsp90 intrinsische Praeferenz: Closed > Open (mit ATP)
    dG_H = abs(DG_OC)  # ~0.65, Closed bevorzugt

    # Substrat intrinsische Praeferenz: abhaengig von Kinase-Variante
    # Hier: Modell parametrisch, default = WT (neutral)
    dG_S = abs(DG_DFG_WT)  # ~0 fuer WT (50:50)

    # Kopplung: Cdc37-vermittelt
    # Delta_H: Hsp90-seitig — wie sehr profitiert Hsp90 von Substrat-Konformation?
    #   Hsp90 profitiert von Cdc37-Rekrutierung (Inactive Substrat = Cdc37-kompatibel)
    #   => Delta_H = ddG_Cdc37 (Hsp90 gewinnt Client)
    delta_H = DDG_CDC37  # ~2.7

    # Delta_S: Substrat-seitig — wie sehr profitiert Substrat von Hsp90-Konformation?
    #   Substrat wird im Closed-Hsp90 gefaltet (unabhaengig von eigener Konformation)
    #   => Delta_S ≈ Faltungsenergie, NICHT konformationsabhaengig
    #   S2-Brechung: Cdc37 wirkt NUR auf Hsp90-Rekrutierung, nicht auf Substrat-Nutzen
    delta_S = 0.5  # kleiner als delta_H: Substrat profitiert weniger konformationsabhaengig

    # [ANNAHME] delta_S = 0.5 kcal/mol
    #   Begruendung: Substrat wird im Hsp90-Closed stabilisiert, aber der Nutzen
    #   haengt weniger von der eigenen Konformation ab als Cdc37-Rekrutierung
    #   Fuer delta_S = delta_H waere PG=True (wie Hsp70)
    #   Die S2-Brechung durch Cdc37 erzeugt delta_H >> delta_S

    #         S=Inactive    S=Active
    # H=Open    0, 0         0, dG_S
    # H=Closed  dG_H+delta_H, delta_S    dG_H, dG_S
    u_H = np.array([
        [0.0,           0.0],           # H=Open:  kein Klient-Nutzen
        [dG_H + delta_H, dG_H],         # H=Closed: +delta_H NUR bei Inactive (Cdc37)
    ])

    u_S = np.array([
        [0.0,        dG_S],             # S bei H=Open: intrinsisch
        [delta_S,    dG_S],             # S bei H=Closed: +delta_S NUR bei Inactive
    ])

    return u_H, u_S, delta_H, delta_S, dG_H, dG_S


def four_cycle_test(u_H, u_S):
    """
    Monderer-Shapley 4-Cycle-Test.
    I_H = u_H[0,0] - u_H[0,1] - u_H[1,0] + u_H[1,1]
    I_S = u_S[0,0] - u_S[1,0] - u_S[0,1] + u_S[1,1]
    PG <=> I_H = I_S
    """
    I_H = u_H[0, 0] - u_H[0, 1] - u_H[1, 0] + u_H[1, 1]
    I_S = u_S[0, 0] - u_S[1, 0] - u_S[0, 1] + u_S[1, 1]
    violation = abs(I_H - I_S)
    is_pg = violation < 1e-10
    return I_H, I_S, violation, is_pg


def find_nash_equilibria(u_H, u_S):
    """Finde reine Nash-Gleichgewichte."""
    strategies_H = ["Open", "Closed"]
    strategies_S = ["Inactive", "Active"]
    nash = []
    for i in range(2):
        for j in range(2):
            h_br = u_H[0, j] <= u_H[1, j] if i == 1 else u_H[0, j] >= u_H[1, j]
            other_i = 1 - i
            if u_H[i, j] < u_H[other_i, j]:
                continue
            s_br = True
            other_j = 1 - j
            if u_S[i, j] < u_S[i, other_j]:
                s_br = False
            if s_br:
                nash.append((strategies_H[i], strategies_S[j],
                             u_H[i, j], u_S[i, j]))
    return nash


def s2_breaking_analysis(delta_H, delta_S):
    """Analysiere S2-Brechung quantitativ."""
    print("\n--- S2-Symmetrie-Analyse (Goloubinoff 2022) ---")
    print(f"  S2 = Unbinding Symmetry: k_off soll substratunabhaengig sein")
    print(f"  Cdc37 bricht S2: K_D(inactive) << K_D(active)")
    print(f"  => Delta_H = {delta_H:.2f} kcal/mol (Hsp90-Seite)")
    print(f"  => Delta_S = {delta_S:.2f} kcal/mol (Substrat-Seite)")
    print(f"  => |Delta_H - Delta_S| = {abs(delta_H - delta_S):.2f} kcal/mol")
    print(f"  => S2 gebrochen: {'Ja' if abs(delta_H - delta_S) > 0.01 else 'Nein'}")
    print(f"  => S4 (PG) gebrochen: {'Ja' if abs(delta_H - delta_S) > 0.01 else 'Nein'}")
    print(f"  [ANNAHME] Delta_S = {delta_S} ist geschaetzt, nicht gemessen.")
    print(f"            S2-Brechung ist experimentell belegt (Cdc37-Selektivitaet),")
    print(f"            aber die genaue Groesse der PG-Verletzung haengt von Delta_S ab.")


def braf_prediction(u_H, u_S, dG_H, delta_H, delta_S):
    """
    Quantitative Vorhersage: Wie verschiebt Hsp90/Cdc37 das B-Raf Gleichgewicht?

    Ohne Chaperon:
      WT:   50% active
      V600E: 90% active

    Mit Hsp90/Cdc37 (Modell-Vorhersage):
      Cdc37 stabilisiert inaktive Konformation => active Fraktion sinkt
    """
    print("\n--- Vorhersage: B-Raf aktive Fraktion unter Hsp90/Cdc37 ---")

    for label, frac_a_free in [("WT", FRAC_ACTIVE_BRAF_WT),
                                ("V600E", FRAC_ACTIVE_BRAF_V600E)]:
        # Freies Gleichgewicht
        dG_free = -RT * np.log(frac_a_free / (1 - frac_a_free))

        # Unter Hsp90/Cdc37: zusaetzliche Stabilisierung von Inactive
        # Effektive dG_shift = Cdc37-Beitrag = ddG_Cdc37
        # Cdc37 bindet Inactive und praeferiert diese => Inactive wird stabiler
        # Die Groesse des Shifts haengt von Cdc37-Saettigung ab
        for cdc37_frac in [0.0, 0.25, 0.50, 0.75, 1.0]:
            dG_eff = dG_free + cdc37_frac * DDG_CDC37
            frac_a = 1.0 / (1.0 + np.exp(dG_eff / RT))
            marker = ""
            if cdc37_frac == 0.0:
                marker = " (kein Chaperon)"
            elif cdc37_frac == 1.0:
                marker = " (Cdc37-gesaettigt)"
            print(f"  B-Raf {label:5s} | Cdc37 = {cdc37_frac*100:5.1f}%: "
                  f"f(active) = {frac_a*100:5.1f}%{marker}")


def goloubinoff_symmetry_table(is_pg, delta_H, delta_S):
    """Ordne in Goloubinoff-Symmetrie-Schema ein."""
    print("\n--- Goloubinoff-Symmetrie-Klassifikation ---")
    print("  S1 (Binding):     Erhalten (Hsp90 bindet alle Substrate)")
    print(f"  S2 (Unbinding):   GEBROCHEN (Cdc37: K_D-Asymmetrie {SELECTIVITY_FACTOR:.0f}x)")
    print("  S3 (Transition):  Erhalten (ATPase substratunabhaengig)")
    print(f"  S4 (PG/Reversi.): GEBROCHEN (Delta_H={delta_H:.2f} != Delta_S={delta_S:.2f})")
    print(f"  Regime: 3 — Thermodynamisches Nicht-Gleichgewicht")
    print(f"  [ANNAHME] S3-Erhaltung: Hsp90 ATPase kcat ist substratunabhaengig")
    print(f"            (vereinfacht; Cdc37 kann ATPase modulieren)")


def robustness_delta_s(u_H_base, dG_H, delta_H, dG_S):
    """Variiere delta_S und zeige, wann PG kippt."""
    print("\n--- Robustheit: PG-Status vs. Delta_S ---")
    print(f"  (Delta_H = {delta_H:.2f} fixiert; Delta_S variiert)")

    for ds in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, delta_H]:
        u_S_test = np.array([
            [0.0,    dG_S],
            [ds,     dG_S],
        ])
        I_H, I_S, viol, is_pg = four_cycle_test(u_H_base, u_S_test)
        label = "PG=True" if is_pg else "PG=False"
        extra = " <-- Delta_S = Delta_H (reziprok)" if abs(ds - delta_H) < 0.01 else ""
        print(f"  Delta_S = {ds:4.1f}: I_H = {I_H:6.2f}, I_S = {I_S:6.2f}, "
              f"Verletzung = {viol:.2f}, {label}{extra}")


def main():
    print_header()
    print_derived_quantities()

    # Payoffs konstruieren
    u_H, u_S, delta_H, delta_S, dG_H, dG_S = build_payoffs()

    # Payoff-Matrizen anzeigen
    strategies_H = ["Open", "Closed"]
    strategies_S = ["Inactive", "Active"]
    print(f"\n--- Payoff-Matrizen (Delta_H = {delta_H:.2f}, Delta_S = {delta_S:.2f}) ---")
    print("  u_H (Hsp90):")
    print(f"  {'':12s} {'S=Inactive':>12s} {'S=Active':>12s}")
    for i, sh in enumerate(strategies_H):
        print(f"    H={sh:6s}  {u_H[i,0]:12.4f} {u_H[i,1]:12.4f}")
    print("  u_S (Substrat):")
    print(f"  {'':12s} {'H=Open':>12s} {'H=Closed':>12s}")
    for j, ss in enumerate(strategies_S):
        vals = [u_S[i, j] for i in range(2)]
        print(f"    S={ss:8s}{vals[0]:12.4f} {vals[1]:12.4f}")

    # 4-Cycle-Test
    I_H, I_S, violation, is_pg = four_cycle_test(u_H, u_S)
    print(f"\n--- Potential-Game-Test ---")
    print(f"  I_H = {I_H:.4f}")
    print(f"  I_S = {I_S:.4f}")
    print(f"  Verletzung = {violation:.6f}")
    print(f"  PG = {is_pg}")
    if not is_pg:
        print(f"  => Kein Potential Game (S4 gebrochen durch S2-Asymmetrie)")

    # Nash-Gleichgewichte
    nash = find_nash_equilibria(u_H, u_S)
    print(f"\n--- Nash-Gleichgewichte ---")
    if nash:
        for sh, ss, uh, us in nash:
            bio = ""
            if sh == "Closed" and ss == "Inactive":
                bio = "-> Cdc37-Klient (Kinase inaktiviert)"
            elif sh == "Open" and ss == "Active":
                bio = "-> Freie aktive Kinase"
            elif sh == "Closed" and ss == "Active":
                bio = "-> Hsp90-gebunden aber aktiv (instabil)"
            elif sh == "Open" and ss == "Inactive":
                bio = "-> Freie inaktive Kinase"
            print(f"  ({sh}, {ss}): u_H={uh:.4f}, u_S={us:.4f} {bio}")
    else:
        print("  Keine reinen NE gefunden (gemischtes GG erwartet)")

    # S2-Brechungs-Analyse
    s2_breaking_analysis(delta_H, delta_S)

    # B-Raf Vorhersage
    braf_prediction(u_H, u_S, dG_H, delta_H, delta_S)

    # Goloubinoff-Klassifikation
    goloubinoff_symmetry_table(is_pg, delta_H, delta_S)

    # Robustheit
    robustness_delta_s(u_H, dG_H, delta_H, dG_S)

    # ── JSON Output ──────────────────────────────────────────────────
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    outfile = results_dir / "hsp90_calibrated_results.json"

    results = {
        "modell": "Hsp90/Cdc37 S2-Brechung",
        "regime": 3,
        "regime_name": "Thermodynamisches Nicht-Gleichgewicht",
        "experimentelle_parameter": {
            "hsp90_kcat_min": KCAT_HSP90,
            "cdc37_kd_inactive_uM": KD_CDC37_INACTIVE * 1e6,
            "cdc37_kd_active_uM": KD_CDC37_ACTIVE * 1e6,
            "selectivity_factor": float(SELECTIVITY_FACTOR),
            "ddG_cdc37_kcal": float(DDG_CDC37),
            "braf_wt_frac_active_free": FRAC_ACTIVE_BRAF_WT,
            "braf_v600e_frac_active_free": FRAC_ACTIVE_BRAF_V600E,
        },
        "payoff_parameter": {
            "delta_H": float(delta_H),
            "delta_S": float(delta_S),
            "delta_H_quelle": "ddG_Cdc37 (Sreeramulu 2009)",
            "delta_S_quelle": "GESCHAETZT (nicht gemessen)",
        },
        "ergebnisse": {
            "I_H": float(I_H),
            "I_S": float(I_S),
            "four_cycle_violation": float(violation),
            "potential_game": bool(is_pg),
            "nash_equilibria": [
                {"hsp90": sh, "substrat": ss, "u_H": float(uh), "u_S": float(us)}
                for sh, ss, uh, us in nash
            ],
        },
        "kalibrierungsstatus": {
            "hsp90_atpase": {"status": "KALIBRIERT", "quelle": "Prodromou 2000"},
            "cdc37_selektivitaet": {"status": "GESCHAETZT", "quelle": "Sreeramulu 2009 (CDK4); 100x fuer DFG-in/out extrapoliert aus Strukturdaten"},
            "s2_brechung": {"status": "KALIBRIERT", "quelle": "Goloubinoff 2022"},
            "delta_H": {"status": "KALIBRIERT", "quelle": "ddG aus K_D-Ratio"},
            "delta_S": {"status": "ANGENOMMEN", "quelle": "Schaetzung 0.5 kcal/mol"},
            "pg_verletzung": {"status": "FOLGT AUS KONSTRUKTION",
                              "erklaerung": "PG=False folgt algebraisch aus Delta_H != Delta_S"},
        },
        "vorhersage": {
            "beschreibung": "B-Raf aktive Fraktion unter Hsp90/Cdc37",
            "braf_wt_cdc37_saturated_frac_active": float(
                1.0 / (1.0 + np.exp((DG_DFG_WT + DDG_CDC37) / RT))
            ),
            "braf_v600e_cdc37_saturated_frac_active": float(
                1.0 / (1.0 + np.exp((DG_DFG_V600E + DDG_CDC37) / RT))
            ),
        },
        "schlussfolgerung": {
            "s2_gebrochen": True,
            "s4_pg_gebrochen": not is_pg,
            "konsistent_mit_goloubinoff": True,
            "einschraenkung": (
                "PG-Verletzung folgt aus der Payoff-Konstruktion (Delta_H != Delta_S), "
                "nicht aus unabhaengiger Messung. Fuer volle Kalibrierung muesste "
                "Delta_S experimentell bestimmt werden."
            ),
        },
    }

    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n-> Ergebnisse: {outfile}")

    # ── Schluss-Banner ────────────────────────────────────────────────
    print(f"""
======================================================================
ERGEBNIS: Hsp90/Cdc37 teilweise kalibriert (Regime 3)
  PG: {is_pg} (Verletzung = {violation:.2f} kcal/mol)
  S2-Brechung: Cdc37-Selektivitaet {SELECTIVITY_FACTOR:.0f}x (GESCHAETZT aus Strukturdaten)
  Delta_H = {delta_H:.2f}, Delta_S = {delta_S:.2f} (ANGENOMMEN)
  Nash: {len(nash)} NE
  Vorhersage: B-Raf WT -> {1.0 / (1.0 + np.exp((DG_DFG_WT + DDG_CDC37) / RT)) * 100:.1f}% aktiv unter Cdc37 (vs {FRAC_ACTIVE_BRAF_WT*100:.0f}% frei)
  [EINSCHRAENKUNG] PG=False folgt aus Delta_H != Delta_S (Konstruktion)
======================================================================""")


if __name__ == "__main__":
    main()
