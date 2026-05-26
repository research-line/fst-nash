"""
Hsp70/DnaK Kalibriertes Spieltheoretisches Modell
===================================================
Kalibrierung des illustrativen Hsp70-Modells (chaperone_cross_validation.py)
mit experimentellen Kinetik-Daten.

Testfrage: Bestaetigt die Kalibrierung die illustrative Vorhersage?
  - Hsp70 bricht S3 (kinetische Asymmetrie: DnaJ-stimulierte Hydrolyse)
  - Hsp70 bewahrt S4 (Potential Game: reziproke Kopplung)
  - Klassifikation: Regime 2 — Kinetisches Nicht-Gleichgewicht

Spieler:
  1. Hsp70 (H): {Open (O, ATP-gebunden), Closed (C, ADP-gebunden)}
  2. Substrat (S): {Unfolded (U), Folded (F)}

Payoff-Ableitung:
  Intrinsische Energien aus Nukleotid-Austausch-Gleichgewicht (reversibel, S4).
  Bindungsenergien aus K_D-Werten (S1+S2 bewahrt: substratunabhaengig).
  Kopplungsenergie aus DnaJ-Ternaerkomplex (reziprok: gleiche physikalische
  Interaktion von beiden Seiten betrachtet → PG strukturell garantiert).

Status: Teilweise kalibriert.
  Kinetik-Parameter (K_D, k_h, k_NEF): KALIBRIERT aus De Los Rios (2014).
  S3-Brechung (3000x Hydrolyseraten-Asymmetrie): KALIBRIERT.
  PG/S4 (Kopplungsreziprozitaet): ANGENOMMEN (Delta_H = Delta_S),
    konsistent mit Goloubinoff (2022) S4-Vorhersage, aber NICHT aus
    unabhaengigen Messungen abgeleitet.

  Fuer volle Kalibrierung noetig:
  - I_H unabhaengig: DnaJ-Hsp70-Substrat Ternaerkomplex-Thermodynamik
  - I_S unabhaengig: Aggregationsschutz-Assays (dG_protection)
  - Pruefung ob |I_H - I_S| < epsilon

Referenzen:
  De Los Rios & Barducci (2014) eLife 3:e02218 — Kinetische Parameter
  Goloubinoff, Sassi, Fauvet, Barducci et al. (2018) Chem Rev 118:4202
  Goloubinoff et al. (2022) Biomolecules 12:832 — S1-S4 Symmetrien
  Kityk et al. (2012) EMBO J 31:1502 — DnaJ J-Domaene Bindung
  Mayer (2013) Mol Cell 51:607 — Hsp70 Mechanismus
"""

import numpy as np
import json
from pathlib import Path


RT = 0.593  # kcal/mol bei 25 C (Messtemperatur De Los Rios 2014)

# ── Experimentelle Kinetik-Parameter ────────────────────────────────
# De Los Rios & Barducci (2014) eLife 3:e02218

K_ON_ATP = 4.5e5       # s-1 M-1, Substrat-Assoziation (Open/ATP)
K_OFF_ATP = 2.0         # s-1,     Substrat-Dissoziation (Open/ATP)
K_ON_ADP = 1.0e3        # s-1 M-1, Substrat-Assoziation (Closed/ADP)
K_OFF_ADP = 4.7e-4      # s-1,     Substrat-Dissoziation (Closed/ADP)

K_H_BASAL = 6.0e-4      # s-1, basale ATP-Hydrolyse (kein/gefaltetes Substrat)
K_H_STIM = 1.8          # s-1, DnaJ+Substrat-stimuliert (ungefaltetes Substrat)

K_ATP_RELEASE = 1.33e-4  # s-1,     spontane ATP-Freisetzung
K_ADP_RELEASE = 0.022    # s-1,     ADP-Freisetzung (GrpE-katalysiert)

K_EFF_NEQ_NM = 0.22      # nM, nicht-GG Ultra-Affinitaet (De Los Rios Fig.3)


def derived_quantities():
    """Thermodynamische Groessen aus den Kinetik-Daten."""
    KD_ATP = K_OFF_ATP / K_ON_ATP
    KD_ADP = K_OFF_ADP / K_ON_ADP
    s3_ratio = K_H_STIM / K_H_BASAL

    dG_bind_ATP = RT * np.log(KD_ATP)
    dG_bind_ADP = RT * np.log(KD_ADP)
    ddG_bind = dG_bind_ADP - dG_bind_ATP

    ddG_S3 = RT * np.log(s3_ratio)

    K_eq_OC = K_ATP_RELEASE / K_ADP_RELEASE
    dG_OC = -RT * np.log(K_eq_OC)

    KD_eff_M = K_EFF_NEQ_NM * 1e-9
    ddG_ultra = RT * np.log(KD_eff_M / KD_ADP)

    return {
        "KD_ATP_uM": round(KD_ATP * 1e6, 2),
        "KD_ADP_uM": round(KD_ADP * 1e6, 2),
        "S3_ratio": round(s3_ratio, 0),
        "dG_bind_ATP_kcal": round(dG_bind_ATP, 2),
        "dG_bind_ADP_kcal": round(dG_bind_ADP, 2),
        "ddG_bind_kcal": round(ddG_bind, 2),
        "ddG_S3_kinetic_kcal": round(ddG_S3, 2),
        "K_eq_OC_reversible": round(K_eq_OC, 5),
        "dG_OC_reversible_kcal": round(dG_OC, 2),
        "ddG_ultra_affinity_kcal": round(ddG_ultra, 2),
    }


# ── Payoff-Konstruktion ────────────────────────────────────────────


def build_payoffs(Delta=3.0, dG_fold=-5.0):
    """Kalibrierte Payoff-Matrizen aus experimentellen Daten.

    Drei Beitraege zum Hsp70-Payoff bei (C, *):
      1. Konformationsenergie: -dG_OC (Open bevorzugt → Strafe fuer Closed)
      2. Bindungsvorteil:      -ddG_bind (Closed bindet fester → Bonus)
      3. Ternaerkomplex:       +Delta NUR bei (C, U) (DnaJ-Kopplung)

    Substrat-Payoff:
      - (U, O) = 0 (Referenz)
      - (U, C) = +Delta (Aggregationsschutz durch Closed Hsp70)
      - (F, *) = -dG_fold (nativ stabil, unabhaengig von Hsp70-Zustand per S2)

    Reziprozitaet: Gleicher Delta-Wert fuer beide Spieler, weil derselbe
    DnaJ-Ternaerkomplex beide stabilisiert.

    Args:
        Delta: Kopplungsenergie des DnaJ-Ternaerkomplexes (kcal/mol).
        dG_fold: Substrat-Faltungsenergie (kcal/mol, negativ = stabil).
    """
    dq = derived_quantities()
    dG_OC = dq["dG_OC_reversible_kcal"]
    ddG_bind = dq["ddG_bind_kcal"]

    u_H = np.zeros((2, 2))
    u_H[0, 0] = 0.0
    u_H[0, 1] = 0.0
    u_H[1, 0] = -dG_OC - ddG_bind + Delta
    u_H[1, 1] = -dG_OC - ddG_bind

    u_S = np.zeros((2, 2))
    u_S[0, 0] = 0.0
    u_S[0, 1] = Delta
    u_S[1, 0] = -dG_fold
    u_S[1, 1] = -dG_fold

    return u_H, u_S


# ── Potential-Game-Test ─────────────────────────────────────────────


def interaction_contrast(u):
    """I = u[0,0] - u[0,1] - u[1,0] + u[1,1] fuer 2x2-Spiel."""
    return u[0, 0] - u[0, 1] - u[1, 0] + u[1, 1]


def four_cycle_test(u_H, u_S):
    """Vier-Zyklen-Test (Monderer & Shapley 1996) fuer 2x2-Spiel."""
    I_H = interaction_contrast(u_H)
    I_S = interaction_contrast(u_S)
    violation = abs(I_H - I_S)
    return {
        "I_H": round(float(I_H), 4),
        "I_S": round(float(I_S), 4),
        "violation": round(float(violation), 6),
        "is_PG": bool(violation < 1e-10),
    }


def compute_potential(u_H, u_S):
    """Potential-Funktion Phi fuer ein Potential Game."""
    Phi = np.zeros((2, 2))
    Phi[0, 0] = 0.0
    Phi[1, 0] = Phi[0, 0] + (u_H[1, 0] - u_H[0, 0])
    Phi[0, 1] = Phi[0, 0] + (u_S[1, 0] - u_S[0, 0])
    Phi[1, 1] = Phi[0, 1] + (u_H[1, 1] - u_H[0, 1])

    check = Phi[1, 0] + (u_S[1, 1] - u_S[0, 1])
    consistency = abs(check - Phi[1, 1])

    states = ["(O,U)", "(C,U)", "(O,F)", "(C,F)"]
    values = [Phi[0, 0], Phi[1, 0], Phi[0, 1], Phi[1, 1]]
    argmax_idx = int(np.argmax(values))

    return {
        "values": {s: round(float(v), 4) for s, v in zip(states, values)},
        "argmax": states[argmax_idx],
        "argmax_value": round(float(values[argmax_idx]), 4),
        "consistency_check": round(float(consistency), 10),
    }


# ── Nash-Gleichgewichte ────────────────────────────────────────────


def find_nash_equilibria(u_H, u_S):
    """Finde reine Nash-Gleichgewichte im 2x2-Spiel."""
    strategies_H = ["Open", "Closed"]
    strategies_S = ["Unfolded", "Folded"]
    nash = []

    for h in range(2):
        for s in range(2):
            h_best = (u_H[h, s] >= u_H[1 - h, s])
            s_best = (u_S[s, h] >= u_S[1 - s, h])
            if h_best and s_best:
                nash.append({
                    "Hsp70": strategies_H[h],
                    "Substrat": strategies_S[s],
                    "u_H": round(float(u_H[h, s]), 4),
                    "u_S": round(float(u_S[s, h]), 4),
                })
    return nash


# ── Detailed-Balance-Test ──────────────────────────────────────────


def detailed_balance_test():
    """Pruefe Detailed Balance im 4-Zustandszyklus (O,U)→(C,U)→(C,F)→(O,F)→(O,U).

    Nimmt substratunabhaengige Faltung/Entfaltung an (S2 bewahrt).
    Dann ist das DB-Verhaeltnis = k_h_stim / k_h_basal.
    """
    db_ratio = K_H_STIM / K_H_BASAL
    dG_cycle = -RT * np.log(db_ratio)

    return {
        "cycle": "(O,U) -> (C,U) -> (C,F) -> (O,F) -> (O,U)",
        "forward_backward_ratio": round(float(db_ratio), 1),
        "dG_cycle_kcal": round(float(dG_cycle), 2),
        "violated": bool(abs(db_ratio - 1.0) > 0.01),
        "erklaerung": (
            "DB-Verletzung kommt ausschliesslich aus S3-Brechung: "
            f"k_h(U)/k_h(F) = {db_ratio:.0f}. "
            "Andere Raten sind substratunabhaengig (S1+S2 bewahrt). "
            f"Zyklus-Energiebeitrag: {dG_cycle:.2f} kcal/mol aus ATP-Hydrolyse."
        ),
    }


# ── S3-Quantifizierung ─────────────────────────────────────────────


def s3_analysis():
    """Quantifiziere die S3-Brechung (substratabhaengige Uebergangsraten)."""
    ratio = K_H_STIM / K_H_BASAL
    ddG = RT * np.log(ratio)

    return {
        "k_h_U_per_s": K_H_STIM,
        "k_h_F_per_s": K_H_BASAL,
        "ratio": round(float(ratio), 0),
        "ddG_S3_kcal": round(float(ddG), 2),
        "mechanismus": (
            "DnaJ (Hsp40) bindet exponierte hydrophobe Reste ungefalteter Substrate. "
            "Gleichzeitige Bindung von DnaJ und Substrat an Hsp70 stimuliert die "
            f"ATPase um Faktor {ratio:.0f}x (Goloubinoff 2022: S3-Brechung). "
            "GrpE-katalysierter Nukleotid-Austausch bleibt substratunabhaengig."
        ),
    }


# ── Vergleich mit illustrativem Modell ──────────────────────────────


def comparison_with_illustrative():
    """Vergleiche kalibriertes mit illustrativem Modell."""
    u_H_ill = np.array([[0.0, 0.0], [-2.0, 4.0]])
    u_S_ill = np.array([[5.0, 2.0], [0.0, 3.0]])
    pg_ill = four_cycle_test(u_H_ill, u_S_ill)
    nash_ill = find_nash_equilibria(u_H_ill, u_S_ill)

    u_H_cal, u_S_cal = build_payoffs()
    pg_cal = four_cycle_test(u_H_cal, u_S_cal)
    nash_cal = find_nash_equilibria(u_H_cal, u_S_cal)

    return {
        "illustrativ": {
            "u_H": u_H_ill.tolist(),
            "u_S": u_S_ill.tolist(),
            "I_H": pg_ill["I_H"],
            "I_S": pg_ill["I_S"],
            "is_PG": pg_ill["is_PG"],
            "Nash_GG": nash_ill,
        },
        "kalibriert": {
            "u_H": u_H_cal.tolist(),
            "u_S": u_S_cal.tolist(),
            "I_H": pg_cal["I_H"],
            "I_S": pg_cal["I_S"],
            "is_PG": pg_cal["is_PG"],
            "Nash_GG": nash_cal,
        },
        "uebereinstimmung": {
            "PG_gleich": pg_ill["is_PG"] == pg_cal["is_PG"],
            "Nash_GG_gleich": (
                len(nash_ill) == len(nash_cal)
                and all(
                    n1["Hsp70"] == n2["Hsp70"] and n1["Substrat"] == n2["Substrat"]
                    for n1, n2 in zip(nash_ill, nash_cal)
                )
            ),
            "NE_divergenz": (
                f"Illustrativ: {len(nash_ill)} NE, Kalibriert: {len(nash_cal)} NE. "
                "Illustratives Modell hat 2 NE (Open+Native, Closed+Misfolded), "
                "kalibriertes nur 1 NE (Open+Folded). Biologisch korrekt: "
                "Hsp70 hat keinen stabilen (Closed,Unfolded)-Zustand — "
                "Substrat praeferiert immer Folded bei physiologischem dG_fold."
            ),
        },
    }


# ── Robustheitsanalyse ──────────────────────────────────────────────


def robustness_structural():
    """Zeige: PG gilt fuer jedes Delta UNTER DER ANNAHME Delta_H = Delta_S.

    Wenn die Kopplung reziprok ist (beide Spieler profitieren gleich),
    dann I_H = I_S = -Delta fuer JEDES Delta. Dies ist eine Konsequenz
    der Payoff-Konstruktion, nicht ein unabhaengiges Ergebnis.
    """
    results = []
    for Delta in [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0]:
        u_H, u_S = build_payoffs(Delta=Delta)
        pg = four_cycle_test(u_H, u_S)
        results.append({
            "Delta": Delta,
            "I_H": pg["I_H"],
            "I_S": pg["I_S"],
            "violation": pg["violation"],
            "is_PG": pg["is_PG"],
        })
    return results


def robustness_asymmetry(n_trials=200):
    """Teste Robustheit gegen Brechung der Reziprozitaet.

    Fuege Rauschen hinzu, das die Reziprozitaet stoert:
    Delta_H != Delta_S (verschiedene Kopplungsenergien fuer die beiden Spieler).
    Zeige: PG-Verletzung skaliert linear mit der Asymmetrie.
    """
    rng = np.random.default_rng(42)
    results = []
    dq = derived_quantities()
    dG_OC = dq["dG_OC_reversible_kcal"]
    ddG_bind = dq["ddG_bind_kcal"]
    Delta_base = 3.0
    dG_fold = -5.0

    for _ in range(n_trials):
        noise = rng.normal(0, 1.0)
        Delta_H = Delta_base + noise
        Delta_S = Delta_base - noise

        u_H = np.array([
            [0.0, 0.0],
            [-dG_OC - ddG_bind + Delta_H, -dG_OC - ddG_bind],
        ])
        u_S = np.array([
            [0.0, Delta_S],
            [-dG_fold, -dG_fold],
        ])

        pg = four_cycle_test(u_H, u_S)
        results.append({
            "asymmetry": round(abs(Delta_H - Delta_S), 4),
            "violation": pg["violation"],
        })

    violations = [r["violation"] for r in results]
    asymmetries = [r["asymmetry"] for r in results]

    if max(asymmetries) > 0:
        slope = np.polyfit(asymmetries, violations, 1)[0]
    else:
        slope = 0.0

    pg_maintained = sum(1 for v in violations if v < 0.01)

    return {
        "n_trials": n_trials,
        "noise_sigma_kcal": 1.0,
        "mean_violation": round(float(np.mean(violations)), 4),
        "max_violation": round(float(np.max(violations)), 4),
        "pg_maintained_strict": pg_maintained,
        "violation_vs_asymmetry_slope": round(float(slope), 4),
        "interpretation": (
            f"PG-Verletzung skaliert linear mit Asymmetrie (Steigung = {slope:.2f}). "
            "Biologisch ist die Asymmetrie nahe Null, weil beide Effekte "
            "(DnaJ-Stimulation fuer Hsp70, Aggregationsschutz fuer Substrat) "
            "aus derselben physikalischen Interaktion stammen."
        ),
    }


def robustness_substrate_variation():
    """Teste verschiedene Substrate (dG_fold variiert)."""
    results = []
    for dG_fold in [-3.0, -5.0, -7.0, -10.0, -15.0]:
        u_H, u_S = build_payoffs(dG_fold=dG_fold)
        pg = four_cycle_test(u_H, u_S)
        nash = find_nash_equilibria(u_H, u_S)
        results.append({
            "dG_fold_kcal": dG_fold,
            "is_PG": pg["is_PG"],
            "n_Nash_GG": len(nash),
            "Nash_GG": [n["Hsp70"] + "+" + n["Substrat"] for n in nash],
        })
    return results


# ── Biologische Interpretation ──────────────────────────────────────


def biological_interpretation(u_H, u_S, nash, pg, db, s3):
    """Zusammenfassende biologische Interpretation."""
    return {
        "regime": "Regime 2: Kinetisches Nicht-GG (S3 gebrochen, S4 intakt)",
        "Nash_GG_bedeutung": (
            "Einziges Nash-GG: (Open, Folded) = biologischer Ruhezustand. "
            "Substrat ist gefaltet, Hsp70 im ATP/Open-Zustand (leer). "
            "Dies ist das thermodynamische Gleichgewicht."
        ),
        "Nicht_GG_Zyklus": (
            "Der Hsp70-Faltungszyklus (C,U → Release → Fold → O,F) "
            "ist KEIN Nash-GG — Substrat praeferiert immer F ueber U. "
            "Der Zyklus wird kinetisch durch S3-Brechung getrieben: "
            f"DnaJ beschleunigt die Hydrolyse {s3['ratio']:.0f}x fuer ungefaltetes Substrat. "
            "Dies erzeugt Ultra-Affinitaet (K_eff = 0.22 nM vs K_D = 0.47 uM)."
        ),
        "PG_bedeutung": (
            "PG bewahrt (S4 intakt): Die Kopplung zwischen Hsp70 und Substrat "
            "ist REZIPROK. Beide profitieren gleich vom DnaJ-Ternaerkomplex. "
            "Physikalische Ursache: Es ist dieselbe Interaktion (DnaJ bindet "
            "exponierte hydrophobe Reste UND stimuliert Hsp70-ATPase), "
            "von zwei Seiten betrachtet."
        ),
        "unterschied_zu_GroEL": (
            "GroEL bricht S4 (nicht-PG): Die Kopplung zwischen cis- und trans-Ring "
            "ist ASYMMETRISCH (ATP-Hydrolyse im cis erzwingt trans-Konformation, "
            "aber nicht umgekehrt). Bei Hsp70 gibt es keine solche Asymmetrie — "
            "DnaJ verknuepft Hsp70 und Substrat symmetrisch."
        ),
    }


# ── Hauptprogramm ──────────────────────────────────────────────────


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


def main():
    print("=" * 70)
    print("Hsp70/DnaK Kalibriertes Spieltheoretisches Modell")
    print("Experimentelle Parameter: De Los Rios & Barducci (2014) eLife")
    print("=" * 70)

    # 1. Abgeleitete Groessen
    dq = derived_quantities()
    print(f"\n--- Abgeleitete thermodynamische Groessen ---")
    print(f"  K_D(ATP/Open):   {dq['KD_ATP_uM']:.2f} uM")
    print(f"  K_D(ADP/Closed): {dq['KD_ADP_uM']:.2f} uM")
    print(f"  ddG_bind:        {dq['ddG_bind_kcal']:.2f} kcal/mol (Closed bindet fester)")
    print(f"  S3-Verhaeltnis:  {dq['S3_ratio']:.0f}x")
    print(f"  ddG_S3:          {dq['ddG_S3_kinetic_kcal']:.2f} kcal/mol (kinetisch)")
    print(f"  dG(O->C, rev.):  {dq['dG_OC_reversible_kcal']:.2f} kcal/mol (Open bevorzugt)")
    print(f"  ddG_ultra:       {dq['ddG_ultra_affinity_kcal']:.2f} kcal/mol (Nicht-GG-Affinitaet)")

    # 2. Payoff-Matrizen
    Delta = 3.0
    dG_fold = -5.0
    u_H, u_S = build_payoffs(Delta=Delta, dG_fold=dG_fold)

    print(f"\n--- Payoff-Matrizen (Delta = {Delta} kcal/mol, dG_fold = {dG_fold}) ---")
    print(f"  u_H (Hsp70):")
    print(f"              Unfolded     Folded")
    print(f"    Open      {u_H[0,0]:8.2f}   {u_H[0,1]:8.2f}")
    print(f"    Closed    {u_H[1,0]:8.2f}   {u_H[1,1]:8.2f}")
    print(f"  u_S (Substrat):")
    print(f"              Open         Closed")
    print(f"    Unfolded  {u_S[0,0]:8.2f}   {u_S[0,1]:8.2f}")
    print(f"    Folded    {u_S[1,0]:8.2f}   {u_S[1,1]:8.2f}")

    # 3. PG-Test
    pg = four_cycle_test(u_H, u_S)
    print(f"\n--- Potential-Game-Test ---")
    print(f"  I_H = {pg['I_H']:.4f}")
    print(f"  I_S = {pg['I_S']:.4f}")
    print(f"  Verletzung = {pg['violation']:.6f}")
    print(f"  PG = {pg['is_PG']}")

    # 4. Potential-Funktion
    if pg["is_PG"]:
        pot = compute_potential(u_H, u_S)
        print(f"\n--- Potential-Funktion Phi ---")
        for state, val in pot["values"].items():
            marker = " <-- NE (Maximum)" if state == pot["argmax"] else ""
            print(f"  Phi{state} = {val:.4f}{marker}")
        print(f"  Konsistenz: {pot['consistency_check']:.1e}")
    else:
        pot = None

    # 5. Nash-GG
    nash = find_nash_equilibria(u_H, u_S)
    print(f"\n--- Nash-Gleichgewichte ---")
    for ne in nash:
        print(f"  ({ne['Hsp70']}, {ne['Substrat']}): u_H={ne['u_H']:.2f}, u_S={ne['u_S']:.2f}")
    if not nash:
        print("  Keine reinen Nash-GG gefunden.")

    # 6. Detailed Balance
    db = detailed_balance_test()
    print(f"\n--- Detailed-Balance-Test ---")
    print(f"  Zyklus: {db['cycle']}")
    print(f"  Forward/Backward: {db['forward_backward_ratio']:.1f}")
    print(f"  dG_Zyklus: {db['dG_cycle_kcal']:.2f} kcal/mol")
    print(f"  Verletzt: {db['violated']}")

    # 7. S3-Analyse
    s3 = s3_analysis()
    print(f"\n--- S3-Brechung ---")
    print(f"  k_h(U) = {s3['k_h_U_per_s']} s-1")
    print(f"  k_h(F) = {s3['k_h_F_per_s']} s-1")
    print(f"  Verhaeltnis = {s3['ratio']:.0f}x")
    print(f"  ddG_S3 = {s3['ddG_S3_kcal']:.2f} kcal/mol")

    # 8. Vergleich mit illustrativem Modell
    comp = comparison_with_illustrative()
    print(f"\n--- Vergleich: Illustrativ vs. Kalibriert ---")
    print(f"  Illustrativ: PG={comp['illustrativ']['is_PG']}, "
          f"I_H={comp['illustrativ']['I_H']}, I_S={comp['illustrativ']['I_S']}")
    print(f"  Kalibriert:  PG={comp['kalibriert']['is_PG']}, "
          f"I_H={comp['kalibriert']['I_H']}, I_S={comp['kalibriert']['I_S']}")
    print(f"  PG-Uebereinstimmung: {comp['uebereinstimmung']['PG_gleich']}")
    print(f"  NE-Uebereinstimmung: {comp['uebereinstimmung']['Nash_GG_gleich']}")

    # 9. Robustheit
    print(f"\n--- Robustheit: Strukturelle PG-Garantie ---")
    rob_struct = robustness_structural()
    for r in rob_struct:
        print(f"  Delta={r['Delta']:5.1f}: I_H={r['I_H']:7.2f}, "
              f"I_S={r['I_S']:7.2f}, PG={r['is_PG']}")

    print(f"\n--- Robustheit: Asymmetrie-Sensitivitaet (200 Trials) ---")
    rob_asym = robustness_asymmetry()
    print(f"  Mittlere PG-Verletzung: {rob_asym['mean_violation']:.4f} kcal/mol")
    print(f"  Maximale PG-Verletzung: {rob_asym['max_violation']:.4f} kcal/mol")
    print(f"  Steigung (Verletzung/Asymmetrie): {rob_asym['violation_vs_asymmetry_slope']:.4f}")

    print(f"\n--- Robustheit: Substrat-Variation ---")
    rob_sub = robustness_substrate_variation()
    for r in rob_sub:
        print(f"  dG_fold={r['dG_fold_kcal']:6.1f}: PG={r['is_PG']}, "
              f"Nash={r['Nash_GG']}")

    # 10. Biologische Interpretation
    bio = biological_interpretation(u_H, u_S, nash, pg, db, s3)

    # ── JSON-Output ─────────────────────────────────────────────────
    results = {
        "experiment": "Hsp70/DnaK kalibriertes spieltheoretisches Modell",
        "datum": "2026-05-26",
        "referenzen": {
            "kinetik": "De Los Rios & Barducci (2014) eLife 3:e02218",
            "symmetrien": "Goloubinoff et al. (2022) Biomolecules 12:832",
            "illustrativ": "chaperone_cross_validation.py (Hsp70 Toy-Modell)",
        },
        "parameter": {
            "experimentell": {
                "k_on_ATP_per_s_M": K_ON_ATP,
                "k_off_ATP_per_s": K_OFF_ATP,
                "k_on_ADP_per_s_M": K_ON_ADP,
                "k_off_ADP_per_s": K_OFF_ADP,
                "k_h_basal_per_s": K_H_BASAL,
                "k_h_stimulated_per_s": K_H_STIM,
                "k_NEF_per_s": K_ADP_RELEASE,
                "K_eff_neq_nM": K_EFF_NEQ_NM,
            },
            "abgeleitet": dq,
            "modell": {
                "Delta_kcal": Delta,
                "dG_fold_kcal": dG_fold,
                "RT_kcal": RT,
                "temperatur": "25 C (Messtemperatur)",
            },
        },
        "payoffs": {
            "u_H": u_H.tolist(),
            "u_S": u_S.tolist(),
            "konvention": "u[strategie_eigener_spieler, strategie_anderer_spieler]",
            "strategien_H": ["Open (ATP)", "Closed (ADP)"],
            "strategien_S": ["Unfolded", "Folded"],
        },
        "pg_test": pg,
        "potential_funktion": pot,
        "nash_gleichgewichte": nash,
        "detailed_balance": db,
        "s3_brechung": s3,
        "vergleich_illustrativ": comp,
        "robustheit": {
            "strukturell": rob_struct,
            "asymmetrie": rob_asym,
            "substrat_variation": rob_sub,
        },
        "biologische_interpretation": bio,
        "schlussfolgerung": {
            "hsp70_ist_PG": True,
            "hsp70_bricht_S3": True,
            "hsp70_bewahrt_S4_angenommen": True,
            "regime": "Regime 2: Kinetisches Nicht-GG",
            "kalibrierungsstatus": {
                "kalibriert": [
                    "K_D(ATP), K_D(ADP) aus De Los Rios 2014",
                    "S3-Ratio (3000x) aus De Los Rios 2014",
                    "dG(O->C) aus NEF/Hydrolyse-Raten",
                ],
                "angenommen": [
                    "Delta_H = Delta_S (Reziprozitaet) — konsistent mit "
                    "Goloubinoff S4, aber nicht unabhaengig gemessen",
                    "Delta = 3.0 kcal/mol (Schaetzwert, Bereich 2-5 kcal/mol)",
                    "dG_fold = -5.0 kcal/mol (typisch fuer globulaere Proteine)",
                ],
            },
            "NE_divergenz": (
                "Kalibriertes Modell: 1 NE (Open, Folded). "
                "Illustratives: 2 NE. NE-Struktur NICHT identisch. "
                "Kalibriertes Modell biologisch praeziser: "
                "(Closed,Unfolded) ist kinetischer Transit, kein stabiles GG."
            ),
            "PG_status": (
                "PG folgt algebraisch aus Delta_H = Delta_S. "
                "Dies ist eine ANNAHME (gleiche physikalische Interaktion), "
                "keine unabhaengige Messung. Volle Kalibrierung erfordert "
                "separate Bestimmung von I_H (Ternaerkomplex-Thermodynamik) "
                "und I_S (Aggregationsschutz-Assays)."
            ),
        },
    }

    out_path = Path(__file__).resolve().parent.parent / "results" / "hsp70_calibrated_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=convert)
    print(f"\n→ Ergebnisse: {out_path}")

    print(f"\n{'='*70}")
    print("ERGEBNIS: Hsp70 teilweise kalibriert (Regime 2)")
    print(f"  S3 gebrochen: k_h(U)/k_h(F) = {int(K_H_STIM/K_H_BASAL)} [KALIBRIERT]")
    print(f"  S4 intakt (PG): I_H = I_S = {pg['I_H']:.2f} [ANGENOMMEN: Delta_H=Delta_S]")
    print(f"  DB verletzt: Faktor {int(K_H_STIM/K_H_BASAL)}")
    print(f"  Nash-GG: (Open, Folded) — kalibriert: 1 NE (vs. illustrativ: 2 NE)")
    print(f"  NE-Divergenz biologisch korrekt: (C,U) ist Transit, kein stabiles GG")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
