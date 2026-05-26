"""
Hsp70→Hsp90 Client-Handoff — 3-Spieler Nash-Modell
====================================================
Experimentelle Daten:
  De Los Rios & Barducci (2014) eLife — Hsp70 Ultra-Affinität
  Boysen et al. (2020) Biophys J — Hsp90 ~29 ATP pro v-Src-Aktivierung
  Taipale et al. (2012) Cell — Hsp90-Kinase Interaktom
  Wang et al. (2022) Nature — Cryo-EM Hsp90-Hop-Hsp70-GR Komplex (2:2:1:1)
  Morán Luengo et al. (2018) Mol Cell — "Hsp90 breaks the deadlock"
  Assimon et al. (2015) — Hop K_D(Hsp70) ~2-4 µM, K_D(Hsp90) ~6-8 µM
  Kirschke et al. (2014) Cell — Koordinierte Chaperon-Zyklen

Spieler:
  1. Hsp70: {ATP-Open (Loslassen), ADP-Closed (Festhalten)}
  2. Client: {Unreif (partiell entfaltet), Reif (near-native)}
  3. Hsp90: {Open (bereit), Closed (prozessierend)}

Biologischer Hintergrund:
  Der Client-Transfer von Hsp70 zu Hsp90 via Hop/STIP1 ist ein aktiver,
  ATP-getriebener Prozess. Hsp70 hält den Client im ADP-Zustand mit
  Ultra-Affinität (K_eff ~0.3 nM), die nur durch ATP-Austausch gelöst wird.
  Hsp90 übernimmt den Client und ermöglicht native Faltung ("breaks deadlock").

  3-Spieler-NE-Analyse: Existiert ein eindeutiges NE, das den biologisch
  funktionalen Handoff-Zustand (Hsp70=Open, Client=Reif, Hsp90=Closed)
  vorhersagt?

Kalibrierungsstatus:
  Hsp70 K_D: KALIBRIERT (De Los Rios 2014)
  Hsp90 K_A: GESCHÄTZT (Boysen 2020: 0.23 µM mit Cdc37)
  Hop-Brücke: KALIBRIERT (Assimon 2015: K_D 2-8 µM)
  Client-Reifung: GESCHÄTZT (Morán Luengo 2018: qualitativ)
"""

import numpy as np
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from fold_switch_template import save_results, RT

RT_37C = RT(37)


# ── Experimentelle Parameter ──────────────────────────────────────

# De Los Rios & Barducci (2014) eLife — Hsp70/DnaK
KD_HSP70_ATP = 1.5e-6      # M (ATP-Zustand, schwache Bindung)
KD_HSP70_ADP = 64.4e-9     # M (ADP-Zustand, starke Bindung)
KEFF_HSP70_NONEQUIL = 0.3e-9  # M (Ultra-Affinität, non-eq)

# Boysen et al. (2020) Biophys J — Hsp90
KA_HSP90_ALONE = 3.0e-6    # M (K_A für v-Src, ohne Cdc37)
KA_HSP90_CDC37 = 0.23e-6   # M (K_A für v-Src, mit Cdc37, 15× besser)
ATP_PER_VSRC = 29           # ATP pro aktiviertem v-Src

# Assimon et al. (2015) — Hop/STIP1
KD_HOP_HSP70 = 3.0e-6      # M (Full-length, FP)
KD_HOP_HSP90 = 7.0e-6      # M (Full-length, FP)

# Hsp90 ATPase
KCAT_HSP90_HUMAN = 0.0009  # s⁻¹ (25°C)

# Abgeleitete Energien
DG_HSP70_ATP = RT_37C * np.log(KD_HSP70_ATP)    # ~ -8.0 kcal/mol
DG_HSP70_ADP = RT_37C * np.log(KD_HSP70_ADP)    # ~ -9.9 kcal/mol
DG_HSP90_CDC37 = RT_37C * np.log(KA_HSP90_CDC37)  # ~ -9.1 kcal/mol
DG_HOP_HSP70 = RT_37C * np.log(KD_HOP_HSP70)    # ~ -7.6 kcal/mol
DG_HOP_HSP90 = RT_37C * np.log(KD_HOP_HSP90)    # ~ -7.1 kcal/mol


def build_3player_payoffs(
    dG_client_fold=-5.0,
    hop_bridge=2.0,
    hsp90_reifung=3.0,
):
    """3-Spieler-Spiel: Hsp70 × Client × Hsp90.

    Jeder Spieler hat 2 Strategien → 2×2×2 = 8 Strategieprofile.
    Payoffs als 3D-Tensoren: u[s_self][s_other1][s_other2].

    Hsp70: {0=ATP-Open, 1=ADP-Closed}
    Client: {0=Unreif, 1=Reif}
    Hsp90: {0=Open, 1=Closed}

    Hop-Brücke: Zusätzlicher Payoff wenn Hsp70=ADP-Closed UND Hsp90=Open
    (beide an Hop gebunden → Lade-Komplex).
    """
    # Payoff-Tensoren: u[s_self, s_partner1, s_partner2]
    # Hsp70: u_70[s_Hsp70, s_Client, s_Hsp90]
    u_70 = np.zeros((2, 2, 2))

    # Hsp70 in ATP-Open (loslassend)
    # Client unreif, Hsp90 open → Hsp70 hat losgelassen, Client driftet
    u_70[0, 0, 0] = 0.0
    # Client unreif, Hsp90 closed → Hsp90 prozessiert ohne Hsp70
    u_70[0, 0, 1] = 0.5
    # Client reif, Hsp90 open → Erfolg: Client fertig, Hsp70 frei
    u_70[0, 1, 0] = 2.0
    # Client reif, Hsp90 closed → Erfolg: Client fertig, Hsp70 frei
    u_70[0, 1, 1] = 2.0

    # Hsp70 in ADP-Closed (festhaltend, Ultra-Affinität)
    ddG_ultra = DG_HSP70_ADP - DG_HSP70_ATP  # ~ -1.9 kcal/mol Bindungsgewinn
    # Client unreif, Hsp90 open → Lade-Komplex via Hop
    u_70[1, 0, 0] = ddG_ultra + hop_bridge
    # Client unreif, Hsp90 closed → Hsp70 hält fest, Hsp90 blockiert
    u_70[1, 0, 1] = ddG_ultra - 1.0  # Deadlock-Strafe
    # Client reif, Hsp90 open → Hsp70 hält unnötig fest
    u_70[1, 1, 0] = ddG_ultra - 2.0  # Ineffizienz
    # Client reif, Hsp90 closed → Hsp70 blockiert reifen Client
    u_70[1, 1, 1] = ddG_ultra - 3.0  # Worst case

    # Client: u_C[s_Client, s_Hsp70, s_Hsp90]
    u_C = np.zeros((2, 2, 2))

    # Client unreif
    # Hsp70 ATP-Open, Hsp90 open → ungeschützt, Aggregationsrisiko
    u_C[0, 0, 0] = -3.0
    # Hsp70 ATP-Open, Hsp90 closed → Hsp90 nimmt unreifen Client
    u_C[0, 0, 1] = -1.0
    # Hsp70 ADP-Closed, Hsp90 open → geschützt durch Hsp70 + Lade-Komplex
    u_C[0, 1, 0] = 1.0 + hop_bridge * 0.5
    # Hsp70 ADP-Closed, Hsp90 closed → geschützt, aber Deadlock
    u_C[0, 1, 1] = 0.5

    # Client reif
    # Hsp70 ATP-Open, Hsp90 open → fertig, frei → Ziel
    u_C[1, 0, 0] = -dG_client_fold  # positiv: stabile Faltung
    # Hsp70 ATP-Open, Hsp90 closed → Hsp90 hält reifen Client
    u_C[1, 0, 1] = -dG_client_fold - 0.5
    # Hsp70 ADP-Closed, Hsp90 open → Hsp70 blockiert reifen Client
    u_C[1, 1, 0] = -dG_client_fold - 2.0
    # Hsp70 ADP-Closed, Hsp90 closed → beide halten reifen Client
    u_C[1, 1, 1] = -dG_client_fold - 3.0

    # Hsp90: u_90[s_Hsp90, s_Hsp70, s_Client]
    u_90 = np.zeros((2, 2, 2))

    # Hsp90 open (bereit für Beladung)
    # Hsp70 ATP-Open, Client unreif → kein Lade-Komplex
    u_90[0, 0, 0] = 0.0
    # Hsp70 ATP-Open, Client reif → unnötig offen
    u_90[0, 0, 1] = -0.5
    # Hsp70 ADP-Closed, Client unreif → Lade-Komplex! Hop-vermittelt
    u_90[0, 1, 0] = hop_bridge
    # Hsp70 ADP-Closed, Client reif → unnötig
    u_90[0, 1, 1] = -1.0

    # Hsp90 closed (prozessierend)
    # Hsp70 ATP-Open, Client unreif → zu früh geschlossen
    u_90[1, 0, 0] = -2.0
    # Hsp70 ATP-Open, Client reif → Reifungs-Komplex (Erfolg)
    u_90[1, 0, 1] = hsp90_reifung
    # Hsp70 ADP-Closed, Client unreif → Deadlock
    u_90[1, 1, 0] = -1.5
    # Hsp70 ADP-Closed, Client reif → Hsp90 prozessiert, Hsp70 stört
    u_90[1, 1, 1] = hsp90_reifung - 2.0

    return u_70, u_C, u_90


def find_nash_3player(u1, u2, u3):
    """Finde reine Nash-GG für 2×2×2 Spiel."""
    names = {
        0: {0: "ATP-Open", 1: "ADP-Closed"},
        1: {0: "Unreif", 1: "Reif"},
        2: {0: "Open", 1: "Closed"},
    }
    nash = []

    for s1 in range(2):
        for s2 in range(2):
            for s3 in range(2):
                is_nash = True

                # Spieler 1 (Hsp70): kann s1 abweichen?
                s1_alt = 1 - s1
                if u1[s1_alt, s2, s3] > u1[s1, s2, s3] + 1e-10:
                    is_nash = False

                # Spieler 2 (Client): kann s2 abweichen?
                if is_nash:
                    s2_alt = 1 - s2
                    if u2[s2_alt, s1, s3] > u2[s2, s1, s3] + 1e-10:
                        is_nash = False

                # Spieler 3 (Hsp90): kann s3 abweichen?
                if is_nash:
                    s3_alt = 1 - s3
                    if u3[s3_alt, s1, s2] > u3[s3, s1, s2] + 1e-10:
                        is_nash = False

                if is_nash:
                    nash.append({
                        "Hsp70": names[0][s1],
                        "Client": names[1][s2],
                        "Hsp90": names[2][s3],
                        "u_Hsp70": round(float(u1[s1, s2, s3]), 2),
                        "u_Client": round(float(u2[s2, s1, s3]), 2),
                        "u_Hsp90": round(float(u3[s3, s1, s2]), 2),
                        "u_total": round(float(
                            u1[s1, s2, s3] + u2[s2, s1, s3] + u3[s3, s1, s2]
                        ), 2),
                    })
    return nash


def four_cycle_test_3player(u1, u2, u3):
    """Paarweise Vier-Zyklen-Tests für 3-Spieler-Spiel.

    Ein 3-Spieler-PG erfordert: für jedes fixierte Profil des dritten
    Spielers muss das resultierende 2-Spieler-Spiel ein PG sein.
    """
    results = []

    for s3 in range(2):
        # Fixiere Hsp90=s3 → 2×2 Spiel Hsp70 × Client
        u1_slice = u1[:, :, s3]
        u2_slice = u2[:, :, s3]  # u2[s_client, s_hsp70] bei fixiertem s3

        I1 = u1_slice[0, 0] - u1_slice[0, 1] - u1_slice[1, 0] + u1_slice[1, 1]
        I2 = u2_slice[0, 0] - u2_slice[0, 1] - u2_slice[1, 0] + u2_slice[1, 1]
        violation = abs(I1 - I2)

        results.append({
            "fixed": f"Hsp90={'Open' if s3==0 else 'Closed'}",
            "I_Hsp70": round(float(I1), 4),
            "I_Client": round(float(I2), 4),
            "violation": round(float(violation), 4),
            "is_PG": bool(violation < 1e-10),
        })

    for s2 in range(2):
        # Fixiere Client=s2 → 2×2 Spiel Hsp70 × Hsp90
        u1_slice = u1[:, s2, :]
        u3_slice = u3[:, :, s2]  # u3[s_hsp90, s_hsp70] bei fixiertem s2

        I1 = u1_slice[0, 0] - u1_slice[0, 1] - u1_slice[1, 0] + u1_slice[1, 1]
        I3 = u3_slice[0, 0] - u3_slice[0, 1] - u3_slice[1, 0] + u3_slice[1, 1]
        violation = abs(I1 - I3)

        results.append({
            "fixed": f"Client={'Unreif' if s2==0 else 'Reif'}",
            "I_Hsp70": round(float(I1), 4),
            "I_Hsp90": round(float(I3), 4),
            "violation": round(float(violation), 4),
            "is_PG": bool(violation < 1e-10),
        })

    max_v = max(r["violation"] for r in results)
    is_PG = all(r["is_PG"] for r in results)

    return {
        "is_PG": is_PG,
        "max_violation": round(max_v, 4),
        "pairwise_tests": results,
    }


def robustness_scan():
    """Scanne über Hop-Brücke und Hsp90-Reifung."""
    results = []
    for hop in [0.0, 1.0, 2.0, 3.0, 5.0]:
        for reifung in [1.0, 3.0, 5.0, 8.0]:
            u_70, u_C, u_90 = build_3player_payoffs(
                hop_bridge=hop, hsp90_reifung=reifung)
            nash = find_nash_3player(u_70, u_C, u_90)
            pg = four_cycle_test_3player(u_70, u_C, u_90)
            profiles = [(n["Hsp70"], n["Client"], n["Hsp90"]) for n in nash]
            results.append({
                "hop_bridge": hop,
                "hsp90_reifung": reifung,
                "is_PG": pg["is_PG"],
                "max_violation": pg["max_violation"],
                "n_NE": len(nash),
                "NE_profiles": profiles,
            })
    return results


def main():
    print("=" * 65)
    print("Hsp70→Hsp90 Client-Handoff — 3-Spieler Nash-Modell")
    print("=" * 65)

    # 1. Abgeleitete Energien
    print(f"\n--- Experimentelle Bindungsenergien ---")
    print(f"  ΔG(Hsp70:Client, ATP):  {DG_HSP70_ATP:.2f} kcal/mol")
    print(f"  ΔG(Hsp70:Client, ADP):  {DG_HSP70_ADP:.2f} kcal/mol")
    print(f"  ΔΔG(Ultra-Affinität):   {DG_HSP70_ADP - DG_HSP70_ATP:.2f} kcal/mol")
    print(f"  ΔG(Hsp90:Client, +Cdc37): {DG_HSP90_CDC37:.2f} kcal/mol")
    print(f"  ΔG(Hop:Hsp70):          {DG_HOP_HSP70:.2f} kcal/mol")
    print(f"  ΔG(Hop:Hsp90):          {DG_HOP_HSP90:.2f} kcal/mol")

    # 2. Payoffs
    u_70, u_C, u_90 = build_3player_payoffs()

    print(f"\n--- Payoff-Tensoren (2×2×2) ---")
    labels_70 = ["ATP-Open", "ADP-Closed"]
    labels_C = ["Unreif", "Reif"]
    labels_90 = ["Open", "Closed"]

    for s3, l3 in enumerate(labels_90):
        print(f"\n  Hsp90={l3}:")
        print(f"    u_Hsp70[s70, sC, {l3}]:")
        print(f"              Client=Unreif  Client=Reif")
        for s1, l1 in enumerate(labels_70):
            print(f"      {l1:<14} {u_70[s1,0,s3]:8.2f}     {u_70[s1,1,s3]:8.2f}")

        print(f"    u_Client[sC, s70, {l3}]:")
        print(f"              Hsp70=ATP-Open  Hsp70=ADP-Closed")
        for s2, l2 in enumerate(labels_C):
            print(f"      {l2:<10}   {u_C[s2,0,s3]:8.2f}        {u_C[s2,1,s3]:8.2f}")

        print(f"    u_Hsp90[{l3}, s70, sC]:")
        print(f"              Client=Unreif  Client=Reif")
        print(f"      Hsp70=ATP-Open     {u_90[s3,0,0]:8.2f}     {u_90[s3,0,1]:8.2f}")
        print(f"      Hsp70=ADP-Closed   {u_90[s3,1,0]:8.2f}     {u_90[s3,1,1]:8.2f}")

    # 3. PG-Test
    pg = four_cycle_test_3player(u_70, u_C, u_90)
    print(f"\n--- Paarweise PG-Tests (3-Spieler) ---")
    for test in pg["pairwise_tests"]:
        keys = [k for k in test if k.startswith("I_")]
        vals = ", ".join(f"{k}={test[k]:.2f}" for k in keys)
        print(f"  Fixiert {test['fixed']}: {vals}, "
              f"Verletzung={test['violation']:.2f}, PG={test['is_PG']}")
    print(f"  Gesamt: PG={pg['is_PG']}, Max-Verletzung={pg['max_violation']:.2f} kcal/mol")

    # 4. Nash-GG
    nash = find_nash_3player(u_70, u_C, u_90)
    print(f"\n--- Nash-Gleichgewichte (3 Spieler) ---")
    print(f"  Anzahl reine NE: {len(nash)}")
    for ne in nash:
        print(f"    Hsp70={ne['Hsp70']}, Client={ne['Client']}, Hsp90={ne['Hsp90']}")
        print(f"      u_70={ne['u_Hsp70']:.2f}, u_C={ne['u_Client']:.2f}, "
              f"u_90={ne['u_Hsp90']:.2f}, Total={ne['u_total']:.2f}")

    # 5. Biologische Interpretation
    print(f"\n--- Biologische Interpretation ---")
    if len(nash) == 1:
        ne = nash[0]
        expected = (ne["Hsp70"] == "ATP-Open" and ne["Client"] == "Reif")
        print(f"  Eindeutiges NE: ({ne['Hsp70']}, {ne['Client']}, {ne['Hsp90']})")
        if expected:
            print(f"  ✓ Biologisch korrekt: Hsp70 lässt los, Client gefaltet")
            print(f"    = Post-Handoff Ruhezustand")
        else:
            print(f"  ⚠ Biologisch unerwartet — Payoffs prüfen")
    elif len(nash) == 0:
        print(f"  Kein reines NE — gemischte Strategien nötig")
    else:
        print(f"  {len(nash)} NE — Multistabilität im Handoff-System")

    # 6. Robustheit
    print(f"\n--- Robustheitsscan ---")
    rob = robustness_scan()
    print(f"  {'Hop':>6} {'Reif.':>6} {'PG?':>5} {'Verl.':>8} {'NE':>4} Profil")
    for r in rob:
        profiles = ", ".join(f"({a},{b},{c})" for a, b, c in r["NE_profiles"])
        print(f"  {r['hop_bridge']:>6.1f} {r['hsp90_reifung']:>6.1f} "
              f"{'Y' if r['is_PG'] else 'N':>5} {r['max_violation']:>8.2f} "
              f"{r['n_NE']:>4} {profiles}")

    # ── JSON Output ──
    results = {
        "experiment": "Hsp70→Hsp90 Client-Handoff — 3-Spieler Nash-Modell",
        "referenzen": {
            "Hsp70_ultraaffinity": "De Los Rios & Barducci (2014) eLife 3:e02218",
            "Hsp90_activation": "Boysen et al. (2020) Biophys J",
            "Hop_binding": "Assimon et al. (2015) PMC4714923",
            "cryo_EM": "Wang et al. (2022) Nature 601:460",
            "deadlock": "Morán Luengo et al. (2018) Mol Cell 70:545",
            "coord_cycles": "Kirschke et al. (2014) Cell 158:434",
        },
        "parameter": {
            "KD_Hsp70_ATP_uM": round(KD_HSP70_ATP * 1e6, 2),
            "KD_Hsp70_ADP_nM": round(KD_HSP70_ADP * 1e9, 1),
            "Keff_nonequil_nM": round(KEFF_HSP70_NONEQUIL * 1e9, 1),
            "KA_Hsp90_Cdc37_uM": round(KA_HSP90_CDC37 * 1e6, 2),
            "KD_Hop_Hsp70_uM": round(KD_HOP_HSP70 * 1e6, 1),
            "KD_Hop_Hsp90_uM": round(KD_HOP_HSP90 * 1e6, 1),
            "ATP_per_vSrc": ATP_PER_VSRC,
        },
        "pg_test": pg,
        "nash_gleichgewichte": nash,
        "robustheit": rob,
        "schlussfolgerung": {
            "n_NE": len(nash),
            "is_PG": pg["is_PG"],
            "max_violation": pg["max_violation"],
        },
    }

    out_path = save_results(results, __file__, "handoff_hsp70_hsp90_results.json")
    print(f"\n→ Ergebnisse: {out_path}")

    print(f"\n{'='*65}")
    print(f"ERGEBNIS: 3-Spieler Handoff {'NICHT-PG' if not pg['is_PG'] else 'PG'}, "
          f"{len(nash)} NE")
    if len(nash) == 1:
        ne = nash[0]
        print(f"  NE: ({ne['Hsp70']}, {ne['Client']}, {ne['Hsp90']})")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
