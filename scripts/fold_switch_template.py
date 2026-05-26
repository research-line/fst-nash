"""
Generisches Template für Fold-Switching Nash-Modelle
=====================================================
Wiederverwendbare Funktionen für:
- Payoff-Konstruktion aus experimentellen ΔG-Werten
- Vier-Zyklen-Test (Monderer & Shapley 1996)
- Nash-Gleichgewicht-Suche (reine Strategien)
- Potential-Funktion-Berechnung
- Goloubinoff-Symmetriebrechungs-Analyse (S1-S4)

Wird importiert von systemspezifischen Scripts (rfah, kaib, etc.).
"""

import numpy as np
import json
from pathlib import Path


def four_cycle_test_2x2(u1, u2):
    """Vier-Zyklen-Test für 2×2 Spiel."""
    I1 = u1[0, 0] - u1[0, 1] - u1[1, 0] + u1[1, 1]
    I2 = u2[0, 0] - u2[0, 1] - u2[1, 0] + u2[1, 1]
    violation = abs(I1 - I2)
    return {
        "I_1": round(float(I1), 4),
        "I_2": round(float(I2), 4),
        "violation": round(float(violation), 6),
        "is_PG": bool(violation < 1e-10),
    }


def four_cycle_test_nxn(u1, u2, n, state_names=None):
    """Vier-Zyklen-Test für n×n Spiel."""
    if state_names is None:
        state_names = [str(i) for i in range(n)]

    max_v = 0.0
    violations = []

    for s1 in range(n):
        for s1p in range(n):
            if s1p == s1:
                continue
            for s2 in range(n):
                for s2p in range(n):
                    if s2p == s2:
                        continue
                    lhs = (u1[s1, s2] - u1[s1p, s2]) - (u1[s1, s2p] - u1[s1p, s2p])
                    rhs = (u2[s2, s1] - u2[s2, s1p]) - (u2[s2p, s1] - u2[s2p, s1p])
                    diff = abs(lhs - rhs)
                    if diff > 1e-10:
                        violations.append({
                            "s1": state_names[s1], "s1'": state_names[s1p],
                            "s2": state_names[s2], "s2'": state_names[s2p],
                            "violation": round(float(diff), 4),
                        })
                        max_v = max(max_v, diff)

    return {
        "max_violation": round(float(max_v), 4),
        "n_violations": len(violations),
        "is_PG": bool(max_v < 1e-10),
        "violations": violations,
    }


def find_nash_pure(u1, u2, n, state_names_1=None, state_names_2=None):
    """Finde reine Nash-GG für n×n Spiel."""
    if state_names_1 is None:
        state_names_1 = [str(i) for i in range(n)]
    if state_names_2 is None:
        state_names_2 = [str(i) for i in range(n)]

    nash = []
    for s1 in range(n):
        for s2 in range(n):
            is_nash = True
            for s1_alt in range(n):
                if u1[s1_alt, s2] > u1[s1, s2] + 1e-10:
                    is_nash = False
                    break
            if not is_nash:
                continue
            for s2_alt in range(n):
                if u2[s2_alt, s1] > u2[s2, s1] + 1e-10:
                    is_nash = False
                    break
            if is_nash:
                nash.append({
                    "player_1": state_names_1[s1],
                    "player_2": state_names_2[s2],
                    "u_1": round(float(u1[s1, s2]), 4),
                    "u_2": round(float(u2[s2, s1]), 4),
                    "u_total": round(float(u1[s1, s2] + u2[s2, s1]), 4),
                })
    return nash


def compute_potential_2x2(u1, u2):
    """Potential-Funktion für 2×2 PG."""
    Phi = np.zeros((2, 2))
    Phi[0, 0] = 0.0
    Phi[1, 0] = Phi[0, 0] + (u1[1, 0] - u1[0, 0])
    Phi[0, 1] = Phi[0, 0] + (u2[0, 1] - u2[0, 0])
    Phi[1, 1] = Phi[0, 1] + (u1[1, 1] - u1[0, 1])

    check = Phi[1, 0] + (u2[1, 1] - u2[1, 0])
    consistency = abs(check - Phi[1, 1])

    values = [Phi[0, 0], Phi[1, 0], Phi[0, 1], Phi[1, 1]]
    argmax_idx = int(np.argmax(values))

    return {
        "Phi": Phi,
        "values": {
            "(0,0)": round(float(Phi[0, 0]), 4),
            "(1,0)": round(float(Phi[1, 0]), 4),
            "(0,1)": round(float(Phi[0, 1]), 4),
            "(1,1)": round(float(Phi[1, 1]), 4),
        },
        "argmax_idx": argmax_idx,
        "consistency": round(float(consistency), 10),
    }


def symmetry_breaking_analysis(u1, u2, n=2):
    """Goloubinoff S1-S4 Symmetriebrechungs-Analyse.

    S1: Spieler-Symmetrie (u1 = u2^T?)
    S2: Strategie-Symmetrie (Substrat-unabhängige Raten?)
    S3: Detailed Balance (Kinetische Zeitumkehr)
    S4: Potential Game (Vier-Zyklen-Test)
    """
    s1_diff = np.max(np.abs(u1 - u2.T))

    if n == 2:
        pg = four_cycle_test_2x2(u1, u2)
    else:
        pg = four_cycle_test_nxn(u1, u2, n)

    return {
        "S1_player_symmetry": {
            "max_diff": round(float(s1_diff), 4),
            "broken": bool(s1_diff > 1e-10),
        },
        "S4_potential_game": {
            "is_PG": pg["is_PG"],
            "violation": pg.get("violation", pg.get("max_violation", 0)),
        },
    }


def save_results(results, script_path, filename):
    """Speichere Ergebnisse als JSON."""
    out_path = (Path(script_path).resolve().parent.parent
                / "results" / filename)
    out_path.parent.mkdir(parents=True, exist_ok=True)

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
    return out_path


RT_25C = 0.593   # kcal/mol (25°C)
RT_30C = 0.602   # kcal/mol (30°C)
RT_37C = 0.616   # kcal/mol (37°C)

def RT(T_celsius):
    """RT in kcal/mol für beliebige Temperatur."""
    return 0.001987 * (T_celsius + 273.15)
