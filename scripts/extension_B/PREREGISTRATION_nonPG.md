# PRE-REGISTRATION — Extension B Phase 2: Modellabhängigkeit von PG/non-PG

> **Status:** Pre-Registered (COMMITTED) | Datum: 2026-05-26
> **Regel:** Vorhersagen sind FIXIERT. Code darf erst NACH diesem Dokument geschrieben werden.
> **Abhängigkeit:** Baut auf PREREGISTRATION.md (PG-Hold-Outs) und R1–R6 auf.
> **Revision:** Grundlegend überarbeitet nach Tautologie-Analyse (siehe §2).

---

## 1. Ziel

**Demonstrieren, dass PG/non-PG im FST-Nash-Framework durch die Spielmodell-Wahl
determiniert wird — nicht durch biologische Daten.**

Phase 1 (PG-Hold-Outs) zeigte: PG = True folgt algebraisch aus der
Shared-Δ-Konvention (Δ_H = Δ_S → |I₁ - I₂| = 0 exakt). Die PG-Ergebnisse
waren Konventions-Checks, keine empirischen Tests.

Phase 2 stellt die symmetrische Frage für non-PG: Wenn Δ_H und Δ_S aus
verschiedenen physikalischen Quellen geschätzt werden, ist Δ_H ≠ Δ_S dann
ein genuiner Test — oder eine weitere Konstruktionseigenschaft?

**Ergebnis der Voranalyse:** non-PG ist ebenfalls konstruktionsbedingt.
Δ_H (Bindungsenergie, typisch 3–7 kcal/mol) und Δ_S (Faltungskinetik-Bonus,
typisch 1–2 kcal/mol) stammen aus strukturell verschiedenen physikalischen
Dimensionen → Δ_H > Δ_S ist durch die Schätzmethodik garantiert, nicht
durch biologische Asymmetrie.

**Kernfrage dieser Phase:** Ist die PG/non-PG-Klassifikation eine Eigenschaft
des biologischen Systems — oder eine Eigenschaft der Spielformulierung?

## 2. Tautologie-Analyse (COMMITTED)

### 2.1 Warum ein non-PG-„Test" scheitern muss

Für das 2×2-Spiel gilt: PG ⟺ I₁ = I₂, wobei
I_k = u_k[0,0] − u_k[0,1] − u_k[1,0] + u_k[1,1].

Im Chaperone-vs-Substrat-Modell:
- I_H hängt von Δ_H ab (Kopplungsvorteil für das Chaperone)
- I_S hängt von Δ_S ab (Kopplungsvorteil für das Substrat)
- PG ⟺ Δ_H = Δ_S

Die Δ-Werte werden aus verschiedenen physikalischen Messungen geschätzt:
- Δ_H ← Substratbindungsenergie (aus Kd), typisch 3–7 kcal/mol
- Δ_S ← Faltungsbeschleunigung in der Kavität, typisch 1–2 kcal/mol

Da diese Größen verschiedene physikalische Dimensionen messen, ist
Δ_H > Δ_S praktisch garantiert. Das Ergebnis non-PG wäre also keine
Systemeigenschaft, sondern ein Artefakt der Schätzmethodik.

### 2.2 Die tiefere Einsicht: Modellabhängigkeit

Für Thermosome existieren zwei gleichermaßen plausible Spielmodelle:

| Modell | Spieler | PG-Ergebnis | Grund |
|--------|---------|-------------|-------|
| A: Ring vs Ring | Ring A, Ring B | **PG (garantiert)** | Symmetrisches Spiel → I₁ = I₂ (Monderer & Shapley 1996) |
| B: Chaperone vs Substrat | Thermosome, Substratprotein | **non-PG (praktisch garantiert)** | Verschiedene Spielertypen → Δ_H ≠ Δ_S aus dimensionaler Asymmetrie |

Das PG-Ergebnis hängt nicht von den biologischen Parametern ab, sondern davon,
WER als Spieler definiert wird. Dies ist kein Thermosome-spezifisches Problem —
es betrifft den gesamten Atlas.

### 2.3 Atlas-weite Konstruktionsbedingtheit

| System | PG-Ergebnis | Konstruktionsbasis |
|--------|-------------|-------------------|
| XCL1 | PG | Symmetrisches Monomer-Monomer-Spiel → PG ist Theorem |
| Hsp70/DnaJ | PG | Shared-Δ-Konvention (Δ_H = Δ_S angenommen) |
| SecA | PG | Shared-Δ-Konvention (Δ_H = Δ_S angenommen) |
| Hold-Outs (Phase 1) | PG | Shared-Δ-Konvention |
| Hsp90/Cdc37 | non-PG | Δ_H kalibriert, Δ_S geschätzt → Δ_H ≠ Δ_S |
| GroEL/GroES | non-PG | Asymmetrische Ringstruktur (cis ≠ trans) |

**Befund:** KEIN EINZIGES PG/non-PG-Ergebnis im Atlas ist modellunabhängig
verifiziert. Jedes Ergebnis folgt aus der Spielformulierung + Δ-Konvention.

## 3. COMMITTED Vorhersagen

### 3.1 Thermosome: Beide Modelle (COMMITTED)

**Modell A — Ring vs Ring:**

| Parameter | Vorhersage | Begründung |
|-----------|------------|------------|
| **PG** | **True** | Symmetrisches Spiel: beide Ringe identisch ((αβ)₄) → I₁ = I₂ |
| **NE** | **2** | Anti-kooperative Alternation: (Tight,Relaxed) und (Relaxed,Tight) |
| **Regime** | **2 (kin. NGG)** | ATP vorhanden, aber symmetrische Kopplung |

**Modell B — Thermosome vs Substrat:**

| Parameter | Vorhersage | Begründung |
|-----------|------------|------------|
| **PG** | **False** | Δ_H > Δ_S aus dimensionaler Asymmetrie (Bindung > Kinetik) |
| **NE** | **1** | Open + Folded: Substrat-Faltung strikt dominant, Thermosome daher offen |
| **Regime** | **3 (therm. NGG)** | ATP-getrieben, gerichteter Zyklus, asymmetrische Kopplung |

### 3.2 Meta-Vorhersage (COMMITTED)

**Die Modellwahl determiniert das PG-Ergebnis vollständig.**
Biologische Parameter (Kd, ΔG, Kooperativität) beeinflussen NE und
Regime-Details, aber NICHT die PG-Klassifikation.

### 3.3 Kipppunkt-Vorhersage (Modell B)

PG in Modell B genau dann, wenn Δ_H = Δ_S.
Einziger Kipppunkt: das Δ-Verhältnis. Bei plausiblen biologischen
Parametern (Δ_H ≈ 3–5, Δ_S ≈ 1–2 kcal/mol) ist PG ausgeschlossen.
Δ_S müsste ≈ 3–5 kcal/mol erreichen, was eine 100–1000×
Faltungsbeschleunigung erfordert — physikalisch unrealistisch.

## 4. Thermosome: Systembiologie

### 4.1 Organismus und Struktur

- **Organismus:** Thermoplasma acidophilum
- **Klasse:** Archaeales Typ-II-Chaperonin (Homolog von eukaryotischem CCT/TRiC)
- **Struktur:** Hexadecamer (αβ)₄(αβ)₄, zwei gestapelte Ringe
- **Lid:** Eingebautes Helical-Protrusion-Lid (kein externer GroES-Kofaktor)
- **ATP:** Eigene ATPase, starke negative Inter-Ring-Kooperativität
- **Zyklus:** Open → ATP-Bindung → Lid-Closure → Faltung → Hydrolyse →
  Lid-Öffnung → Produktfreisetzung

### 4.2 Regelkonflikt-Analyse (R4 vs R1)

| Regel | Merkmal | Vorhersage | Modellabhängig? |
|-------|---------|------------|----------------|
| **R4** | Beide Ringe identisch ((αβ)₄), kein Kofaktor | PG | **Ja:** Gilt nur für Ring-vs-Ring (Modell A) |
| **R1** | Eigene ATPase, substratabhängige Stimulation | non-PG | **Ja:** Gilt nur für Chaperone-vs-Substrat (Modell B) |
| **R2** | Bindet entfaltete Proteine (moderat selektiv) | non-PG | Wie R1: nur für Modell B |

**Einsicht:** R4 und R1 widersprechen sich nicht biologisch — sie
beschreiben verschiedene Spielformulierungen desselben Systems.
Der „Regelkonflikt" ist ein Modellwahlproblem.

### 4.3 Literaturparameter (COMMITTED)

Alle Parameter aus Primärliteratur, vor dem Code fixiert.

**ATP-Bindung und -Hydrolyse:**
- Kd(ATP, αβ-Thermosome): 0.65 μM
  (Gutsche, Mihalache & Baumeister 2000, JMB 300:187-196)
- ΔG(ATP-Hydrolyse): −7.3 kcal/mol (Standard, zelluläre Bedingungen)
- ATPase-Ratenlimitierung: Produktfreisetzung (Bigotti, Bellamy & Clarke 2006,
  JMB 362:835-843)

**Inter-Ring-Kooperativität:**
- Starke negative Inter-Ring-Kooperativität: 8 ATP/Hexadecamer = nur ein Ring
  (Bigotti & Clarke 2006)
- Erster Ring: hohe ATP-Affinität; Zweiter Ring: stark reduziert
  (Bigotti & Clarke 2005, JMB 348:13-26)
- ADP schwächt Bindung des zweiten Rings → asymmetrische ATP/ADP-Komplexe
  bevorzugt gegenüber symmetrischen

**Substratbindung:**
- Kd(Substrat) für T. acidophilum: NICHT PUBLIZIERT
- Nächster verfügbarer Wert: Kd(6-APA / rTHS, M. jannaschii) ≈ 17 μM
  (andere Spezies, als Schätzwert verwendbar)
- Substratbindung an apikale Domänen im offenen Zustand
- Hydrophober Patch (Helix-Turn-Helix-Motiv) als Bindungsregion

**Konformationsenergien:**
- ΔG(Open→Closed): Aus allosterischen Konstanten ableitbar.
  L₂ = 2×10⁻⁹ (T→R ohne GroES-Analog), L₂' = 4×10⁻⁵ (T→R mit
  allosterischem Signal) → ΔG ≈ 6–12 kcal/mol für Konformationswechsel
  (analog zu GroEL, Yifrach & Horovitz 1995)
- Temperatur: 55°C (328 K), RT ≈ 0.652 kcal/mol

**EINSCHRÄNKUNG:** Kd(Substrat) ist geschätzt (nicht für T. acidophilum
direkt gemessen). Im Paper als "partially calibrated" kennzeichnen.

### 4.4 Payoff-Matrizen (COMMITTED)

#### Modell A: Ring vs Ring (2×2, symmetrisch)

**Spieler:** Ring A, Ring B (strukturell identisch)
**Strategien:** T (tight/cis, ATP gebunden, Faltung aktiv) vs R (relaxed/trans, Substrat ladend)

u_A[T, T] = dG_conf                  (beide falten: Ressourcen-Konkurrenz)
u_A[T, R] = dG_conf + Δ_ring         (einer faltet, einer lädt: Zyklus)
u_A[R, T] = Δ_ring                   (umgekehrt)
u_A[R, R] = 0                        (beide relaxed: kein Zyklus)

Da u_A = u_B (symmetrisches Spiel): I₁ = I₂ → PG exakt.

Δ_ring ≈ RT·ln(L₂'/L₂) ≈ 0.652 · ln(2×10⁴) ≈ 6.5 kcal/mol

#### Modell B: Thermosome vs Substrat (2×2, asymmetrisch)

**Spieler:** H = Thermosome-Komplex, S = Substratprotein
**Strategien:** H: Open (O) / Closed (C); S: Unfolded (U) / Folded (F)

u_H[O, U] = 0                        (Referenz)
u_H[O, F] = dG_bind_folded           (schwache Bindung an gefaltetes Substrat)
u_H[C, U] = dG_conf + Δ_H            (ATP + Lid-Closure + Substrat in Kavität)
u_H[C, F] = dG_conf                   (Kavität ohne Faltungsbedarf)

u_S[O, U] = 0                        (Referenz)
u_S[O, F] = dG_fold                   (intrinsische Faltungsenergie)
u_S[C, U] = Δ_S                      (Kavität schützt vor Aggregation)
u_S[C, F] = dG_fold                   (gefaltet, Kavität irrelevant)

Δ_H ≈ 3–5 kcal/mol (konformationsabhängiger Anteil der Substratbindung)
Δ_S ≈ 1.5 kcal/mol (Faltungskinetik-Vorteil, aus GroEL-Analogie)

I_H = Δ_H, I_S = Δ_S → PG ⟺ Δ_H = Δ_S. Bei obigen Werten: non-PG.

## 5. Falsifikationskriterien (COMMITTED)

### 5.1 Meta-Vorhersage-Falsifikation

**Falsifiziert wenn:** Ein plausibler Parametersatz existiert, bei dem
Modell A non-PG ergibt ODER Modell B PG ergibt (mit biologisch
realistischen Δ-Werten).

Modell A: Nicht falsifizierbar (PG ist Theorem für symmetrische Spiele).
Modell B: Falsifiziert wenn Δ_S ≈ Δ_H biologisch motivierbar ist.

### 5.2 NE-Falsifikation

**Modell A:** Falsifiziert wenn NE ≠ 2 (erwartet: 2 NE wegen symmetrischer Payoffs).
**Modell B:** Falsifiziert wenn NE ≠ 1 (erwartet: 1 NE, Closed+Unfolded dominant).

### 5.3 Robustheitsanalyse (COMMITTED)

- Variation von Δ_H über 1–10 kcal/mol
- Variation von Δ_S über 0.5–5.0 kcal/mol
- Kipppunkt dokumentieren: bei welchem Δ_S kippt Modell B zu PG?
- Variation von Kd(Substrat) über 1 Größenordnung (1.7–170 μM)

### 5.4 Anti-Tautologie-Check

**Dieser Test ist KEIN genuiner non-PG-Test.** Das wird explizit dokumentiert.
Der Wert liegt in der meta-wissenschaftlichen Erkenntnis:
PG/non-PG ist eine Eigenschaft der Spielformulierung, nicht des Systems.

## 6. Implikationen für das Paper

### 6.1 Publikationswert

Die zentrale Erkenntnis von Phase 2 ist NICHT „Thermosome ist non-PG",
sondern: **Alle PG/non-PG-Ergebnisse im Atlas sind konstruktionsbedingt.**

| PG-Ergebnis | Konstruktionsmechanismus |
|-------------|------------------------|
| PG = True | Symmetrisches Spiel (Theorem) ODER Shared-Δ (Konvention) |
| PG = False | Dimensionale Δ-Asymmetrie (Schätzmethodik) ODER strukturelle Asymmetrie |

### 6.2 Ehrliche Framing-Optionen

**Option A (konservativ):** „Der PG-Test klassifiziert Spielformulierungen,
nicht biologische Systeme. Die Klassifikation ist konsistent mit bekannter
Biologie, aber nicht unabhängig davon."

**Option B (stärker):** „Die Modellabhängigkeit von PG/non-PG zeigt, dass
die Spieler-Partition die informationstragende Entscheidung ist — nicht
die Payoff-Kalibrierung. Die Frage ‚Wer spielt?' ist wichtiger als
‚Wie viel gewinnt er?'."

### 6.3 Abschnitt im Paper

Extension B Phase 2 gehört in den Discussion-Abschnitt als
„Model Dependency and the Limits of PG Classification."

## 7. Referenzen

1. Ditzel L, Löwe J, Stock D et al. (1998) Crystal structure of the thermosome,
   the archaeal chaperonin and homolog of CCT. Cell 93:125-138.
2. Gutsche I, Mihalache O, Baumeister W (2000) ATPase cycle of an archaeal
   chaperonin. J Mol Biol 300:187-196.
3. Bigotti MG, Clarke AR (2005) Cooperativity in the thermosome.
   J Mol Biol 348:13-26.
4. Bigotti MG, Bellamy SRW, Clarke AR (2006) The asymmetric ATPase cycle of the
   thermosome: elucidation of the binding, hydrolysis and product-release steps.
   J Mol Biol 362:835-843.
5. Steinbacher S, Ditzel L (2001) Review: nucleotide binding to the
   Thermoplasma thermosome. J Struct Biol 135:147-156.
6. Monderer D, Shapley LS (1996) Potential Games. Games Econ Behav 14:124-143.
7. Yifrach O, Horovitz A (1995) Nested cooperativity in the ATPase activity
   of the oligomeric chaperonin GroEL. Biochemistry 34:5303-5308.
8. Brinker A, Pfeifer G, Kerner MJ et al. (2001) Dual function of protein
   confinement in chaperonin-assisted protein folding. Cell 107:223-233.

## 8. Checkliste (wird nach Abschluss aktualisiert)

- [x] Pre-Registration geschrieben (dieses Dokument)
- [x] Tautologie-Analyse durchgeführt (§2)
- [x] Literaturparameter verifiziert (Autoren/Journal/Jahr per WebSearch bestätigt; exakte Zahlenwerte aus Primärliteratur nicht gegengeprüft — Volltextzugang nötig)
- [x] Script geschrieben: `thermosome_holdout.py` (BEIDE Modelle)
- [x] Modell A getestet: PG=True, NE=2 (I₁=I₂=−12.92) ✓
- [x] Modell B getestet: PG=False, NE=1 (I₁=−4.0, I₂=−1.5, Verletzung=2.5) ✓
- [x] Robustheitsanalyse (Kipppunkt bei Δ_S=Δ_H=4.0, tautologisch; 2. NE bei Δ_S≥5.0)
- [x] Meta-Ergebnis dokumentiert in BEWEISNOTIZ.md (Phase 2 Abschnitt, 2026-05-26)
