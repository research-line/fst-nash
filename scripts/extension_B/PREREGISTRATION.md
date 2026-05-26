# PRE-REGISTRATION — Extension B: Regime-Vorhersage-Brücke

> **Status:** Pre-Registered (COMMITTED) | Datum: 2026-05-26
> **Regel:** Mapping-Regeln und Vorhersagen sind FIXIERT.
> Code darf erst NACH diesem Dokument geschrieben werden.
> Änderungen an Regeln nach Implementierung erfordern neues Pre-Registration-Dokument.

---

## 1. Ziel

Regime-Vorhersagen (PG/non-PG, NE-Zahl, Regime-Klasse) aus biologischen
Features ableiten statt Payoff-Matrizen manuell kalibrieren.
Mapping-Regeln R1–R6 bilden qualitative biologische Merkmale auf qualitative
spieltheoretische Kategorien ab — keine numerischen Payoff-Werte.
Erfolg: Hold-Out-Systeme korrekt klassifiziert (Regime + NE-Zahl).
Misserfolg: ≤50% Hold-Out-Treffer → Mapping-Regeln haben keinen prädiktiven Gehalt.

## 2. Mapping-Regeln (COMMITTED)

Jede Regel bildet ein biologisches Feature auf einen Payoff-Parameter ab.
Biologische Begründung pro Regel ist Pflicht.

### R1: ATP-Hydrolyse → Kopplungsasymmetrie (Δ_H ≠ Δ_S)

**Feature:** Besitzt das System eine eigene ATPase-Domäne?
**Mapping:** ATP-Hydrolyse → gerichteter Energiefluss → Δ_H ≠ Δ_S
**Begründung:** ATP-Hydrolyse erzeugt Nicht-Reziprozität (Chaperon investiert
Energie, Substrat nicht). Goloubinoff (2022): S4-Brechung erfordert
irreversiblen Antrieb.

| ATP-Status | Vorhersage |
|------------|------------|
| Eigene ATPase + substratabhängige Stimulation | Δ_H ≠ Δ_S (non-PG wahrscheinlich) |
| Eigene ATPase ohne Substrat-Stimulation | Δ_H ≈ Δ_S (PG möglich) |
| Kein ATP | Δ_H = Δ_S (PG, wenn symmetrisch) |

**Ausnahme identifiziert (Atlas):** Hsp70/DnaJ hat ATP + Substrat-Stimulation
(3000×) und ist trotzdem PG. Grund: reziproke Kopplung (Δ_H = Δ_S per Annahme).
→ R1 allein ist NICHT hinreichend. Regel muss mit R2 kombiniert werden.

### R2: Substrat-Selektivität → Kopplungsasymmetrie-Richtung

**Feature:** Bindet das Chaperon Substrate konformationsabhängig (selektiv)?
**Mapping:** Konformationsselektion → Δ_H > Δ_S (Chaperon profitiert stärker
von spezifischem Substrat-Zustand als Substrat vom Chaperon-Zustand)
**Begründung:** Cdc37 bindet nur DFG-out-Kinasen (100× selektiver).
Bei reziproker Bindung (DnaK bindet jedes hydrophobe Motiv ähnlich) bleibt PG.

| Selektivitäts-Muster | Vorhersage |
|----------------------|------------|
| Hohe Konformationsselektion (z.B. Cdc37: DFG-out only) | Δ_H >> Δ_S → non-PG |
| Geringe Selektivität (breites Substratspektrum) | Δ_H ≈ Δ_S → PG möglich |

### R3: Konformationskosten → NE-Multiplizität

**Feature:** Freie Energie des Konformationswechsels (ΔG_conf)
**Mapping:** |ΔG_conf| → NE-Zahl
**Begründung:** Aus Atlas-Beobachtung (Q10, Hypothese v3):
2 NE entsteht, wenn Konformationswechsel thermodynamischen Preis hat (kein
Zustand strikt dominant). 1 NE, wenn ein Zustand strikt dominiert.

| ΔG_conf | Vorhersage |
|---------|------------|
| ΔG ≈ 0 (beide Konformationen gleich stabil) + beidseitige Funktion | 2 NE |
| |ΔG| > kT (ein Zustand stabiler) | 1 NE |
| Nur ein Zustand funktionell | 1 NE |

**Caveat:** Dies ist nahe an der Definition von NE-Multiplizität in 2×2-Spielen
("dominant strategy → 1 NE"). Der biologische Gehalt liegt in der Zuordnung
konkreter ΔG_conf-Werte zu NE-Übergängen, nicht in der spieltheoretischen Logik.

### R4: Oligomerer Zustand → Spieleranzahl/Symmetrie

**Feature:** Homo-Oligomer vs. Hetero-Komplex
**Mapping:** Symmetrisches Homo-Oligomer → symmetrisches Koordinationsspiel → PG trivial
Asymmetrischer Hetero-Komplex → unterschiedliche Payoff-Funktionen → PG-Bruch möglich
**Begründung:** Symmetrische Spiele sind immer Potential Games (mathematisch).
Asymmetrie ist notwendig (nicht hinreichend) für PG-Bruch.

### R5: Redox-/Temperatur-Schalter → Konditionale Payoff-Änderung

**Feature:** Besitzt das Chaperon einen diskreten Aktivierungsschalter?
**Mapping:** Binärer Schalter (aktiv/inaktiv) → Payoff-Matrix ist zustandsabhängig.
Im inaktiven Zustand: Chaperon-Interaktion ≈ 0 (kein Spiel).
Im aktiven Zustand: volle Payoff-Matrix.
**Begründung:** Hsp33 wechselt zwischen reduziert-inaktiv und oxidiert-aktiv.
sHsps wechseln zwischen Oligomer-inaktiv und Dimer-aktiv bei Hitzeschock.
Der Schalter selbst ist kein spieltheoretisches Phänomen — er definiert die
BEDINGUNGEN, unter denen das Spiel stattfindet.

### R6: Substrat-Übergabe → Multi-Spieler-Potential

**Feature:** Übergibt das Chaperon sein Substrat an ein nachfolgendes System?
**Mapping:** Übergabe vorhanden → 3-Spieler-Spiel (Chaperon-Substrat-Empfänger)
→ non-PG wahrscheinlich (verschiedene Spieler, verschiedene Ziele)
**Begründung:** Hsp70→Hsp90 Handoff (3-Spieler) ist non-PG im Atlas.
Prefoldin→GroEL Übergabe hat ähnliche Struktur.

## 3. Hold-Out-Systeme (COMMITTED)

Diese Systeme sind NICHT im Atlas (12 kalibrierte Systeme). Die Regime-
Vorhersagen werden VOR dem Coding committed.

### Hold-Out 1: DnaJ allein (ohne Hsp70)

**System:** E. coli DnaJ/Hsp40 als unabhängiger Holdase
**Biologische Eigenschaften:**
- Kein eigenes ATP (R1: kein ATP → PG-Tendenz)
- Bindet hydrophobe Motive breit (R2: geringe Selektivität → Δ_H ≈ Δ_S)
- Homo-Dimer (R4: symmetrisch → PG-Tendenz)
- Kein diskreter Schalter (R5: nicht zutreffend)
- Übergibt an Hsp70 (R6: in vivo ja, aber als Holdase allein: nein)

**COMMITTED Vorhersage:**
- **PG:** Ja (kein ATP, geringe Selektivität, symmetrisch)
- **NE:** 1 (DnaJ-Bound = einziger stabiler Zustand, Substrat profitiert
  immer von Bindung, kein Konformations-Trade-off für DnaJ)
- **Regime:** 1 (GG) — reines Gleichgewichts-Holdase
- **Begründung:** DnaJ ohne Hsp70 ist ein passiver Hydrophobizitäts-Sensor.
  Kein gerichteter Zyklus, keine Substrat-Selektivität.

### Hold-Out 2: Hsp26 (S. cerevisiae sHsp)

**System:** Yeast Hsp26 als temperaturaktivierter Holdase
**Biologische Eigenschaften:**
- Kein ATP (R1: PG-Tendenz)
- Bindet entfaltete Proteine unspezifisch (R2: geringe Selektivität)
- 24-40mer Homo-Oligomer (R4: symmetrisch)
- **Temperatur-Schalter** (R5: Oligomer→Dimer bei Hitzeschock)
- Übergibt an Hsp70/Hsp104 für Refaltung (R6: Übergabe vorhanden,
  aber als Holdase allein: 2-Spieler-Spiel)

**COMMITTED Vorhersage:**
- **PG:** Ja (kein ATP, symmetrisch, unspezifische Bindung)
- **NE:** 1 (Bound = einziger funktioneller Zustand. Obwohl das Oligomer
  als "inaktiv" gilt, ist der aktivierte Dimer-Zustand der relevante
  Spielraum. Im aktiven Zustand: Substrat bindet → 1 NE)
- **Regime:** 1 (GG)
- **Begründung:** sHsps sind die einfachsten Chaperone: Bindung ohne
  ATP, ohne Selektivität, ohne Zyklus. Temperaturschalter definiert,
  WANN gespielt wird, nicht WIE.

### Hold-Out 3: Hsp33 (E. coli, redox-reguliert)

**System:** Hsp33 als redox-aktivierter Holdase
**Biologische Eigenschaften:**
- Kein ATP (R1: PG-Tendenz)
- Bindet entfaltete Substrate mit K_d = 3–300 nM (SUPREX, Xu et al. 2010)
- **ABER:** Bindet NUR im oxidierten Zustand (R2: konditional, nicht konformationsselektiv
  für Substrat, sondern für eigenen Zustand)
- Monomer (reduziert) → Dimer (oxidiert) (R4: Dimerisierung = Aktivierung)
- **Redox-Schalter** (R5: Zn²⁺-Release + 2 Disulfidbrücken)
- Übergibt an DnaK/DnaJ/GrpE für Refaltung (R6: Übergabe vorhanden)

**COMMITTED Vorhersage:**
- **PG:** Ja (kein ATP, im aktiven Zustand symmetrisches Dimer)
- **NE:** 1 (Im oxidierten Zustand: Substrat-Bindung = einziger stabiler
  Zustand. Hsp33 ist konformationell fixiert (oxidiert/entfaltet) →
  kein Konformations-Trade-off → dominante Strategie)
- **Regime:** 1 (GG) — innerhalb des aktiven Fensters
- **Begründung:** Der Redox-Schalter IST das interessante Phänomen,
  aber er liegt AUSSERHALB des spieltheoretischen Modells. Im aktiven
  Zustand ist Hsp33 ein passiver Holdase wie DnaJ.
- **Besonderheit:** Wenn man den Redox-Schalter als Konformation
  modelliert (Reduced-Folded vs. Oxidized-Unfolded), entsteht ein
  2-Zustands-System. Vorhersage: |ΔG(red→ox)| >> kT (Zn²⁺-Affinität
  > 10¹⁷ M⁻¹) → dominant: reduzierter Zustand → 1 NE (Reduced, Unbound).
  Oxidativer Stress erzwingt den Wechsel extern (nicht Nash-getrieben).

### Hold-Out 4: Hsc70 allein (ohne DnaJ/Hsp40)

**System:** Humanes Hsc70 (HSPA8) ohne J-Domänen-Cochaperon
**Biologische Eigenschaften:**
- **Eigene ATPase** (R1: basale Rate ~0.02–0.04 min⁻¹, OHNE Substrat-Stimulation
  durch J-Domäne. DnaJ stimuliert >1000×, hier fehlt DnaJ)
- Bindet hydrophobe Motive BREIT (R2: geringe Selektivität, K_d im µM–mM-Bereich,
  ähnlich DnaK. Keine Konformationsselektion wie Cdc37)
- Monomer (R4: asymmetrischer Hetero-Komplex mit Substrat → PG-Bruch möglich,
  aber gleiche Situation wie Hsp70/DnaJ im Atlas, das PG ist)
- Kein diskreter Schalter (R5: nicht zutreffend)
- In vivo arbeitet Hsc70 MIT Cochaperonen; hier als isoliertes System modelliert

**COMMITTED Vorhersage:**
- **PG:** Ja (R1 Zeile 2: eigene ATPase OHNE Substrat-Stimulation → Δ_H ≈ Δ_S.
  R2: breite Substratspezifität → Δ_H ≈ Δ_S. Beide Regeln konvergieren auf PG)
- **NE:** 1 (Im ADP-Zustand: Substrat-Bindung = stabiler Zustand.
  Kein Konformations-Trade-off ohne J-Domänen-Stimulation → dominante Strategie)
- **Regime:** 1 (GG) — Hsc70 allein verhält sich funktionell wie ein passiver Holdase
- **Begründung:** Hsc70 testet R1 Zeile 2 direkt: "Eigene ATPase ohne Substrat-
  Stimulation → PG möglich." Falls Hsc70 allein non-PG wäre, wäre R1 Zeile 2
  falsch (ATP ohne Stimulation reicht doch für PG-Bruch). Das unterscheidet
  Hsc70 von den anderen drei Hold-Outs, die KEIN ATP haben und daher R1 Zeile 3
  testen ("Kein ATP → PG").
- **Konsistenzcheck mit Atlas:** Hsp70/DnaJ ist PG im Atlas (kin. NGG: S3 gebrochen,
  S4 intakt). DnaJ hat substratabhängige Stimulation (~1000×), aber Hsp70/DnaJ
  fällt trotzdem unter R1 Zeile 1 + R2 (breite Spezifität → PG). Hsc70 allein
  (R1 Zeile 2, schwächere ATPase) sollte erst recht PG sein.

## 4. Erfolgs-/Misserfolgskriterien

### Primärkriterium (Regime-Vorhersage)

| Treffer | Bewertung |
|---------|-----------|
| 4/4 Hold-Outs korrekt klassifiziert | Konsistenz mit Mapping-Regeln bestätigt |
| 3/4 korrekt | Teilweise Konsistenz, einzelne Regel-Revision nötig |
| 2/4 korrekt | Schwache Konsistenz, systematische Revision nötig |
| ≤1/4 | Mapping-Regeln haben keinen prädiktiven Gehalt |

**Wichtige Limitation (PG-only Test-Set):** Alle vier Hold-Outs werden als
PG vorhergesagt. Ein 4/4-Ergebnis bestätigt, dass die PG-Seite der
Mapping-Regeln generalisiert, testet aber NICHT die non-PG-Seite.
Konkret: R1 Zeile 1 (ATP + Substrat-Stimulation → non-PG) und R6
(geschlossene Kammer → non-PG) werden durch dieses Hold-Out-Set nicht
geprüft. Ein vollständiger Test erfordert zusätzliche non-PG-Hold-Outs
(z.B. ein ATP-abhängiges Foldase-System), die über diesen
Proof-of-Concept hinausgehen.

### Sekundärkriterium (NE-Zahl)

Für Systeme mit erwarteter NE=1: Sensitivitätsanalyse zeigt, dass 1 NE
robust über ±50% Parametervariation erhalten bleibt.

### Tertiärkriterium (Quantitative Vorhersage)

Falls experimentelle Bindungskonstanten verfügbar: Modell-Population
weicht <20% von experimentellem Wert ab.

## 5. Implementierungsplan

1. ✅ PREREGISTRATION.md schreiben (dieses Dokument)
2. ✅ Literaturverifikation: Alle Referenzen verifiziert (PubMed + WebSearch),
   3 Referenz-Fehler korrigiert (Gässler→Rüdiger, Franzmann Journal, Groitl→Xu)
3. ✅ Scripts geschrieben: `dnaj_holdout.py`, `hsp26_holdout.py`, `hsp33_holdout.py`,
   `hsc70_holdout.py` (shared-Delta-Konvention, PG-Konvention-Labels)
4. ✅ Vorhersagen getestet: 4/4 PG=Konvention-konsistent, 4/4 NE=1 (strukturell konsistent)
5. ✅ Ergebnisse dokumentiert: Hit/Miss-Tabelle + Chronologie-Transparenz in BEWEISNOTIZ.md

## 6. Anti-Tautologie-Checks

### Check 1: Nicht-triviale Vorhersagen

Alle vier Hold-Outs werden als PG/Regime 1 vorhergesagt. Dies ist eine
**riskante Vorhersage**, weil:
- Trigger Factor (im Atlas) ist non-PG trotz fehlendem ATP
- KaiB (im Atlas) ist non-PG trotz indirektem ATP
Falls ein Hold-Out non-PG ist (z.B. Hsp33 durch Redox-Asymmetrie),
wäre die PG-Vorhersage falsifiziert.

**Ehrliche Einschränkung:** Das Test-Set ist ein PG-Bestätigungstest,
kein PG-vs-non-PG-Diskriminierungstest. R1 Zeile 1 (ATP + Stimulation
→ non-PG) und R6 (Kammer → non-PG) bleiben ungetestet. Erfolg bei
4/4 zeigt Konsistenz der PG-Seite, nicht Gesamtvalidität der Regeln.

**Diskriminierender Test (Hsc70):** Hsc70 allein hat ATP (anders als die
anderen drei Hold-Outs). Die PG-Vorhersage für Hsc70 testet R1 Zeile 2
direkt: "Eigene ATPase OHNE Substrat-Stimulation → Δ_H ≈ Δ_S (PG möglich)."
R2 (breite Spezifität → PG) konvergiert. Falls Hsc70 non-PG wäre, wäre
R1 Zeile 2 falsifiziert (ATP ohne Stimulation reicht doch für PG-Bruch).

### Check 2: Unabhängigkeit von Atlas-Kalibrierung

Die Mapping-Regeln sind aus Atlas-Beobachtungen ABGELEITET (nicht unabhängig).
Dies ist eine fundamentale Limitation: Hold-Outs testen die
GENERALISIERUNG der Regeln, nicht ihre UNABHÄNGIGKEIT.

Echte Unabhängigkeit erfordert Regeln aus ERSTER PRINZIPIEN (Thermodynamik,
statistische Mechanik), nicht aus empirischem Pattern-Matching.
Dies ist langfristiges Ziel, nicht Gegenstand dieses Proof-of-Concept.

### Check 3: Falsifizierbarkeit

Die Vorhersagen sind falsifizierbar:
- DnaJ als non-PG → R1 (kein ATP → PG) wäre widerlegt
- Hsp26 mit 2 NE → R3 (NE hängt von ΔG_conf ab) müsste revidiert werden
- Hsp33 als non-PG → R5 (Schalter ist orthogonal zu PG) wäre widerlegt

## 7. Referenzen (für Hold-Out-Kalibrierung)

- **DnaJ:** Rüdiger, Schneider-Mergener & Bukau (2001) EMBO J 20:1042
  (DnaJ-Substratspezifität, erklärt unabhängige Aggregations-Suppression);
  Liberek et al. (1991) PNAS 88:2874 (DnaK-ATPase-Zyklus: DnaJ+GrpE als
  Co-Chaperone; Kontext hier: DnaJ ALLEIN hat KEINE ATPase, konsistent mit
  R1 Zeile 3 "kein ATP → PG")
- **Hsp26:** Haslbeck et al. (1999) EMBO J 18:6744 (Temperaturaktivierung);
  Franzmann et al. (2005) J Mol Biol 350:1083 (Oligomer-Dissoziation nicht nötig);
  Haslbeck et al. (2021) Nat Commun 12:6768 (Phosphorylierung/Cryo-EM)
- **Hsp33:** Graumann et al. (2001) Structure 9:377 (Zwei-Schritt-Aktivierung);
  Xu, Schmitt, Tang, Jakob & Fitzgerald (2010) Biochemistry 49:1346 (SUPREX/K_d);
  Reichmann et al. (2012) Cell 148:947 (Arbeitszyklus, Übergabe an DnaK)
- **Hsc70 allein:** Mayer & Bukau (2005) CMLS 62:670 (Hsp70-Review, basale ATPase);
  Ha & McKay (1994) Biochemistry 33:14625 (Hsc70 ATPase-Kinetik);
  Kampinga & Craig (2010) Nat Rev Mol Cell Biol 11:579 (J-Domänen-Stimulation);
  Rüdiger et al. (1997) EMBO J 16:1501 (DnaK-Substratspezifität, hydrophobe Motive)
