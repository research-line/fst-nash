# C2: Formaler Beweis der Tautologie-Proposition

> Teilblatt zur BEWEISNOTIZ.md | Erstellt: 2026-05-25

## Proposition

**Sei F: T^{2N} → ℝ ein glattes Potential mit periodischen Cosinus-Kopplungen
der Form F(θ) = Σ_{(i,j)} w_{ij} · (1 − cos(θ_i − θ_j − δ_{ij})).
Dann gilt an jedem nicht-entarteten strikten lokalen Minimum θ* von F
(d.h. θ* ist kritischer Punkt und H = ∇²F(θ*) hat keinen Null-Eigenwert):**

**(a)** Die Hesse-Matrix H = ∇²F(θ*) ist positiv definit.

**(b)** Für die Jacobi-Matrix J = I − η·H mit Lernrate η ∈ (0, 2/λ_max) gilt ρ(J) < 1.

**(c)** Das Ergebnis ρ(J) < 1 ist unabhängig von den Kopplungsparametern w_{ij}, δ_{ij}
und der Proteinstruktur — es folgt ausschließlich aus der Glattheit von F und der
Definition eines lokalen Minimums.

## Beweis

### Teil (a): H positiv definit am Minimum

Sei θ* ein striktes lokales Minimum von F.

Per Definition: ∃ ε > 0 sodass F(θ) > F(θ*) für alle θ mit 0 < ‖θ − θ*‖ < ε.

Da F zweimal stetig differenzierbar ist (Cosinus-Kopplungen sind C^∞), folgt
aus der Taylor-Entwicklung zweiter Ordnung:

  F(θ* + h) = F(θ*) + ∇F(θ*)ᵀ h + ½ hᵀ H h + o(‖h‖²)

Am kritischen Punkt gilt ∇F(θ*) = 0 (notwendige Bedingung erster Ordnung).
Also:

  F(θ* + h) − F(θ*) = ½ hᵀ H h + o(‖h‖²) > 0  für alle h ≠ 0 mit ‖h‖ < ε

Dies impliziert hᵀ H h ≥ 0 für alle h (H positiv semidefinit).

Damit ist H positiv semidefinit. Nicht-Entartung (Voraussetzung) schließt
Null-Eigenwerte aus, also ist H positiv definit. ∎

**Zur Nicht-Entartung:** Im allgemeinen impliziert ein striktes Minimum NICHT
positiv definite Hesse (Gegenbeispiel: f(x) = x⁴, f''(0) = 0).
Für Cosinus-Kopplungen ist Nicht-Entartung jedoch generisch:
- Die Hesse hat Graph-Laplacian-Struktur mit Gewichten w_{ij}·cos(θ_i−θ_j−δ_{ij}).
- Am globalen Minimum sind alle cos(…) = 1 → H definitiv positiv definit.
- An anderen lokalen Minima ist Entartung ein Maß-Null-Ereignis im Parameterraum.
- Empirisch: Alle L-BFGS-Endpunkte (60/60 getestet) haben λ_min > 0.003. ∎

### Teil (b): ρ(J) < 1

H positiv definit ⟹ Eigenwerte 0 < λ_min ≤ ... ≤ λ_max.

J = I − η·H hat Eigenwerte μ_k = 1 − η·λ_k.

Für η ∈ (0, 2/λ_max):
- μ_k = 1 − η·λ_k > 1 − 2·λ_k/λ_max ≥ 1 − 2 = −1
- μ_k = 1 − η·λ_k < 1 − 0 = 1  (da η > 0, λ_k > 0)

Also |μ_k| < 1 für alle k, damit ρ(J) = max_k |μ_k| < 1. ∎

### Teil (c): Parameterunabhängigkeit

Der Beweis verwendet nur:
1. F ist C² (glatt) — garantiert durch Cosinus-Terme
2. θ* ist ein striktes lokales Minimum — garantiert durch L-BFGS/Gradientenabstieg
3. η ist hinreichend klein — frei wählbar

Keine Eigenschaft der Parameter w_{ij}, δ_{ij} oder der Proteinsequenz geht ein.
Insbesondere liefern zufällige Parameter dasselbe Ergebnis (empirisch bestätigt:
50/50 = 100% Nash-stabil mit Random-Parametern). ∎

## Bemerkung zur biologischen Relevanz

Das Ergebnis ρ(J) < 1 ist eine mathematische Eigenschaft JEDES lokalen Minimums
JEDES glatten Potentials auf dem Torus — nicht eine Eigenschaft gefalteter Proteine.
Es hat keinen biologischen Informationsgehalt, weil:

1. Es an JEDEM Minimum gilt, nicht nur am biologisch relevanten.
2. Es mit BELIEBIGEN Parametern gilt, nicht nur den gelernten.
3. Es keine Korrelation mit experimentellen Observablen zeigt (B-Faktoren: r ≈ −0.1).

## Verbindung zum Potential-Game-Theorem

Die gleiche Tautologie gilt in diskreter Form: Wenn ein Spiel ein Potential Game ist
(alle Spieler minimieren gemeinsam eine Funktion Φ), dann sind Nash-GG die lokalen
Optima von Φ. Die Nash-Analyse fügt nichts zur direkten Optimierung hinzu.

MWC-Allosterie mit alignierten Untereinheiten-Utilities IST ein Potential Game.
Die Spieltheorie wird erst nicht-trivial, wenn Utilities NICHT aligniert sind
(verschiedene Untereinheiten, verschiedene Ziele).
