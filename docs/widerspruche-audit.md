# Widersprüche zwischen den Repos — Audit

Date: 2026-04-05

## 1. ANE Peak TFLOPS: 13,86 vs 19

| Repo | Behauptung |
|------|-----------|
| **M4_RE** | "ANE: 13,86 TFLOPS FP16 peak" |
| **ANE-Training** | "ANE Peak: 13,86 TFLOPS (M4)" |
| **Fiber-Inference** | "ANE: 19 TFLOPS FP16 / 38 TOPS INT8" |

**Widerspruch:** M4_RE und ANE-Training sagen 13,86 TFLOPS (GEMESSEN).
Fiber-Inference sagt 19 TFLOPS (Apple's MARKETING-Zahl, nicht gemessen).

**Korrektur nötig:** Fiber-Inference muss auf 13,86 TFLOPS korrigiert werden.

## 2. Parallel Effizienz: "Keine Interferenz" vs "40% langsamer"

| Repo | Behauptung |
|------|-----------|
| **M4_RE** | "Gemessene Gesamtleistung: 18,4 TFLOPS, keine Interferenz!" |
| **M4_RE** | "Alle laufen gleichzeitig ohne Interferenz" |
| **Fiber-Inference** (Paper) | "Parallel ist 40% LANGSAMER wegen Bandwidth-Konkurrenz" |

**Widerspruch:** M4_RE sagt "keine Interferenz", Fiber-Inference beweist das Gegenteil.

**Erklärung:** M4_RE Exp 06 testete UNABHÄNGIGE Workloads (jede Unit eigene Matmul).
Fiber-Inference testete DASSELBE Modell auf allen Units (geteilte Weights = Bandwidth-Kampf).
Beide sind korrekt — aber für verschiedene Szenarien. Die CLAUDE.md Dateien
unterscheiden das nicht.

**Korrektur nötig:** M4_RE CLAUDE.md muss den Kontext ergänzen:
"Keine Interferenz bei unabhängigen Workloads. Bei geteilten Weights: Bandwidth-limitiert."

## 3. Status-Behauptungen in Fiber-Inference veraltet

| Behauptung in CLAUDE.md | Realität |
|------------------------|---------|
| "10.241 tok/s Prefill, 256× Boost" | Gegen schwache CPU Baseline. Ollama MLX: 2,2× schneller |
| "ANE: 19 TFLOPS, Phase 2 Ziel" | ANE Peak ist 13,86 TFLOPS. Phase 2 fertig aber MLX gewinnt |
| "Ziel: ~18,4 TFLOPS = 2-3× Boost" | Parallel ist langsamer, nicht schneller |
| "Gesamt parallel: ~18,4 TFLOPS" | Nur bei unabhängigen Workloads. LLM: Bandwidth-limitiert |
| "Kein existierendes Framework nutzt alle Units" | Stimmt, aber aus gutem Grund (Bandwidth) |

## 4. Veraltete Architektur-Beschreibung

Fiber-Inference CLAUDE.md beschreibt noch die "Ziel-Architektur" mit allen 5 Units parallel.
Das Paper beweist dass das nicht funktioniert. Die CLAUDE.md wurde nie aktualisiert.

## Was zu tun ist

1. **Fiber-Inference CLAUDE.md:** Komplett updaten mit finalen Erkenntnissen
2. **M4_RE CLAUDE.md:** Parallel-Ergebnis präzisieren (unabhängig vs geteilt)
3. **ANE TFLOPS:** Überall auf 13,86 (gemessen) statt 19 (Marketing) korrigieren
