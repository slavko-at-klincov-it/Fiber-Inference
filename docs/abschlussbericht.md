# Abschlussbericht: Fiber-Inference — Apple-Silicon-Native LLM Architektur

Date: 2026-04-05

## Was war das Ziel?

Eine eigene LLM-Inference-Architektur bauen die den Apple M4 Chip voll ausnutzt.
Alle 5 Compute Units (ANE, GPU, AMX, P-cores, E-cores) parallel verwenden.
2-4x schneller als existierende Frameworks (Ollama, llama.cpp, MLX).

## Was haben wir gebaut?

### Code (40 Commits, ~7000 Zeilen)
- ANE Attention Kernel (MIL, GQA, RoPE, Fused SDPA)
- ANE FFN Kernel (MIL, SwiGLU)
- GPU Decode Pipeline (Metal, Multi-Row MatVec)
- AMX FFN Pipeline (Accelerate/cblas)
- Fused Residual+RMSNorm Metal Kernel
- Synthetisches Modell-Framework (Fiber-768)
- BLZT + karpathy Checkpoint Loader
- Stories-110M Text-Generierung (kohaerenter Text auf ANE)
- Hardware Sweep (alle 5 Units charakterisiert)
- Limit Tests (seq, dim, GQA, INT8, Dynamic Weights, Sliding Window)
- Proof System (3-Wege Token-Match Verifikation)
- Ollama Baseline Benchmarks

### Dokumentation (20 Dateien in docs/)
- architektur-grundlagen.md — LLM Begriffe, warum Apple Silicon
- hardware-sweep.md — ANE/GPU/AMX GFLOPS pro Dimension
- limits-findings.md — ANE Grenzen (seq, dim, SRAM)
- fiber768-findings.md — Architektur-Prototyp Ergebnisse
- proof.md + proof-sweep.md — Korrektheitsbeweis
- large-model-sweep.md + large-model-ane.md — 15M-9B Tests
- ane-decode-findings.md — Decode-Versuche
- int8-findings.md — INT8 scheitert
- dynamic-weights-findings.md — Dynamic Weights scheitert ab IC≥256
- sliding-window-findings.md — Kein Speed-Vorteil
- specialist-review.md — Externe Bewertung
- llama-cpp-analyse.md — 5 technische Unterschiede zu llama.cpp
- llama-cpp-vergleich.md — Fairer Speed-Vergleich
- ane-vs-ollama-gemessen.md — Korrektur: MLX schneller als ANE
- combo-vergleich.md — Alle Kombinationen vs Ollama
- realistisches-potenzial.md — Was tatsaechlich moeglich ist
- einsatzmoeglichkeiten.md — Wo kleine Modelle glaenzen
- finale-erkenntnisse.md — Was jede Unit kann und wann
- zwischenbericht.md — Projekt-Status
- abschlussbericht.md — Dieses Dokument

## Was haben wir herausgefunden?

### 1. ANE ist schnell — aber MLX ist schneller

| Engine | dim=1024, 28L, 128 tok Prefill |
|--------|-------------------------------|
| **Ollama (MLX, warm)** | **15,200 tok/s** |
| Unsere ANE | 6,864 tok/s |
| Unsere GPU (MPS) | 2,888 tok/s |
| Unser CPU/AMX | 1,226 tok/s |

MLX ist 2.2x schneller als unsere beste ANE Konfiguration.
Grund: Quantisierte Weights, Metal4 Hardware-Matmul, Zero-Copy Unified Memory,
Jahre Apple-Ingenieur-Optimierung.

### 2. ANE hat harte Limits die nicht umgehbar sind

| Feature | Status | Ursache |
|---------|--------|---------|
| SRAM 32MB | Attention bricht ab dim>1024 | Hardware-Limit |
| INT8 Weights | Compiler lehnt ab | MIL Typ-Mismatch |
| Dynamic Weights | Scheitert ab IC≥256 | Compiler-Limit |
| Sliding Window | 0% Speedup | Matmul berechnet immer voll |
| Decode (KV Cache) | Nicht moeglich | Dynamic Weights noetig |

### 3. Apple baut MLX fuer IHRE Chips — nicht ANE fuer LLMs

- Apple veroeffentlichte MLX im Dezember 2023
- MLX ist optimiert fuer Apple GPU (Metal)
- M5 hat Neural Accelerators IN der GPU (nicht ANE)
- Ollama wechselte im Maerz 2026 von llama.cpp zu MLX (+93% Speed)
- Kein Mainstream-Framework nutzt ANE fuer LLM Inference

### 4. AMX ist schneller als GPU bei dim≤1024 fuer FFN

Einzige wirklich neue Entdeckung:
- AMX: 1,604 GFLOPS bei dim=768 FFN
- GPU: 886 GFLOPS bei dim=768 FFN
- AMX ist 1.8x schneller

Aber irrelevant weil MLX die GPU besser nutzt als unsere MPS Implementation.

### 5. Unsere GPU Kernels sind 2-3x langsamer als llama.cpp/MLX

| Technik | llama.cpp hat | Wir haben | Fehlt |
|---------|-------------|-----------|-------|
| Multi-Row MatVec | Ja (110 Kernels) | Ja (1 Kernel, +40%) | 109 Kernels |
| Flash Attention | Ja | Nein | Fundamental |
| SIMD Group Matrix | Ja | Nein | Metal-Expertise |
| Metal4 Tensor API | Ja | Nein | Hardware-Zugang |
| Fused RMSNorm | Ja | Ja (+0% weil GPU pipelined) | — |

## Warum das Projekt scheitert

**Wir kaempfen gegen Apple selbst.**

Apple baut die Chips (M4, M5). Apple baut das Framework (MLX). Apple optimiert
beides zusammen. Gegen diese vertikale Integration kann Reverse-Engineering
nicht gewinnen.

Konkret:
- MLX nutzt Metal4 Tensor API (mpp::matmul2d) — Apple's eigene Hardware-Matmul
- MLX nutzt Unified Memory nativ — kein IOSurface-Overhead
- MLX nutzt quantisierte Weights — halb so viel Memory Bandwidth
- MLX wird von Apple aktiv weiterentwickelt
- ANE wird von Apple NICHT fuer LLMs vorgesehen

## Was die Arbeit wert war

1. **Informierte Entscheidung:** Wir wissen WARUM es nicht geht — mit Zahlen belegt
2. **Hardware-Wissen:** Alle 5 M4 Units systematisch charakterisiert
3. **ANE Limits:** 6 Features getestet, 4 scheitern — nirgendwo sonst dokumentiert
4. **Korrektheitsbeweis:** ANE kann Transformer korrekt ausfuehren (32/32 Match)
5. **Multi-Row MatVec:** +40% GPU Decode mit einer Kernel-Aenderung

## Verwandte Projekte — Status

| Projekt | Was | Ergebnis |
|---------|-----|---------|
| Fiber-Inference | ANE+GPU LLM Inference | ANE verliert gegen MLX |
| M4_RE | Hardware Reverse Engineering | Wertvolle Daten, 33 Experimente |
| ANE-Training | Training auf ANE | Funktioniert, aber kleine Modelle |
| fiber-train | ANE+GPU Training | GPU-only gewinnt |
| gdc-lm | Custom Conv-Architektur | Nie fertig, ueberholt durch MLX |

## Empfehlung

Fuer lokale LLM Inference auf Apple Silicon: **Ollama mit MLX Backend verwenden.**
Es ist Open Source, von Apple optimiert, und schneller als alles was wir bauen koennen.

Falls weiterhin an Apple Silicon ML gearbeitet werden soll:
**MLX als Basis nehmen und dort erweitern** — nicht eigene Engine von Grund auf.
