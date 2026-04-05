---
title: Fiber-Inference — Dokumentation
---

# Fiber-Inference

Systematische Evaluation aller 5 Compute Units des Apple M4 für LLM Inference.

## Research Paper

**[Vollständiges Paper (27 Seiten)](paper)** — Alle Messungen, Analysen und Schlussfolgerungen.

## Einzelne Findings

### Kernmessungen
- [Hardware Sweep](hardware-sweep) — ANE/GPU/AMX GFLOPS pro Dimension
- [ANE Korrektheitsbeweis](proof) — 32/32 Token-Match
- [Proof Sweep](proof-sweep) — Systematischer Proof über alle Konfigurationen
- [ANE vs Ollama (MLX)](ane-vs-ollama-gemessen) — Fairer Vergleich

### Compute Units
- [Finale Erkenntnisse](finale-erkenntnisse) — Was jede Unit kann und wann
- [Parallel Units Test](parallel-units-test) — Multi-Unit = langsamer (-40%)
- [AMX/FFN Findings](phase2-findings) — AMX 1,8× schneller als GPU

### ANE Deep Dive
- [ANE Decode Findings](ane-decode-findings) — Warum ANE kein Decode kann
- [Large Model ANE](large-model-ane) — ANE bei dim > 1024
- [INT8 Findings](int8-findings) — ANE unterstützt kein INT8
- [Dynamic Weights](dynamic-weights-findings) — Compiler-Limits
- [Sliding Window](sliding-window-findings) — Nicht unterstützt

### Vergleiche & Analyse
- [Combo-Vergleich](combo-vergleich) — Alle Unit-Kombinationen
- [Large Model Sweep](large-model-sweep) — Skalierung über Dimensionen
- [Limits Findings](limits-findings) — Wo die Hardware aufhört
- [llama.cpp Analyse](llama-cpp-analyse) — Vergleich mit llama.cpp
- [llama.cpp Vergleich](llama-cpp-vergleich) — Benchmarks
- [MLX Fork Chancen](mlx-fork-chancen) — Lohnt sich ein MLX-Fork?
- [Realistisches Potenzial](realistisches-potenzial) — Was wirklich möglich ist

### Projektberichte
- [Abschlussbericht](abschlussbericht) — Projektzusammenfassung
- [Zwischenbericht](zwischenbericht) — Stand der Arbeit
- [Architektur-Grundlagen](architektur-grundlagen) — LLM + Apple Silicon Basics
- [Einsatzmöglichkeiten](einsatzmoeglichkeiten) — Wo Fiber-Inference Sinn macht

---

[GitHub Repository](https://github.com/slavko-at-klincov-it/Fiber-Inference)
