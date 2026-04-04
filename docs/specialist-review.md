# Spezialisten-Review: Kritische Bewertung + Ungenutztes Potenzial

Date: 2026-04-04

## ML-Spezialist: Kritische Bewertung

### Rating: 6/10

### Staerken
- Genuinely novel: ANE fuer LLM Inference via private APIs — das hat niemand sonst gemacht
- Wertvolle Hardware-Charakterisierung (Sweep-Daten sind originale Forschung)
- Ehrliche Dokumentation der Limitierungen
- Beweis dass ANE korrekte Transformer-Layer ausfuehren kann
- libane C API ist eine Engineering-Leistung

### Schwaechen (ehrlich)

**1. Speedup-Zahlen sind aufgeblaeht**
- Die 29x-126x vergleichen gegen **schwache CPU Baseline** (single-token cblas_sgemv)
- Faire Vergleiche waeren gegen llama.cpp oder MLX mit batched Prefill
- llama.cpp auf M4 fuer 110M-Modelle: geschaetzt 3,000-5,000 tok/s Prefill
- **Realistischer ANE-Vorteil: 2-4x gegenueber optimiertem GPU Prefill** (nicht 100x+)

**2. Kein Decode-Pfad**
- Alle ANE-Zahlen sind Prefill-only
- Decode (Token-by-Token) ist das was User spueren
- GPU Decode: 37 tok/s — das ist die aktuelle User-Erfahrung
- Ohne ANE Decode ist das Projekt ein "schneller Prefiller, normaler Decoder"

**3. Token-Match reicht nicht als Korrektheits-Beweis**
- Argmax-Match beweist nur dass der Top-Token gleich ist
- Logit-Verteilung kann komplett anders sein (FP16 vs FP32)
- Ueber 500+ Tokens wuerden Rundungsfehler akkumulieren
- Noetig: Perplexity-Evaluation auf Benchmark-Dataset, KL-Divergenz

**4. Private API Abhaengigkeit**
- libane nutzt reverse-engineered private Classes
- Kann bei jedem macOS Update brechen
- Nicht App Store kompatibel, nicht shippbar

**5. dim ≤ 1024 und seq ≤ 256 limitiert Modellqualitaet**
- Alle Produktions-LLMs (Llama 8B, Qwen 8B) nutzen dim=3584-4096
- dim=768-1024 ergibt 100M-1B Parameter — limitierte Qualitaet
- seq=256 schliesst viele Anwendungen aus (RAG, Code, Dokumente)

### Was fehlt fuer "usable inference engine"
1. ANE Decode implementieren
2. Fairer A/B Benchmark gegen llama.cpp/MLX auf gleicher Hardware
3. Echtes trainiertes Modell bei dim=768-1024 mit koherentem Text
4. INT8 Quantisierung auf ANE (38 TOPS statt 19 TFLOPS)
5. Perplexity-Evaluation

---

## Optimierungs-Spezialist: Ungenutztes Potenzial

### Gefundene Bugs

**Qwen3-4B Compile-Fehler: INTEGER DIVISION BUG!**
- Qwen3-4B hat 20 heads, 8 kv_heads → gqa_ratio = 20/8 = 2 (integer truncation!)
- Richtig waere 2.5, was nicht ganzzahlig ist → Tile reshape Mismatch
- **Fix:** GQA Ratio muss n_heads % n_kv_heads == 0 pruefen, oder Qwen3-4B hat kv_heads=4 (nicht 8)
- Trivial zu fixen

### Top-8 Optimierungen nach Impact

| Prio | Was | Erwarteter Impact | Aufwand |
|------|-----|-------------------|---------|
| **1** | **ANE Decode (padded seq=128)** | **5-10x Decode (37→200-400 tok/s)** | Mittel |
| **2** | **Qwen3-4B GQA Bug fixen** | Unblocked Qwen3-4B (1,864+ tok/s) | Trivial |
| 3 | Full-Layer Fusion (Attn+FFN ein Kernel) | 10-15% Prefill | Klein |
| 4 | Dynamic Weights fuer 7B+ | Ermoeglicht grosse Modelle, 2 statt 64 Compiles | Mittel |
| 5 | head_dim=128 in Fiber-768 | ~18% schnellere Attention (gratis) | Trivial |
| 6 | Speculative ANE Decode (8-Token Batch) | 4-8x effektiver Decode | Hoch |
| 7 | KV-Cache auf IOSurface | Eliminiert Copy-Overhead, ermoeglicht ANE Decode | Mittel |
| 8 | dim=1280 Sweet Spot testen | Bessere Qualitaet/Speed Balance als 1024 | Klein |

### Groesster einzelner Hebel: ANE Decode

Aktuell: Prefill 10,241 tok/s (ANE) + Decode 37 tok/s (GPU) = **Decode ist 277x langsamer als Prefill.**

Ansatz: ANE Kernel mit seq=128 kompilieren, 1 echten Token + 127 Padding einfuegen.
Selbst bei 50% Effizienz durch Padding-Waste: 0.5ms/Layer statt 5ms/Layer = **10x Decode Speedup → ~370 tok/s.**

### Ungetestete Kombinationen
- dim=1280 (Luecke zwischen 1024 und 2048)
- head_dim=128 bei dim=768 (6 heads statt 12)
- FFN ratio > 4x auf ANE
- seq=192 (gemessener Sweet Spot, nie im Architektur verwendet)
- Chunked Attention: 2x128 fuer seq=256

---

## Fazit beider Spezialisten

Die Speedup-Zahlen sind **real fuer das was sie messen** (ANE vs unoptimierte CPU Baseline),
aber **uebertrieben im Kontext von Produktions-Engines**. Der realistische Vorteil gegenueber
optimiertem GPU Inference ist 2-4x fuer Prefill.

Der **echte Durchbruch waere ANE Decode** — das wuerde die User-spuerbare Latenz um 5-10x
verbessern, nicht nur den Prefill.

Das Projekt ist **wertvolle Hardware-Forschung** (6/10), nicht eine fertige Inference-Engine.
Die wertvollsten Beitraege sind die Hardware-Daten und die libane API.
