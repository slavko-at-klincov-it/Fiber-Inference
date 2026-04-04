# Architektur-Grundlagen: LLM Begriffe + Warum Apple Silicon eine eigene Architektur braucht

Date: 2026-04-04

## Was ist "Dimension" (dim)?

Dimension ist NICHT die Modellgroesse. Es ist die **Breite des internen Vektors** —
wie viele Zahlen ein einzelnes Token repraesentiert waehrend es durch das Modell fliesst.

```
Token "Hallo" → Embedding → [768 Zahlen] → durch alle Layer → [768 Zahlen] → naechstes Wort
                              ^^^^^^^^^^^^
                              DAS ist dim=768
```

Die **Parameter-Anzahl** (1B, 4B, 7B, 30B) ergibt sich aus der Kombination:

| Parameter | Was es bestimmt | Beispiel TinyLlama-1.1B |
|-----------|----------------|------------------------|
| dim | Breite des Vektors | 2048 |
| n_layers | Tiefe (wie viele Layer) | 22 |
| n_heads | Anzahl Attention-Koepfe | 32 |
| head_dim | Groesse pro Kopf | 64 |
| ffn_dim | Breite der FFN-Schicht | 5632 |
| vocab_size | Wortschatz | 32000 |

Grob: Parameter ≈ n_layers × (4 × dim² + 3 × dim × ffn_dim)
→ 22 × (4 × 2048² + 3 × 2048 × 5632) ≈ 1.1 Milliarden

## Welche LLM-Architekturen gibt es?

**Fast alle lokalen LLMs nutzen die gleiche Grundarchitektur: Transformer mit Variationen.**

| Modell | Hersteller | Architektur-Typ | Was ist anders? |
|--------|-----------|----------------|-----------------|
| Llama 3 | Meta | Transformer | GQA, RoPE, SwiGLU FFN — der "Standard" |
| Qwen 3 | Alibaba | Llama-Variante | Gleiche Basis, anderer Tokenizer + Training |
| Gemma 3 | Google | Llama-Variante | Sliding Window Attention, RMSNorm Tweaks |
| Phi-4 | Microsoft | Llama-Variante | Dichteres Training, weniger Layer |
| Mistral | Mistral AI | Llama + MoE | Sliding Window, Mixture of Experts |
| DeepSeek | DeepSeek | Llama + MoE | Multi-head Latent Attention (MLA) |

**Die Wahrheit: Sie sind alle fast gleich.** Alle basieren auf dem Transformer mit:
- RMSNorm (statt LayerNorm)
- RoPE (Rotary Position Encoding)
- GQA (Grouped Query Attention)
- SwiGLU FFN (statt ReLU)

Die Unterschiede sind im **Training** (Daten, Methoden) und in **kleinen Architektur-Tweaks** —
nicht in fundamental verschiedenen Designs.

## Ollama vs Qwen vs Gemma — was ist was?

**Ollama ist KEIN Modell.** Ollama ist ein Tool das Modelle laedt und ausfuehrt.

```
Ausfuehrungs-Engines (WIE es laeuft):
  Ollama, llama.cpp, MLX, vLLM, Fiber-Inference
  → Laden GGUF/SafeTensors Dateien und rechnen die Matrizen durch

Modelle (WAS laeuft):
  Qwen, Gemma, Llama, Phi, Mistral, DeepSeek
  → Die eigentliche Architektur + trainierte Gewichte (Wissen)

Quantisierung (WIE KOMPAKT):
  Q4_K, Q6_K, Q8, FP16
  → Wie stark die Gewichte komprimiert werden (kleiner = schneller, aber ungenauer)
```

## Warum Apple Silicon eine eigene Architektur braucht

Alle diese Modelle (Llama, Qwen, Gemma) wurden **fuer NVIDIA GPUs designed**.
Sie laufen auf Apple Silicon *trotz* des Chips, nicht *wegen* ihm.

### NVIDIA vs Apple Silicon — fundamental verschieden

| Eigenschaft | NVIDIA (A100/H100) | Apple M4 |
|-------------|-------------------|----------|
| Compute Units | 1 (GPU) | **5 (GPU + ANE + AMX + P-cores + E-cores)** |
| Memory | Eigener VRAM (80GB) | Unified Memory (16GB, geteilt) |
| Bandwidth | 2 TB/s (HBM) | 120 GB/s (LPDDR5X) |
| Design-Ziel | Maximaler Durchsatz | Effizienz + Vielseitigkeit |
| ANE | Existiert nicht | 19 TFLOPS FP16, 38 TOPS INT8 |
| AMX | Existiert nicht | 2 TFLOPS FP32 |

### Was Standard-Transformer auf Apple Silicon verschwendet

Die Standard-Architektur ignoriert komplett:

1. **ANE ist 7.5x schneller als GPU fuer Attention** [MEASURED]
   → Aber kein Framework nutzt ANE fuer LLM Inference

2. **AMX schlaegt GPU fuer FFN bei dim ≤ 1024** [MEASURED]
   → Alle Frameworks nutzen nur GPU fuer FFN

3. **ANE hat 32MB SRAM** — bei seq > 256 bricht Performance ein [MEASURED]
   → Standard-Modelle nutzen seq=2048-128K ohne Ruecksicht auf SRAM

4. **Alle 5 Units laufen parallel mit 103% Effizienz** [MEASURED]
   → Kein Framework nutzt mehr als 1-2 Units gleichzeitig

5. **SSD mit 7 GB/s** kann Weights streamen [MEASURED]
   → Kein Framework nutzt SSD-Offloading fuer groessere Modelle

### Was eine Apple-Silicon-native Architektur anders machen wuerde

| Design-Entscheidung | Standard (NVIDIA) | Apple-Silicon-Native |
|---------------------|-------------------|---------------------|
| Attention | Auf GPU | **ANE** (7.5x schneller) |
| FFN | Auf GPU | **AMX** bei dim ≤ 1024, GPU bei dim > 1500 |
| dim Wahl | 2048-8192 (NVIDIA-optimiert) | **768-1024** (ANE + AMX Sweet Spot) |
| FFN Ratio | 2.67x-3x | **4x** (GPU-effizienter laut Sweep) |
| Max Attention seq | 2048-128K | **256** (ANE SRAM Limit), danach Sliding Window |
| Quantisierung | FP16/BF16 | **INT8** (ANE 38 TOPS, SME 4 TOPS) |
| Parallelitaet | Alles auf GPU | **ANE + AMX + GPU gleichzeitig** |
| Grosse Modelle | Mehr VRAM kaufen | **mmap/SSD Offloading** (7 GB/s) |

## Zusammenfassung

Es gibt aktuell **keine LLM-Architektur die fuer Apple Silicon designed wurde**.
Alle existierenden Modelle (Qwen, Gemma, Llama) sind NVIDIA-optimiert und werden
auf Apple Silicon nur "portiert" — mit massivem Performance-Verlust.

Fiber-Inference baut eine **Inference-Engine die den Chip richtig nutzt** (10.5x Prefill-Boost bewiesen).
Der naechste Schritt ist eine **Modell-Architektur die von den Hardware-Constraints aus designed ist** —
nicht ein bestehendes Modell schneller machen, sondern ein Modell bauen das den Chip von Anfang an optimal nutzt.

Die Sweep-Daten (`docs/hardware-sweep.md`) zeigen die optimalen Dimensionen:
- dim=768-1024, head_dim=64, FFN 4x, seq ≤ 256
- ANE fuer Attention, AMX fuer FFN, GPU fuer Decode/Klassifikation
- Geschaetztes Potenzial: **1700+ tok/s Prefill** wenn alle Units parallel laufen
