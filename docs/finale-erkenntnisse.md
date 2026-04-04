# Finale Erkenntnisse: Was jede Compute Unit kann und wann

Date: 2026-04-05

## Die 5 M4 Compute Units — Endgueltige Bewertung [ALLES GEMESSEN]

| Unit | Peak | Staerke | Schwaeche | Bester Einsatz |
|------|------|---------|-----------|---------------|
| **ANE** | 19 TFLOPS FP16 | Schnellster Prefill bei dim≤1024, 0.4-0.8ms/Layer | SRAM 32MB Limit, kein Decode, kein INT8, kein Dynamic | Prefill fuer kleine Modelle (≤1B) |
| **GPU** | ~7 TFLOPS FP16 | Flexibel, Decode, quantisierte Weights, Flash Attention | Langsamer als ANE bei dim≤1024 | Decode, grosse Modelle (2B+) |
| **AMX** | ~1.6 TFLOPS FP32 | Schlaegt GPU bei dim≤1024 fuer FFN | Nur FP32, kein FP16 Vorteil | FFN bei kleinen dim, falls GPU besetzt |
| **CPU P-cores** | ~0.2 TFLOPS | Orchestrierung, triviale Ops | Langsam fuer Matmul | RoPE, Sampling, Residual |
| **CPU E-cores** | ~0.36 TFLOPS | Background, niedrige Power | Langsam | SSD Prefetch |

## Wann welche Unit gewinnt [GEMESSEN]

### Prefill (Prompt verarbeiten)

| Modellgroesse | dim | Bester Accelerator | tok/s | vs llama.cpp |
|--------------|-----|-------------------|-------|-------------|
| ~110M | 768 | **ANE** | 1,219 | **2.5x schneller** |
| ~500M | 1024 | **ANE** | ~800 | ~1.5x schneller |
| ~1.1B | 2048 | **GPU (llama.cpp)** | 482 | ANE 7.7x langsamer |
| ~4B | 2560 | **GPU** | ~150 | ANE SRAM-limitiert |
| ~8B | 4096 | **GPU** | ~100 | ANE stark limitiert |

**Kipppunkt: dim ≈ 1024-1500.** Darunter ANE, darueber GPU.

### Decode (Text generieren, Token fuer Token)

| Engine | TinyLlama 1.1B Decode | Methode |
|--------|----------------------|---------|
| llama.cpp | **119 tok/s** | Metal GPU, 110 optimierte Kernels |
| Fiber (Multi-Row) | 52 tok/s | Metal GPU, 1 optimierter Kernel |
| Fiber (Original) | 37 tok/s | Metal GPU, unoptimiert |
| Fiber (CPU/AMX) | 69 tok/s | cblas_sgemv |

**GPU gewinnt immer fuer Decode** — ANE kann kein inkrementelles Decode (Dynamic Weights, KV Cache als Input scheitern bei dim≥256).

## Was funktioniert und was nicht [GETESTET]

| Feature | Status | Ergebnis | Dokumentation |
|---------|--------|----------|--------------|
| ANE Prefill (baked weights) | **FUNKTIONIERT** | 10K+ tok/s bei dim=768 | docs/fiber768-findings.md |
| ANE Korrektheit | **BEWIESEN** | 32/32 Token Match | docs/proof.md |
| Kohaerenter Text auf ANE | **FUNKTIONIERT** | Stories-110M Kindergeschichten | docs/fiber768-findings.md |
| ANE Decode (Re-Prefill) | Korrekt, langsam | 65 tok/s (O(n²)) | docs/ane-decode-findings.md |
| ANE Decode (Hybrid CPU+ANE) | Korrekt, nicht schneller | 66 tok/s | docs/ane-decode-findings.md |
| ANE INT8 | **SCHEITERT** | Compiler lehnt ab | docs/int8-findings.md |
| ANE Dynamic Weights | **SCHEITERT** ab IC≥256 | Compiler-Limit | docs/dynamic-weights-findings.md |
| ANE Sliding Window | **NUTZLOS** | 0% Speed-Gewinn | docs/sliding-window-findings.md |
| GPU Multi-Row MatVec | **FUNKTIONIERT** | +40% Decode (37→52 tok/s) | Commit e0201e4 |
| GPU Fused RMSNorm | Korrekt, kein Speed | +0% | Commit a4c20b9 |

## Die optimale Architektur (basierend auf allen Tests)

```
Modelle ≤ 1B (dim ≤ 1024):
  Prefill: ANE (2-3x schneller als GPU)
  Decode:  GPU (ANE kann nicht)
  → Time-to-First-Token Vorteil

Modelle 2B-9B (dim 2048-4096):
  Prefill: GPU (ANE hat SRAM-Probleme)
  Decode:  GPU
  → Kein ANE-Vorteil, GPU ist besser

Fuer alle Modelle:
  Decode ist der Bottleneck fuer User-Erfahrung
  llama.cpp hat die besten GPU Decode Kernels (2x schneller als unsere)
```

## Vergleich mit der Konkurrenz [GEMESSEN + RECHERCHIERT]

| Engine | Nutzt ANE? | Decode 7B M4 | Prefill |
|--------|-----------|-------------|---------|
| llama.cpp | Nein | ~24 tok/s | ~220 tok/s |
| Ollama (MLX) | Nein | ~50 tok/s | ~400 tok/s |
| MLX | Nein | ~25-35 tok/s | ~300 tok/s |
| ANEMLL | Ja (CoreML) | ~9 tok/s | langsam |
| **Fiber** | **Ja (libane)** | 52 tok/s (1.1B) | **1,219 tok/s (110M)** |

**Niemand nutzt ANE effektiv fuer LLMs.** Unser ANE Prefill bei dim≤1024 ist einzigartig schnell. Aber fuer Decode und grosse Modelle gibt es keinen ANE-Vorteil.

## 35 Commits, 18 Dokumente, ~7000 Zeilen Code

Alles reproduzierbar:
```bash
# Kohaerenter Text auf ANE:
./fiber-inference --arch proof --prompt "Once upon a time"

# GPU Decode Benchmark:
./fiber-inference --model models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --prompt "Hello" --tokens 32 --no-ane

# llama.cpp Vergleich:
llama-bench -m models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -p 128 -n 32 -ngl 99

# Hardware Sweep:
cd bench && ./bench-sweep
```
