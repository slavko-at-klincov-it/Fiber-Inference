# CLAUDE.md — Fiber-Inference

## Was dieses Projekt ist

Multi-Accelerator LLM Inference Engine fuer Apple M4 (16GB).
Kombiniert ANE Attention + GPU FFN (Fiber-Loading) mit mmap/SSD Offloading
um groessere Modelle und laengere Kontexte zu ermoeglichen als jedes existierende Framework.

## Status

Fiber-768 Architektur: ANE-native LLM Inference.
- **10,241 tok/s Prefill** mit ANE Attention + ANE FFN [MEASURED]
- 256x Boost gegenueber GPU-only Baseline (40 tok/s)
- Eigene Architektur: dim=768, 12 heads, 4x FFN, 24 Layers (~800M params)
- Details: `docs/fiber768-findings.md`, `docs/phase2-findings.md`

## Hardware — Die 5 Compute Units

Apple M4 Mac Mini hat 5 programmierbare Compute Units die ALLE PARALLEL laufen
(103.3% Effizienz gemessen, null Interferenz — M4_RE Exp 06).

| # | Unit              | Peak Throughput                    | Zugang                        | Status          |
|---|-------------------|------------------------------------|-------------------------------|-----------------|
| 1 | GPU (10-core)     | ~7 TFLOPS FP16                     | Metal Compute Shaders         | Aktiv (Phase 1) |
| 2 | ANE (16-core)     | 19 TFLOPS FP16 / 38 TOPS INT8     | CoreML / libane               | Phase 2 Ziel    |
| 3 | AMX/SME (P-cluster)| 2 TFLOPS FP32 / 4 TOPS INT8      | Accelerate oder ARM SME ASM   | Ungenutzt       |
| 4 | CPU P-cores (4x)  | ~0.2 TFLOPS NEON                   | C / NEON Intrinsics           | Minimal         |
| 5 | CPU E-cores (6x)  | ~0.36 TFLOPS NEON                  | C / NEON Intrinsics           | Minimal         |

- **Gesamt parallel: ~18.4 TFLOPS** — aktuell nutzen wir nur GPU (~7 TFLOPS = 38% des Chips)
- 16 GB Unified Memory, 120 GB/s Bandwidth (geteilt zwischen allen Units)
- Interne SSD: ~7 GB/s Read (Modelle liegen jetzt lokal)

### AMX/SME Detail

Der AMX/SME ist ein Matrix-Coprozessor im CPU-Cluster. Auf M4 ueber den
offiziellen ARM SME Standard programmierbar (FMOPA, SMOPA Instruktionen).
- FP32 und FP16 gleiche Throughput (~2 TFLOPS) — FP16 ist widening-only
- INT8: 4 TOPS (2x FP32) — relevant fuer quantisierte Weights
- Accelerate/cblas nutzt AMX automatisch, aber custom SME Kernels sind 1.21x schneller
- Laeuft parallel zu GPU und ANE auf geteiltem Unified Memory

### Kein existierendes Framework nutzt alle Units

Stand 2026-04: MLX nutzt nur GPU. llama.cpp nutzt GPU + Accelerate (nicht parallel).
CoreML nutzt nur ANE. ANEMLL nur ANE. **Niemand** macht GPU+ANE+AMX parallel.
Das ist die Luecke die Fiber-Inference schliesst.

## Verwandte Projekte

- **M4_RE**: `/Users/slavkoklincov/Code/M4_RE` — 33 Hardware-Experimente, alle [MEASURED]
- **ANE-Training**: `/Users/slavkoklincov/Code/ANE-Training` — ANE Training via libane (47.6 tok/s Inference)
- **fiber-train**: `/Users/slavkoklincov/Code/fiber-train` — Training-Experimente (GPU-only gewinnt dort)
- **gdc-lm**: `/Users/slavkoklincov/Code/gdc-lm` — Custom LLM Architektur vom Hardware aus designed
- **libane**: C API fuer direkten ANE-Zugang (symlinked in lib/)

## Architektur (Ziel) — Alle 5 Units

```
Token Loop:
  For each layer:
    ANE:     Fused SDPA (RMSNorm+QKV+RoPE+Attn+Wo)    -- 19 TFLOPS, 7.5x schneller als GPU
    GPU:     FFN (Dequant+W1+SiLU+W3+W2)               -- ~7 TFLOPS, parallel zu ANE
    AMX/SME: Output Projection / Embedding / Prefill    -- 2 TFLOPS, parallel zu GPU+ANE
    P-cores: Residual Add, Sampling, RoPE               -- Pointer-Ops, Orchestrierung
    E-cores: SSD Prefetch naechster Layer                -- Background, keine Interferenz
```

Ziel: ~18.4 TFLOPS statt ~7 TFLOPS = **2-3x Boost** gegenueber GPU-only Frameworks.

## Architektur (aktuell — Phase 1)

```
Token Loop:
  gpu_embed(token) → FP16 buf_x
  Single Metal Command Buffer:
    For each of 22 layers (one encoder, ~10 dispatches/layer):
      RMSNorm → Q4K/Q6K MatVec (QKV) → Fused RoPE → Fused KV-Store
      → Attention Decode → MatVec (Wo) → Residual
      → RMSNorm → MatVec (W1,W3) → SiLU → MatVec (W2) → Residual
    Classifier: RMSNorm → MatVec (output) → F32 Logits
  waitUntilCompleted
  sampler_sample(logits) → next_token
```

## Bewiesene Zahlen

### Fiber-Inference
- Phase 1 Decode: 40 tok/s, 61 MB RSS [MEASURED]
- Phase 1 Prefill: 24 tok/s [MEASURED]

### M4_RE Experimente (Grundlage fuer Architektur)
- Exp 06: Alle 3 Accelerators parallel = 18.4 TFLOPS, 103.3% Effizienz [MEASURED]
- Exp 09: Fiber Forward (ANE+GPU) = 41% schneller als single-accelerator [MEASURED]
- Exp 09: IOSurface Bridge = 0.048ms zero-copy [MEASURED]
- Exp 12: SME auf M4 = 1,576 GFLOPS FP32 (tiled), 1,775 GOPS INT8 [MEASURED]
- Exp 17: mmap 26GB Model = 9.2GB RSS [MEASURED]
- Exp 28: ANE Attention = 7.5x schneller als GPU [MEASURED]
- Exp 31: UMA Coherency = CPU→GPU sofort sichtbar, kein Flush noetig [MEASURED]

### AMX/SME Benchmarks (M4_RE)
- cblas/AMX FP32: 1,929 GFLOPS peak [MEASURED]
- SME FP32 tiled: 1,576 GFLOPS (0.82x cblas) [MEASURED]
- SME INT8→INT32: 1,775 GOPS [MEASURED]
- AMX FP16: nur widening (FP16→FP32), KEIN 2x Vorteil ueber FP32 [MEASURED]
- Custom SME Kernels: 1.21x schneller als Accelerate moeglich [MEASURED, Uni Jena Paper]

## Build

```bash
make
./fiber-inference --model models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --prompt "Hello"
```

## Coding Conventions

- Objective-C/C fuer alles (maximale Kontrolle ueber Metal/MPS/ANE/mmap)
- Metal Shading Language fuer GPU Compute Kernels
- Markierungen: [MEASURED], [THEORETICAL], [UNEXPLORED]
- Benchmarks: immer mit Einheiten, Warmup, mehrere Runs, Median
- Alle Aktivierungen FP16, interne Akkumulation F32
- NIEMALS waitUntilCompleted zwischen Layern — alles in einem Command Buffer
- Q6_K Dequant: Nibble-Mapping exakt nach GGML (ql[l]&0xF, ql[l+32]&0xF, ql[l]>>4, ql[l+32]>>4)

## Sprache

User spricht Deutsch. Code und Kommentare auf Englisch.
