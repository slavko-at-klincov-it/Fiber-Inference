# CLAUDE.md — Fiber-Inference

## Was dieses Projekt ist

Multi-Accelerator LLM Inference Engine fuer Apple M4 (16GB).
Kombiniert ANE Attention + GPU FFN (Fiber-Loading) mit mmap/SSD Offloading
um groessere Modelle und laengere Kontexte zu ermoeglichen als jedes existierende Framework.

## Status

Phase 1 fertig: GPU-only Inference mit TinyLlama-1.1B Q4_K_M.
- **40 tok/s** Decode, **61 MB RSS** [MEASURED]
- Zero-copy mmap Weights, Single Command Buffer, FP16 Buffers, SIMD Reductions
- Q4_K + Q6_K + Q5_K Dequant-MatVec Fused Kernels

## Hardware

- Apple M4 Mac Mini: 10-core CPU (4P+6E), 10-core GPU, 16-core ANE (38 TOPS)
- 16 GB Unified Memory, 120 GB/s Bandwidth
- Interne SSD: ~7 GB/s Read (Modelle liegen jetzt lokal)

## Verwandtes Projekt

Baut auf bewiesenen Techniken aus M4_RE auf:
- **M4_RE**: `/Users/slavkoklincov/Documents/ClaudeCodeFolder/M4_RE`
- **ANE-Training**: `/Users/slavkoklincov/Documents/ClaudeCodeFolder/ANE-Training`
- **libane**: C API fuer direkten ANE-Zugang (symlinked in lib/)

## Architektur (Ziel)

```
Token Loop:
  For each layer:
    ANE: Fused SDPA (RMSNorm+QKV+RoPE+Attn+Wo)     -- 7.5x schneller als GPU
    GPU: FFN (Dequant+W1+SiLU+W3+W2)                 -- parallel zu ANE
    CPU: Residual Add                                  -- nur Pointer-Ops
    SSD: Prefetch naechster Layer (E-Core background)  -- keine GPU-Interferenz
```

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

- Phase 1 Decode: 40 tok/s, 61 MB RSS [MEASURED]
- Phase 1 Prefill: 24 tok/s [MEASURED]
- ANE Attention: 7.5x schneller als GPU (Exp 28) [MEASURED in M4_RE]
- Fiber Forward: 41% schneller als single-accelerator (Exp 09) [MEASURED in M4_RE]
- IOSurface Bridge: 0.048ms zero-copy (Exp 09) [MEASURED in M4_RE]
- mmap 26GB Model: 9.2GB RSS (Exp 17) [MEASURED in M4_RE]

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
