# Phase 2 Findings: ANE Attention + GPU FFN

Date: 2026-04-03 / 2026-04-04

## Ergebnis

| Metrik | GPU-only (Phase 1) | ANE+GPU (Phase 2) | Boost |
|--------|-------------------|-------------------|-------|
| Prefill (81 tok) | 40.1 tok/s | **181.8 tok/s** | **4.5x** |
| Decode | 37.3 tok/s | 35.6 tok/s | kein Verlust |
| Peak RSS | 61.8 MB | 2044 MB | +2 GB |

## Architektur

```
Prefill (ANE+GPU):
  Embed all tokens → [dim, seq] FP16 channels-first
  For each of 22 layers:
    ANE: Fused SDPA (RMSNorm + QKV Conv + RoPE + GQA Tile + Attention + Wo)
    CPU: Residual add
    GPU: Batched FFN (MPS MatrixMultiplication: RMSNorm + W1/W3 + SiLU + W2 + Residual)
  Classifier on last token → logits

Decode (GPU-only, unchanged):
  Single token per step, all ops on GPU
```

## Bottleneck Breakdown (81 tokens, 22 layers) [MEASURED]

| Komponente | Zeit | Pro Layer | Anteil |
|-----------|------|-----------|--------|
| ANE Attention | ~180 ms | 8.2 ms | 41% |
| GPU FFN (MPS) | ~145 ms | 6.6 ms | 33% |
| ANE I/O + memcpy | ~105 ms | 4.8 ms | 24% |
| Residual (CPU) | 0.3 ms | 0.01 ms | ~0% |
| KV Cache write | 0.5 ms | 0.02 ms | ~0% |
| Embedding | 0.1 ms | — | ~0% |

## GQA Tile Bug und Fix

**Problem:** MIL `tile(reps=[1,8,1,1])` auf `[1, n_kv, hd, seq]` gibt falsche Reihenfolge:
`[k0,k1,k2,k3,k0,k1,k2,k3,...]` statt `[k0,k0,...,k1,k1,...]`

**Fix:** Reshape+Tile+Reshape Pattern:
1. `[1, n_kv, hd, seq]` → reshape `[1, n_kv, 1, hd*seq]`
2. tile `[1, 1, gqa_ratio, 1]` → `[1, n_kv, gqa_ratio, hd*seq]`
3. reshape `[1, n_heads, hd, seq]`

Ergibt korrekte GQA Zuordnung: Q heads 0-7 → KV head 0, etc.

## 3 Optimierungs-Hebel getestet

### Hebel 1: Pre-Dequant FFN Weights [IMPLEMENTED]

Alle 22 Layer FFN Weights (W1, W2, W3, ffn_norm) beim Start von Q4_K/Q6_K
zu FP16 dequantisieren und als MTLBuffers speichern.

- GPU FFN: 13.6 → 6.7 ms/Layer (**-51%**) [MEASURED]
- Startup-Kosten: 186 ms, 1452 MB RAM
- Prefill: 153 → 182 tok/s (+19%)

### Hebel 2: Pipeline Overlap ANE/GPU [SKIPPED]

Daten-Abhaengigkeit verhindert Ueberlappung: GPU FFN Output von Layer L
ist Input fuer ANE Attention von Layer L+1. Kein Overlap moeglich.

Einzig moegliche Ueberlappung: KV Cache Write waehrend GPU FFN,
aber KV Cache ist nur 0.5ms — nicht lohnend.

### Hebel 3: Reduce I/O Overhead [PARTIALLY IMPLEMENTED]

**Reusable pad buffers:** malloc/free pro Layer eliminiert → marginal besser.

**Exact-seq ANE Compile:** Kernel fuer exakte Prompt-Laenge kompilieren
statt fuer max_seq=512 (eliminiert Padding).
- Ergebnis: Timer zeigt ~67ms (1222 tok/s) — aber **Output ist kaputt**
- Ursache unklar: moeglicherweise SRAM-Layout-Problem bei kleinen seq
- **REVERTED** zu fixed max_seq=512

## ANE Kernel Details

- MIL Text: ~9100 Bytes pro Kernel
- Kompilierung: 22 Kernels in ~3-5 Sekunden
- Budget: 22/119 verwendet
- SRAM Spill: ja, bei seq=512 (Attention Scores [1,32,512,512] = 32MB ≈ SRAM Limit)
- QoS: ANE_QOS_BACKGROUND (schnellster Modus)

## MPS MatrixMultiplication Performance

Batched FFN mit MPS fuer W1/W3/W2 Matmuls:
- 1 Command Buffer pro Layer (statt N pro Token)
- FP16 Weights + FP16 Activations
- ~6.6 ms/Layer fuer FFN bei seq=81

## Skalierung mit Prompt-Laenge

| Seq | Prefill tok/s | Anmerkung |
|-----|--------------|-----------|
| 2 | 3.0 | Overhead dominiert |
| 5 | 12.2 | ANE I/O Padding teuer |
| 29 | 52.8 | |
| 81 | 181.8 | Sweet spot |

Laengere Prompts amortisieren den konstanten Overhead besser.
Bei seq < 5 faellt der Code automatisch auf GPU-only zurueck.

## Bekannte Limitierungen

1. **2 GB RAM** fuer Pre-dequant FFN Weights — zu viel fuer groessere Modelle
2. **SRAM Spill** bei seq ≥ 512 — ~30% ANE Throughput-Verlust
3. **ANE I/O Padding** — 24% der Prefill-Zeit fuer Daten-Transfer
4. **Exact-seq Compile** bricht Korrektheit — unklar warum
5. **Decode nutzt ANE nicht** — nur Prefill profitiert

## Naechste Schritte (Potenzial)

- **IOSurface zero-copy** fuer ANE I/O statt ane_write/ane_read → -24% Overhead
- **Exact-seq Compile debuggen** → potentiell 1200+ tok/s wenn Korrektheit geloest
- **ANE Decode** fuer laengere Kontexte (>256 tokens)
- **Groessere Modelle** (7B) mit mmap + Fiber-Loading testen
