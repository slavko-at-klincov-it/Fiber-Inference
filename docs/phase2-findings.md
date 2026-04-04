# Phase 2 Findings: ANE Attention + GPU FFN

Date: 2026-04-03 / 2026-04-04

## Ergebnis (aktuell nach allen Optimierungen)

| Metrik | GPU-only (Phase 1) | ANE+GPU (Phase 2 final) | Boost |
|--------|-------------------|------------------------|-------|
| Prefill (81 tok) | 40.1 tok/s | **~420 tok/s** | **10.5x** |
| Decode | 37.3 tok/s | 36.6 tok/s | kein Verlust |
| Peak RSS | 61.8 MB | 1967 MB | +1.9 GB |

### Optimierungs-Verlauf

| Schritt | Prefill tok/s | Boost vs GPU-only |
|---------|--------------|-------------------|
| GPU-only Baseline | 40 | 1.0x |
| + ANE Attention (per-token FFN) | 153 | 3.8x |
| + Batched MPS FFN | 182 | 4.5x |
| + Pre-dequant FFN weights | 182 | 4.5x |
| + Exact-seq compile (max(128,seq)) | 417 | **10.4x** |
| + IOSurface zero-copy | 420 | **10.5x** |

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

---

## Runde 2: Exact-seq Fix + IOSurface Zero-Copy (2026-04-04)

### Exact-seq Compile Fix [IMPLEMENTED]

**Root Cause:** ANE Compiler braucht Minimum 128 fuer Matmul-Dimensionen.
Bei seq<128 entsteht Attention Scores `[1,32,seq,seq]` mit seq<128 → HWX Compiler-Fehler.

**Fix:** `ane_seq = max(128, n_prompt)` statt hardcoded 512.

**Impact:**

| Metrik | Vorher (seq=512) | Nachher (max(128,seq)) | Delta |
|--------|-----------------|----------------------|-------|
| ANE attn | 180 ms (8.2 ms/L) | 22 ms (1.0 ms/L) | **-88%** |
| GPU FFN | 145 ms (6.6 ms/L) | 100 ms (4.6 ms/L) | -31% |
| Unaccounted | 105 ms | 65 ms | -38% |
| **Prefill** | **182 tok/s** | **417 tok/s** | **2.3x** |

Die ANE Attention ist bei passender Sequenzlaenge extrem schnell (1.0 ms/Layer).
SRAM Spill tritt trotzdem auf (Weight-Tensoren > 32MB), aber das Impact ist bei
kleinerem seq deutlich geringer.

### IOSurface Zero-Copy [IMPLEMENTED]

Ersetze `ane_write()`/`ane_read()` mit direktem IOSurface-Zugriff:
- `ane_lock_input()` → `ane_input_ptr()` → memcpy → `ane_unlock_input()`
- `ane_lock_output()` → `ane_output_ptr()` → memcpy → `ane_unlock_output()`

**Impact:** +4% (416→433 tok/s). Weniger als erwartet weil `ane_write/ane_read`
intern auch memcpy auf IOSurface macht — Unterschied ist nur der eliminierte
interne Buffer-Alloc in libane.

### Aktuelles Bottleneck Breakdown (seq=81, 22 Layers) [MEASURED]

| Komponente | Zeit | Pro Layer | Anteil |
|-----------|------|-----------|--------|
| GPU FFN (MPS) | 100 ms | 4.6 ms | **53%** ← neuer Hauptflaschenhals |
| Unaccounted (I/O) | 65 ms | 3.0 ms | 34% |
| ANE Attention | 22 ms | 1.0 ms | 12% |
| Rest (embed, KV, residual) | 1 ms | — | <1% |

GPU FFN ist jetzt der Hauptflaschenhals — nicht mehr ANE I/O.

## Naechste Schritte (Potenzial)

- **GPU FFN Overhead** — 53% der Zeit. MPS Objekt-Erstellung + memcpy pro Layer kostet ~3ms/L
- **Groessere Modelle** (7B) mit mmap + Fiber-Loading testen
- **ANE Decode** fuer laengere Kontexte (>256 tokens)
- **Cached MPS Objects** — MPSMatrix/MPSMatrixMultiplication einmal erstellen statt pro Layer
