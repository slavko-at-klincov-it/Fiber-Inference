# ANE Decode Findings: Re-Prefill Ansatz

Date: 2026-04-04

## Ergebnis [MEASURED]

Stories-110M, "Once upon a time", 32 Decode-Tokens.

| Metrik | CPU/AMX | ANE (Re-Prefill) |
|--------|---------|-------------------|
| Prefill | 64.6 tok/s | **473.4 tok/s (7.3x)** |
| Decode | **68.6 tok/s** | 64.7 tok/s (0.9x — langsamer) |
| Token Match | — | **32/32 (100%)** |
| Output | "meck...syst..." | **identisch** |

## Warum ANE Decode (Re-Prefill) langsamer ist

Re-Prefill wiederholt bei jedem neuen Token den gesamten Kontext:
- Token 1: verarbeitet 5 Tokens (Prompt + 1 generiert)
- Token 2: verarbeitet 6 Tokens
- ...
- Token 32: verarbeitet 36 Tokens
- Total: ~620 Token-Slots verarbeitet (O(n²))

CPU Decode: 32 einzelne Token-Forwards (O(n), mit KV Cache).

**Re-Prefill ist strukturell O(n²) vs O(n) fuer echtes Decode.**
Bei kurzen Kontexten (≤32 Tokens) dominiert der Overhead.

## Was bewiesen ist

1. **32/32 Token Match** — ANE Forward produziert exakt gleiche Tokens wie CPU
2. **7.3x Prefill Speedup** — konsistent mit frueheren Messungen
3. **ANE kann Decode-Schleifen ausfuehren** — keine Stabilitaetsprobleme ueber 32 Iterationen
4. **Kein FP16 Precision-Drift** — nach 32 Re-Prefills mit akkumulierten Residuals
   sind alle Tokens noch identisch mit der FP32 CPU Baseline

## Naechster Schritt: Echtes inkrementelles Decode

Fuer Speed-Verbesserung braucht es:
- KV Cache als ANE **Input** (nicht Re-Prefill)
- 3-Input MIL Kernel: x[dim,1] + k_cache[kv_dim,ctx] + v_cache[kv_dim,ctx]
- `ane_compile()` mit n_inputs=3 (unterstuetzt laut libane/ane.h)
- Erwartung: ~200-400 tok/s Decode (5-6x ueber CPU Baseline)

## Fixes in diesem Commit

- **GQA Bug gefixt:** n_heads % n_kv_heads != 0 wird jetzt abgelehnt
- **Qwen3-4B kompiliert:** 760 tok/s mit korrekten heads=32/kv=8
- **head_dim=128 getestet:** 8.6% schneller bei dim=768 (25,294 vs 23,294 tok/s)
