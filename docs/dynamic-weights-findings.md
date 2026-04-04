# Dynamic Weights auf ANE: Funktioniert nur bis IC=128 [MEASURED]

Date: 2026-04-04

## Test

`ane_mil_linear_dynamic()` — compile-once Kernel mit Weights als Input-Channels.

| IC | OC | SEQ | Input Size | Ergebnis |
|----|-----|-----|-----------|----------|
| 64 | 64 | 32 | 12 KB | **OK** |
| 128 | 128 | 32 | 40 KB | **OK** |
| 256 | 256 | 32 | 144 KB | **FAIL** |
| 256 | 768 | 32 | 400 KB | **FAIL** |
| 768 | 2048 | 32 | 3120 KB | **FAIL** |

## Analyse

Dynamic Weights packen die Weight-Matrix als extra Spatial-Positionen im Input-Tensor.
Bei IC=256, OC=256: Input wird `[1, 256, 1, 32+256]` = 256 × 288 × 2 = 144 KB.
Der ANE Compiler kann die Spatial-Dimension (288) nicht effizient in SRAM mappen
wenn die Channel-Dimension (256) gleichzeitig gross ist.

Bei IC=128: 128 × (32+128) = 128 × 160 × 2 = 40 KB — passt noch.
Bei IC=256: 256 × 288 = 144 KB — uebersteigt ein internes Compiler-Limit.

## Konsequenz

Dynamic Weights sind **nicht nutzbar** fuer unsere Architektur (dim=768).
Das Compile-Budget bleibt bei 24 Kernels pro Modell (1 pro Layer × 2 Ops).

Fuer Decode mit KV Cache als Input: **muss ein anderer Ansatz** gefunden werden.
Moeglich: Multi-Input MIL (ungetestet), oder KV Cache in Weight-Blobs verpacken
(recompile pro Position — unpraktisch).

## Fazit

Alle drei getesteten Optimierungen treffen auf fundamentale ANE Limits:
1. **INT8:** Compiler lehnt INT8 mit FP16 MIL ab
2. **Sliding Window:** Mask aendert nicht die Compute-Groesse
3. **Dynamic Weights:** Funktioniert nur bis IC=128

Der ANE ist extrem schnell fuer **baked-weight Prefill** (10K+ tok/s) aber
hat harte Constraints die fortgeschrittene Features wie Decode, INT8 und
Dynamic Weights bei Produktions-Dimensionen verhindern.
