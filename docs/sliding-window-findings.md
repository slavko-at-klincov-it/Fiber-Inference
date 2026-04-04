# Sliding Window Attention: Kein Speed-Vorteil [MEASURED]

Date: 2026-04-04

## Test

Sliding Window Mask: attend only last W positions (mask-basiert).
Getestet bei seq=256, 384, 512 mit window=128, 192, 256.

## Ergebnis

| seq | Full Causal | Window=128 | Window=192 |
|-----|------------|------------|------------|
| 128 | 0.23 ms | — | — |
| 256 | 0.41 ms | 0.41 ms (0%) | — |
| 384 | 1.07 ms | 1.07 ms (0%) | 1.07 ms (0%) |
| 512 | 2.28 ms | 2.38 ms (-4%) | — |

**Kein Speed-Unterschied.** Sogar leicht langsamer bei seq=512.

## Warum

Die Sliding Window Mask aendert nur Softmax-Gewichte (-inf fuer out-of-window),
aber der **Q @ K^T Matmul berechnet trotzdem alle seq×seq Elemente**.
Die SRAM-Belastung kommt vom Matmul Intermediate `[heads, seq, seq]`,
nicht vom Softmax.

Fuer echte Sliding Window Speedup muesste die Matmul-Groesse selbst reduziert
werden — nur [heads, seq, window] statt [heads, seq, seq]. Das braucht einen
fundamentalen MIL-Umbau (kein Drop-in Mask Replacement).

## Fazit

Mask-basierte Sliding Window ist **nutzlos fuer Speed** auf ANE.
Die Mask behaelt die Korrektheit (nur Window-Positionen bekommen Gewicht),
aber die Performance ist identisch zu Full Causal.

Fuer seq > 256 auf ANE braucht es entweder:
1. **Chunked Attention:** seq in 128er Bloecke aufteilen, pro Block ein ANE Dispatch
2. **Matmul-Level Window:** MIL umschreiben fuer [seq, window] statt [seq, seq]
3. **Akzeptieren:** seq > 256 ist SRAM-limitiert, 2-5x langsamer pro Layer
