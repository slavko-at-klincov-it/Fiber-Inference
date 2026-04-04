# Large Model Sweep: 15M bis 4B Parameter

Date: 2026-04-04

## Ergebnis: 12/12 Token-Match, 29x-126x Speedup [MEASURED]

128 tokens, synthetische Weights, GQA (kv=8) fuer groesse Modelle.

| Modell | dim | heads | kv | ffn | Layers | CPU ms | ANE ms | Speedup | Match |
|--------|-----|-------|-----|------|--------|--------|--------|---------|-------|
| ~15M | 256 | 4 | 4 | 684 | 12 | 89 | 3.1 | 29x | YES |
| ~30M | 384 | 6 | 6 | 1024 | 12 | 181 | 4.1 | 44x | YES |
| ~50M | 512 | 8 | 8 | 1365 | 12 | 307 | 5.0 | 62x | YES |
| ~110M | 768 | 12 | 12 | 2048 | 12 | 661 | 8.4 | 79x | YES |
| ~500M | 1024 | 16 | 8 | 2730 | 16 | 1,423 | 15.4 | 92x | YES |
| **~1B** | **1024** | **16** | **8** | **4096** | **24** | **2,893** | **27.5** | **105x** | **YES** |
| ~1B-wide | 1536 | 24 | 8 | 4096 | 16 | 3,014 | 27.6 | 109x | YES |
| **~2B** | **2048** | **32** | **8** | **5461** | **22** | **7,467** | **59.5** | **126x** | **YES** |
| ~3B | 2048 | 32 | 8 | 5461 | 32 | 10,752 | 93.8 | 115x | YES |
| ~4B | 2560 | 40 | 8 | 6912 | 28 | 16,141 | 174.3 | 93x | YES |
| 768d-32L | 768 | 12 | 12 | 2048 | 32 | 1,788 | 20.6 | 87x | YES |
| 768d-48L | 768 | 12 | 12 | 2048 | 48 | 2,679 | 30.5 | 88x | YES |

## Erkenntnisse

### Speedup skaliert mit Modellgroesse — bis 2B
- ~15M: 29x
- ~110M: 79x
- ~1B: 105x
- **~2B: 126x (Peak)**
- ~4B: 93x (leichter Rueckgang)

ANE wird effizienter bei groesseren Modellen weil die Compute-Intensitaet steigt
und der Dispatch-Overhead relativ kleiner wird.

### Kipppunkt bei dim=2560
Bei ~4B (dim=2560) faellt der Speedup auf 93x zurueck. Ursachen:
- Weight-Tensoren [2560, 2560] = 13MB FP16 pro Matrix — SRAM Spill
- ANE eval: 174ms / 28 Layers = 6.2ms/Layer (vs 2.7ms/Layer bei dim=2048)

### ANE per-Layer Zeiten

| dim | ANE ms/Layer | SRAM Spill? |
|-----|-------------|-------------|
| 256 | 0.26 | ja (weights) |
| 512 | 0.42 | ja |
| 768 | 0.70 | ja |
| 1024 | 0.96 | ja |
| 1536 | 1.73 | ja |
| 2048 | 2.70 | ja |
| 2560 | 6.23 | **stark** |

Bis dim=2048 skaliert ANE sauber (~linear). Bei dim=2560 springt es auf 6.2ms — 
der SRAM-Druck wird zu gross.

### CPU Baseline per-Layer Zeiten

| dim | CPU ms/Layer |
|-----|-------------|
| 256 | 7.4 |
| 768 | 55.1 |
| 1024 | 88.9 |
| 2048 | 339.4 |
| 2560 | 576.5 |

CPU skaliert quadratisch (dominiert von dim×dim Matmuls).

### Modell-Groesse vs Latenz

| Modell | ANE Prefill (128 tok) | Throughput |
|--------|----------------------|-----------|
| ~110M | 8.4 ms | 15,238 tok/s |
| ~500M | 15.4 ms | 8,312 tok/s |
| ~1B | 27.5 ms | 4,655 tok/s |
| ~2B | 59.5 ms | 2,151 tok/s |
| ~3B | 93.8 ms | 1,364 tok/s |
| ~4B | 174.3 ms | 734 tok/s |

Selbst bei 4B Parametern: **734 tok/s Prefill auf einem Mac Mini!**

## Empfehlung

| Ziel | Optimale Config |
|------|----------------|
| Maximale Speed | dim=768, 12-24L (~110M-220M) → 10,000+ tok/s |
| Beste Qualitaet/Speed | dim=1024, 24L (~1B) → 4,600 tok/s, **105x Speedup** |
| Maximale Qualitaet | dim=2048, 22-32L (~2-3B) → 1,300-2,100 tok/s, **115-126x** |
| Nicht empfohlen | dim > 2560 → SRAM-Kipppunkt |

## Reproduzierbar

```bash
./fiber-inference --arch sweep
```
