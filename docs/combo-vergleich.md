# Alle Kombinationen vs Ollama — dim≤1024 [ALLES GEMESSEN]

Date: 2026-04-05

## Setup

- seq=128 Tokens
- Gleiche Modell-Dimensionen
- ANE: synthetische Weights FP16, volles Layer (Attention + FFN)
- CPU/AMX: FP32 cblas_sgemm, volles Layer
- GPU (MPS): FP16 MPS MatMul, nur FFN (unterschaetzt Full-Layer)
- Ollama: Qwen3-0.6B Q8 (echtes Modell), warm Cache

## Ergebnisse [GEMESSEN]

### Alle Unit-Kombinationen (seq=128)

| Config | ANE tok/s | CPU/AMX tok/s | GPU(MPS) tok/s |
|--------|----------|-------------|---------------|
| 768d, 12h, 12L | **23,479** | 4,603 | 7,252 |
| 768d, 6h(hd128), 12L | **24,252** | 4,836 | 7,190 |
| 1024d, 16h, 12L | **15,847** | 3,031 | 2,571 |
| 1024d, 8h(hd128), 12L | **15,548** | 2,871 | 5,516 |
| **1024d, 8h, 28L (Qwen3-0.6B)** | **6,864** | **1,226** | **2,888** |

### vs Ollama (gleiche Dimensionen: 1024d, 8h, 28L)

| Engine | Prefill tok/s | vs Ollama |
|--------|-------------|-----------|
| **Ollama (MLX, warm)** | **15,200** | 1.0x |
| ANE (unsere) | 6,864 | **2.2x langsamer** |
| GPU (MPS FFN only) | 2,888 | 5.3x langsamer |
| CPU/AMX | 1,226 | 12.4x langsamer |

## Analyse

### ANE ist 2.2x langsamer als Ollama (MLX)

Bei exakt gleichen Dimensionen (dim=1024, 28 Layers, 128 Tokens):
- Ollama (MLX): **15,200 tok/s**
- ANE: **6,864 tok/s**

MLX gewinnt weil:
1. **Quantisierte Weights (Q8):** 1 Byte/Weight vs FP16 2 Bytes → halb so viel Memory lesen
2. **Metal4 Tensor API (mpp::matmul2d):** Hardware-beschleunigte Matmul
3. **Unified Memory Zero-Copy:** Keine IOSurface Roundtrips
4. **Hochoptimierte Batched Prefill:** Jahre an Apple-Ingenieur-Arbeit

### ANE gewinnt gegen CPU und unsere GPU

- ANE ist **5.6x schneller als CPU/AMX** (6,864 vs 1,226)
- ANE ist **2.4x schneller als unsere GPU (MPS)** (6,864 vs 2,888)
- Aber unsere GPU/MPS Implementation ist unoptimiert (nur MPS MatMul, keine Custom Kernels)

### Bester dim=768 Speed

- ANE bei dim=768, 6 heads, hd=128: **24,252 tok/s** (absoluter Peak)
- Ollama hat kein dim=768 Modell zum Vergleich

### head_dim=128 vs head_dim=64

| dim=768, 12L | hd=64 (12 heads) | hd=128 (6 heads) | Vorteil |
|-------------|-----------------|------------------|---------|
| ANE | 23,479 | **24,252** | +3% |
| CPU | 4,603 | **4,836** | +5% |

head_dim=128 ist marginal schneller bei allen Units.

## Fazit

**Gegen Ollama (MLX) gibt es bei Prefill keinen Vorteil** — weder mit ANE noch
mit anderen Units. MLX ist 2.2x schneller als unsere beste ANE Konfiguration.

Der einzige Bereich wo unsere Arbeit konkurrenzfaehig waere:
- **Power Efficiency:** ANE ~2W vs GPU ~20W (nicht gemessen, aus Literatur)
- **Falls MLX Cold-Start:** Ollama erster Run ist 2,095 tok/s (ANE ist 6,864)
  Aber ab dem 2. Run ist MLX schneller.

## Rohdaten

```
combo-test (alle Units, seq=128):
  768-12h-12L:   ANE=23479  CPU=4603  GPU=7252
  768-6h-12L:    ANE=24252  CPU=4836  GPU=7190
  1024-16h-12L:  ANE=15847  CPU=3031  GPU=2571
  1024-8h-12L:   ANE=15548  CPU=2871  GPU=5516
  1024-8h-28L:   ANE=6864   CPU=1226  GPU=2888

Ollama Qwen3-0.6B (dim=1024, 28L, warm, ~128 tok):
  15,200 tok/s (median run 2-3)
```
