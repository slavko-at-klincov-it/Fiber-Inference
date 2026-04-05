# ANE vs Ollama (MLX): Gemessener Vergleich [ALLE ZAHLEN GEMESSEN]

Date: 2026-04-05

## ACHTUNG: Nicht 1:1 vergleichbar

- **Ollama:** Echtes Modell (trainierte Weights, quantisiert Q4/Q8), langer Prompt (~120 tok)
- **ANE:** Synthetische Random Weights, seq=128, nur Prefill (kein Decode)
- **Ollama bei warmem Cache** (Modell schon geladen) vs ANE inkl. I/O

Trotzdem die besten Zahlen die wir haben:

## Prefill-Vergleich [GEMESSEN]

### Qwen3-0.6B (dim=1024, 28 Layers)

| Engine | Prefill tok/s | Bedingungen |
|--------|-------------|-------------|
| Ollama (kalt, 1. Run) | 549 | Modell laden + Prefill |
| **Ollama (warm, Median)** | **19,700** | MLX GPU, langer Prompt ~120 tok, cached |
| Ollama (warm, kurzer Prompt) | 2,400 | MLX GPU, ~20 tok |
| **ANE (unsere Engine)** | **7,150** | Synthetische Weights, seq=128 |

**ERGEBNIS: Ollama (MLX) ist bei langem Prompt 2.8x SCHNELLER als unsere ANE.**

Ollama's MLX batched Prefill bei warmem Cache erreicht 19,700 tok/s bei langem Prompt.
Unsere ANE schafft 7,150 tok/s. Der MLX Vorteil kommt von:
- Batched GPU Matmul (alle Tokens gleichzeitig, hochoptimiert)
- Quantisierte Weights (Q4/Q8 = weniger Memory Bandwidth)
- Metal4 Tensor API (Hardware-Matmul)
- Warmer Cache (Modell schon im GPU Memory)

### Qwen3-4B (dim=2560, 36 Layers)

| Engine | Prefill tok/s | Bedingungen |
|--------|-------------|-------------|
| Ollama (kalt) | 372 | Modell laden |
| **Ollama (warm)** | **5,095** | MLX GPU, langer Prompt |
| **ANE (unsere Engine)** | **701** | Synthetische Weights, seq=128 |

**ERGEBNIS: Ollama ist bei 4B 7.3x schneller als unsere ANE.**

Bei dim=2560 verliert ANE deutlich wegen SRAM-Spill.

### Qwen3-8B (dim=4096, 36 Layers) — nur ANE

| Engine | Prefill tok/s |
|--------|-------------|
| **ANE** | **149** |
| Ollama (geschaetzt) | ~2,000-3,000 |

## Was das bedeutet

### Meine vorherige Behauptung war FALSCH

Ich habe gesagt "ANE ist 3.1x schneller als Ollama bei Qwen3-0.6B" — basierend auf:
- Ollama erster Run (549 tok/s, kalt) vs ANE (6,880 tok/s)

Das war ein unfairer Vergleich. Bei warmem Cache ist Ollama (MLX) **schneller** als ANE.

### Die korrigierte Wahrheit

| Modell | ANE vs Ollama (warm) |
|--------|---------------------|
| Qwen3-0.6B (dim=1024) | **ANE 2.8x LANGSAMER** |
| Qwen3-4B (dim=2560) | **ANE 7.3x LANGSAMER** |

**ANE Prefill ist NICHT schneller als Ollama (MLX).** MLX's optimierte Metal GPU
Kernels mit quantisierten Weights und Metal4 Tensor API schlagen ANE.

### Warum ANE verliert

1. **MLX nutzt quantisierte Weights** (Q4/Q8) → weniger Memory lesen → schneller
2. **MLX nutzt Metal4 Tensor API** (mpp::matmul2d) → Hardware-Matmul
3. **MLX batched Prefill** ist hochoptimiert fuer grosse Batches
4. **ANE arbeitet mit FP16** (doppelt so viel Memory wie Q8, 4x wie Q4)
5. **ANE hat SRAM-Limit** bei grossen Modellen

### Was bleibt von unserem Projekt?

Die Hardware-Forschung bleibt wertvoll:
- Wir wissen GENAU was jede Unit kann
- Wir haben ANE Limits systematisch dokumentiert
- Wir haben bewiesen dass ANE korrekte Transformer-Layer ausfuehrt
- AMX-Entdeckung (1.8x schneller als GPU bei FFN dim≤1024)

Aber fuer Produktions-Inference: **MLX (Ollama) ist schneller. Punkt.**
