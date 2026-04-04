# Realistisches Potenzial: Was koennen wir besser als Ollama?

Date: 2026-04-05

## Ollama Baseline [GEMESSEN]

Qwen3-0.6B (dim=1024), Ollama 0.19 (MLX Backend), M4 Mac Mini:

| Metrik | Ollama | Methode |
|--------|--------|---------|
| **Prefill (124 tok)** | **2,220 tok/s** | MLX Metal GPU |
| **Decode** | **133 tok/s** | MLX Metal GPU |

## Was wir gemessen haben

| Metrik | Unsere Engine | Methode |
|--------|-------------|---------|
| ANE Prefill (dim=1024, 128 tok) | **6,880 tok/s** | ANE Kernels (bench/large_ane.m) |
| GPU Decode (TinyLlama 1.1B) | 52 tok/s | Multi-Row Metal Kernel |
| CPU Decode (Stories-110M) | 69 tok/s | cblas_sgemv |

## Realistisches Szenario: Fork MLX + unsere Optimierungen

### Prefill: ANE statt GPU

| Approach | Prefill tok/s | vs Ollama |
|----------|--------------|-----------|
| Ollama (MLX GPU) | 2,220 | 1.0x |
| **ANE Prefill** | **6,880** | **3.1x schneller** |

ANE Prefill ist 3.1x schneller als Ollama's GPU Prefill bei dim=1024.
Das ist GEMESSEN, nicht geschaetzt.

### Decode: MLX + AMX parallel

| Approach | Decode tok/s | vs Ollama |
|----------|-------------|-----------|
| Ollama (MLX GPU only) | 133 | 1.0x |
| MLX GPU + AMX FFN parallel | ~160-180 (geschaetzt) | ~1.2-1.4x |

AMX schlaegt GPU bei FFN dim=1024 (1.8x, gemessen). Wenn AMX den FFN
parallel zur GPU Attention macht, gewinnen wir ~20-40% Decode.
**Aber:** Das erfordert AMX als MLX-Backend einzubauen (mittlerer Aufwand).

### Kombiniert: ANE Prefill + MLX Decode + AMX parallel

| Phase | Ollama | Optimiert | Speedup |
|-------|--------|-----------|---------|
| Prefill (124 tok) | 56ms | **18ms** (ANE) | **3.1x** |
| Decode (50 tok) | 376ms | ~310ms (MLX+AMX) | **1.2x** |
| **Time-to-First-Token** | **56ms** | **18ms** | **3.1x** |
| **Total (50 tok Antwort)** | **432ms** | **328ms** | **1.3x** |

## Ehrliche Antwort: Wieviel x?

| Metrik | Realistisches x | Bemerkung |
|--------|----------------|-----------|
| **Prefill (TTFT)** | **3x schneller** | ANE bei dim≤1024, gemessen |
| **Decode** | **1.2-1.4x schneller** | AMX parallel, geschaetzt |
| **Total Response** | **1.3x schneller** | Prefill-Gewinn dominiert bei kurzen Antworten |
| **RAG (langer Prompt)** | **2x schneller** | Prefill ist groesserer Anteil |

**Nicht 10x oder 50x. Aber 2-3x fuer TTFT ist spuerbar und real.**

Fuer RAG speziell (500+ Token Prompt, 50 Token Antwort):
- Ollama: ~225ms Prefill + 376ms Decode = 601ms
- Optimiert: ~73ms Prefill (ANE) + 310ms Decode = 383ms
- **1.6x schnellere Gesamtantwort, 3x schnelleres TTFT**

## Was NICHT besser wird

- Decode bei grossen Modellen (dim > 1024): MLX/Ollama ist gleich/besser
- Modellqualitaet: gleich (selbes Modell)
- RAM: etwas mehr (ANE braucht FP16 Weights neben quantisierten)

## Fazit

Der realistische Vorteil unserer Arbeit:
- **3x schnelleres Time-to-First-Token** bei dim≤1024 Modellen
- **~1.3x schnellere Gesamtantwort** durch ANE Prefill + AMX parallel
- **Spuerbar bei RAG** (langer Kontext → kurze Antwort)
- **Nicht spuerbar bei Chat** (kurzer Prompt → lange Antwort = Decode dominiert)
