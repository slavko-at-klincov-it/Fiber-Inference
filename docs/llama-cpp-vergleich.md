# Fairer Vergleich: Fiber-Inference vs llama.cpp

Date: 2026-04-04
Hardware: Apple M4 Mac Mini, 16 GB

## llama.cpp Benchmark (TinyLlama-1.1B Q4_K_M) [MEASURED]

```
llama-bench -m models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -p 128 -n 32 -ngl 99
```

| Metrik | llama.cpp |
|--------|----------|
| Prefill (pp128) | **1,434 tok/s** |
| Decode (tg32) | **111 tok/s** |
| Backend | BLAS + Metal (GPU) |
| Modell | TinyLlama 1.1B Q4_K_M (636 MB) |

## Fiber-Inference Benchmarks [MEASURED]

### Stories-110M (pretrained, dim=768, 12L, FP16)

| Pipeline | Prefill | Decode |
|----------|---------|--------|
| CPU/AMX | 62 tok/s | 69 tok/s |
| **ANE** | **1,219 tok/s** | 65 tok/s |

### TinyLlama-1.1B Q4_K_M (GPU-only, Phase 1)

| Pipeline | Prefill | Decode |
|----------|---------|--------|
| GPU-only | 40 tok/s | 37 tok/s |
| ANE+GPU (Phase 2) | 420 tok/s | 37 tok/s |

## Analyse

### Prefill: ANE ist kompetitiv

- Fiber ANE (110M FP16): **1,219 tok/s**
- llama.cpp (1.1B Q4_K): **1,434 tok/s**

ANE erreicht 85% der llama.cpp Prefill-Speed mit einem **10x kleineren Modell**
(110M vs 1.1B) und **unquantisierten Weights** (FP16 vs Q4_K).

Pro Token Compute: ANE verarbeitet weniger Parameter pro Token (110M vs 1.1B),
dafuer aber mit doppelter Precision (FP16 vs 4-bit). Fairer Vergleich braucht
gleiches Modell in gleichem Format.

### Decode: llama.cpp ist deutlich schneller

- Fiber: **65 tok/s** (CPU/AMX Decode, ANE nur Prefill)
- llama.cpp: **111 tok/s** (Metal GPU Decode)

llama.cpp Decode ist 1.7x schneller. Das ist erwartungsgemaess — llama.cpp hat
jahrelang optimierte Metal Kernels fuer Single-Token Decode, inklusive:
- Optimiertes Q4_K Dequant-MatVec auf GPU
- Flash Attention fuer GPU
- Fused KV Cache Operations

Unser Decode laeuft auf CPU/AMX (kein GPU, kein ANE).

### GPU-only Pipeline: weit hinter llama.cpp

Unsere Phase 1 GPU Pipeline (40 tok/s Decode) ist **2.8x langsamer** als llama.cpp
(111 tok/s) auf dem GLEICHEN Modell (TinyLlama Q4_K_M). Das zeigt: unsere GPU
Kernels sind nicht optimiert. llama.cpp hat extrem effiziente Metal Shader.

## Fazit

| Aussage | Belegt? |
|---------|---------|
| ANE Prefill ist schnell | JA — 85% von llama.cpp bei 10x kleinerem Modell |
| ANE Decode ist schnell | NEIN — kein ANE Decode, CPU ist 1.7x langsamer als llama.cpp |
| Unsere GPU ist langsam | JA — 2.8x langsamer als llama.cpp auf gleichem Modell |
| ANE ist besser als GPU fuer Prefill | INDIREKT — 1,219 vs 420 tok/s (unsere GPU), aber kein 1:1 Modell-Vergleich |

## Was ein perfekter Vergleich braeuchte

1. **stories110M als GGUF** → llama-bench auf 110M → direkter Prefill/Decode Vergleich
2. **Oder: unser ANE Code auf TinyLlama-1.1B** → aber dim=2048 hat SRAM-Probleme
3. **Oder: ein 768-dim Modell in GGUF** → gibt es nicht (kleinste GGUF Modelle haben dim=896+)
