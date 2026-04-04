# Analyse: llama.cpp Metal vs unsere Kernels + Architektur-Landschaft

Date: 2026-04-05

## Warum llama.cpp 3x schneller decoded als wir

5 konkrete technische Unterschiede, alle messbar:

| Technik | llama.cpp | Fiber-Inference | Geschaetzter Impact |
|---------|----------|-----------------|-------------------|
| Multi-Row MatVec | Mehrere Zeilen/SIMD, Input einmal lesen | 1 Zeile/Threadgroup, Input re-read | **~2-3x Decode** |
| Flash Attention | Fused Q·K+Softmax+V, SIMD Matrix Ops | Separate Kernel, skalare Ops | ~1.5x Attention |
| Fused RMSNorm | Norm+Multiply+Add in 1 Dispatch | 3 separate Dispatches | ~10% |
| simdgroup_float8x8 | Hardware Matrix Multiply Units | threadgroup Reduction | ~1.5x Matmul |
| Metal4 Tensor API | mpp::tensor_ops::matmul2d | Nicht genutzt | ~1.3x Prefill |

## Was WIR haben das NIEMAND sonst hat

| Technik | Status | Wer sonst? |
|---------|--------|-----------|
| ANE Prefill fuer LLM | **Funktioniert, 1,219 tok/s** | ANEMLL (langsamer, CoreML-basiert) |
| ANE Korrektheitsbeweis | **32/32 Token Match** | Orion (nur 124M, Forschung) |
| Hardware-Sweep aller 5 Units | **Dokumentiert** | Niemand |
| ANE + GPU Hybrid Pipeline | **Bewiesen** | Niemand in Produktion |

## Architektur-Landschaft (April 2026) [RECHERCHIERT]

| Engine | Compute Units | ANE? | Decode 7B M4 base |
|--------|-------------|------|-------------------|
| Ollama 0.19 | MLX (GPU) | Nein | ~50 tok/s |
| llama.cpp | Metal GPU | Nein | ~24 tok/s |
| MLX | Metal GPU | Nein | ~25-35 tok/s |
| vLLM-MLX | MLX (GPU) | Nein | ~30 tok/s |
| ANEMLL | CoreML → ANE | JA | ~9 tok/s (langsam!) |
| Orion | Private ANE APIs | JA | Nur 124M |
| **Fiber** | **ANE + GPU** | **JA** | 37 tok/s (unoptimiert) |

**Niemand hat ANE + GPU Hybrid in Produktion.** ANEMLL ist rein ANE (langsam).
Wir sind die einzigen mit funktionierender ANE Prefill + GPU Decode Pipeline.

## Die Kombination die fehlt

llama.cpp hat die besten Metal GPU Kernels (110 Stk, jahrelang optimiert).
Wir haben den einzigen funktionierenden ANE Prefill.

**Kombination:** Unseren ANE Prefill VOR llama.cpp's GPU Decode schalten.
Nicht eigene GPU Kernels schreiben — llama.cpp's Kernels nutzen.

Oder: Die 5 fehlenden GPU-Techniken in unsere Kernels einbauen.
Multi-Row MatVec allein wuerde Decode von 37 auf ~80-100 tok/s bringen.
