# MLX Fork: Echte Chancen zur Verbesserung [RECHERCHIERT]

Date: 2026-04-05

## Was MLX NICHT hat (bestaetigt durch GitHub Issues + Quellen)

| Feature | MLX Status | Impact wenn eingebaut | Quelle |
|---------|-----------|----------------------|--------|
| **PagedAttention** | Nicht drin (angeboten, abgelehnt) | **+77-131% Throughput** bei mehreren Usern | Issue #2228, EricLBuehler bot Code an |
| **Q4_K_M Quantisierung** | Wird zu FP16 gecastet | **~2x weniger Memory**, schnellerer Decode | MLX GGUF README |
| **Speculative Decoding** | Minimal (nur batch=1, temp=0) | **2-3x Decode Speed** | PR #954, nicht produktionsreif |
| **Continuous Batching** | Nicht drin | **4.3x Scaling bei 16 Requests** | vllm-mlx Paper (EuroMLSys '26) |
| **Flash Attention (IO-aware)** | Nein, nur `steel_attention` (tiled) | Schnellerer Long-Context Prefill | Issue #129, #2955 |
| **Long-Context (>65K)** | GPU Watchdog Kill Bug | Ueberhaupt moeglich machen | Issue #3302, Fix existiert (575 Zeilen) |
| **ANE Prefill** | Nicht genutzt | 3.1x Prefill bei dim≤1024 (unsere Messung) | Unsere Arbeit |
| **AMX fuer FFN** | Nicht direkt genutzt | 1.8x FFN bei dim≤1024 (unsere Messung) | Unsere Arbeit |

## Die 3 groessten Chancen

### 1. PagedAttention einbauen → +77-131% Throughput

Ein Community-Mitglied (Eric Buehler, mistral.rs Entwickler) hat funktionierende
Metal PagedAttention Kernels angeboten. MLX Team hat es NICHT gemergt.

Gemessene Zahlen (von Buehler):
- Qwen3-30B: +77% Throughput mit PagedAttention
- Llama-3.2-3B: +131% Throughput

Das ist der groesste einzelne Gewinn. Der Code existiert bereits.

### 2. Q4_K_M Support → ~2x weniger Memory + schnellerer Decode

MLX unterstuetzt nur Q4_0, Q4_1, Q8_0 nativ. Alles andere (Q4_K_M, Q5_K, Q6_K)
wird zu FP16 gecastet = doppelter Memory-Verbrauch.

llama.cpp hat Q4_K_M seit Jahren. Ein 7B Modell:
- Q4_K_M: 3.8 GB
- MLX FP16 Fallback: 14 GB
- Das entscheidet ob ein Modell auf 16GB RAM passt oder nicht.

### 3. Speculative Decoding → 2-3x Decode Speed

Kleines Draft-Modell generiert N Tokens, grosses Modell verifiziert alle auf einmal.
Bei Akzeptanzrate 70-80% effektiv 2-3x schnellere Token-Generierung.

MLX hat nur eine Minimal-Implementation (batch=1, temp=0). Voll ausgebaut
mit rotierendem KV Cache und konfigurierbarer Temperatur waere das ein
signifikanter Decode-Boost.

## Realistischer Gesamtgewinn eines MLX-Forks

| Optimierung | Erwarteter Gewinn | Aufwand |
|-------------|------------------|---------|
| PagedAttention | +77-131% Throughput (multi-user) | Mittel (Code existiert) |
| Q4_K_M native | ~2x Memory-Reduktion | Hoch (Kernel-Arbeit) |
| Speculative Decoding | 2-3x Decode (single-user) | Mittel |
| Continuous Batching | 4.3x bei 16 Requests | Hoch (vllm-mlx hat Vorlage) |
| Long-Context Fix | 65K+ Tokens moeglich | Klein (Fix existiert, 575 Zeilen) |
| ANE Prefill (unsere Arbeit) | Unclear vs MLX GPU | Mittel |

**Single-User Szenario (1 Request):**
- Speculative Decoding: **2-3x Decode** (133 → 260-400 tok/s)
- Das waere spuerbar schneller als Standard-Ollama

**Multi-User Szenario (Server, 16 Requests):**
- PagedAttention + Continuous Batching: **4-8x Throughput**
- Aber: vllm-mlx macht das schon

## Die ehrliche Frage: Lohnt sich ein Fork?

**Fuer Single-User (lokaler Assistent):**
- Speculative Decoding → ja, spuerbarer Gewinn (2-3x Decode)
- Q4_K_M → ja, groessere Modelle auf 16GB
- Aufwand: Wochen bis Monate

**Fuer Multi-User (Server):**
- vllm-mlx existiert schon und hat PagedAttention + Continuous Batching
- Keinen Fork noetig, einfach vllm-mlx verwenden

**Fuer Forschung:**
- ANE Prefill als MLX Backend → moeglich, aber MLX GPU ist schon 2.2x schneller
- AMX als zusaetzliches Backend → moeglich, aber marginaler Gewinn

## Quellen

- MLX GitHub: github.com/ml-explore/mlx
- PagedAttention Angebot: Issue #2228
- Q4_K_M Limitation: MLX GGUF README
- Speculative Decoding: PR #954 in mlx-examples
- vllm-mlx Paper: arxiv.org/html/2601.19139v2
- Long-Context Bug: Issue #3302
- M5 Neural Accelerators: machinelearning.apple.com/research/exploring-llms-mlx-m5
