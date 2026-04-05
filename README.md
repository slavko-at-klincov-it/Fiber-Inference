# Fiber-Inference

**Systematische Evaluation aller 5 Compute Units des Apple M4 für LLM Inference**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-macOS%20(Apple%20Silicon)-lightgrey.svg)]()
[![Language](https://img.shields.io/badge/Language-Objective--C%20%2F%20C%20%2F%20Metal-orange.svg)]()

> 48 Commits, über 200 Messungen, 27 Dokumentationsdateien.
> Vollständiges Research Paper: [`docs/paper.md`](docs/paper.md)

---

## Ergebnis

Die Apple Neural Engine (ANE) kann korrekte Transformer-Layer ausführen und erreicht bis zu **21.490 tok/s Prefill**. Aber Apples eigenes MLX-Framework ist **2,2× schneller** — vertikale Hardware-Software-Integration schlägt Reverse Engineering.

## Kernfindungen

| # | Finding | Detail |
|---|---------|--------|
| 1 | **ANE Prefill funktioniert** | 32/32 Token-Match, kohärenter Text generiert |
| 2 | **AMX ist 1,8× schneller als GPU** | Für FFN bei dim≤1024 — kein Framework nutzt das |
| 3 | **Parallele Units sind langsamer** | Memory-Bandwidth-Konkurrenz (-40%) |
| 4 | **MLX gewinnt** | 2,2× schneller als unsere beste Konfiguration |
| 5 | **Bandwidth > FLOPS** | Die zentrale Erkenntnis für Apple Silicon LLMs |

## Gemessene Hardware — Die 5 Compute Units

| Unit | Gemessener Peak | Stärke | Limitierung |
|------|----------------|--------|-------------|
| GPU (10-core) | ~3,5 TFLOPS FP16 (MPS) | Flexibel, Decode, Quantisierung | Langsamer als ANE bei dim≤1024 |
| ANE (16-core) | 13,86 TFLOPS FP16 | Schnellster Prefill bei dim≤1024 | 32MB SRAM, kein Decode, kein INT8 |
| AMX/SME | ~1,6 TFLOPS FP32 | 1,8× schneller als GPU FFN bei dim≤1024 | Nur FP32, kein FP16 Vorteil |
| CPU P-cores (4×) | ~0,2 TFLOPS NEON | Orchestrierung | Langsam für Matmul |
| CPU E-cores (6×) | ~0,36 TFLOPS NEON | Background | Langsam |

> **Hinweis:** 19 TFLOPS ist Apples Marketing-Zahl. Gemessen: 13,86 TFLOPS.

## Voraussetzungen

- **Hardware:** Apple Silicon Mac (M4 getestet)
- **OS:** macOS mit Xcode Command Line Tools
- **Abhängigkeit:** [ANE-Training](https://github.com/slavko-at-klincov-it/ANE-Training) — für `libane` (ANE-Zugriff)

## Build

```bash
# ANE-Training Repo clonen (Abhängigkeit für libane)
git clone https://github.com/slavko-at-klincov-it/ANE-Training.git ../ANE-Training
cd ../ANE-Training/libane && make && cd -

# Fiber-Inference bauen
make

# Oder mit custom ANE_DIR:
ANE_DIR=/path/to/ANE-Training make
```

## Benutzung

```bash
# GGUF-Modell Inference
./fiber-inference --model models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --prompt "Hello" --tokens 32

# ANE Proof (benötigt stories110M.bin in models/)
./fiber-inference --arch proof --prompt "Once upon a time"

# Hardware Benchmark Sweep
cd bench && make && ./bench-sweep
```

## Projektstruktur

```
├── src/                    # Inference Engine (Objective-C/C)
│   ├── main.m              # CLI Entry Point
│   ├── model.m             # GGUF Model Loading
│   ├── gpu_ffn.m           # GPU FFN (Metal/MPS)
│   ├── ane_attn.m          # ANE Attention Layer
│   ├── amx_ffn.m           # AMX/SME FFN
│   ├── fiber_model.m       # Multi-Unit Orchestration
│   ├── fiber_proof.m       # ANE Correctness Proof
│   └── ...
├── metal/kernels.metal     # GPU Compute Kernels
├── bench/                  # Benchmark Suite
│   ├── sweep.m             # ANE/GPU/AMX GFLOPS Sweep
│   ├── parallel_units.m    # Multi-Unit Parallel Test
│   └── ...
├── docs/                   # Research Dokumentation
│   ├── paper.md            # Vollständiges Paper (27 Seiten)
│   ├── hardware-sweep.md   # GFLOPS pro Dimension
│   ├── proof.md            # ANE Korrektheitsbeweis
│   └── ...                 # 30+ weitere Findings
├── include/ane.h           # ANE Header
└── models/                 # Model Files (gitignored)
```

## Verwandte Repositories

| Repo | Beschreibung | Commits |
|------|-------------|---------|
| [**ANE-Training**](https://github.com/slavko-at-klincov-it/ANE-Training) | ANE Training + libane C API | 99 |
| **M4_RE** | Hardware Reverse Engineering (33 Experimente) | 48 |

## Dokumentation

Alle Findings in [`docs/`](docs/):

- [`paper.md`](docs/paper.md) — Vollständiges Research Paper
- [`hardware-sweep.md`](docs/hardware-sweep.md) — ANE/GPU/AMX GFLOPS pro Dimension
- [`proof.md`](docs/proof.md) — ANE Korrektheitsbeweis
- [`parallel-units-test.md`](docs/parallel-units-test.md) — Multi-Unit Test
- [`ane-vs-ollama-gemessen.md`](docs/ane-vs-ollama-gemessen.md) — Fairer ANE vs MLX Vergleich
- [`finale-erkenntnisse.md`](docs/finale-erkenntnisse.md) — Was jede Unit kann und wann

## Technische Details

- **Sprachen:** Objective-C, C, Metal Shading Language
- **Frameworks:** Metal, MetalPerformanceShaders, Accelerate, IOSurface
- **Aktivierungen:** FP16, interne Akkumulation FP32
- **Benchmarks:** Warmup, mehrere Runs, Median-Werte

## Lizenz

[MIT](LICENSE)
