# Fiber-Inference

Systematische Evaluation aller 5 Compute Units des Apple M4 für LLM Inference.

48 Commits, über 200 Messungen, 27 Dokumentationsdateien.

## Ergebnis

Die Apple Neural Engine (ANE) kann korrekte Transformer-Layer ausführen und
erreicht bis zu 21.490 tok/s Prefill. Aber Apple's eigenes MLX-Framework
ist 2,2× schneller — vertikale Hardware-Software-Integration schlägt
Reverse Engineering.

**Paper:** `docs/paper.md` (auch als PDF in Documents/)

## Kernfindungen

1. **ANE Prefill funktioniert** — 32/32 Token-Match, kohärenter Text generiert
2. **AMX ist 1,8× schneller als GPU** für FFN bei dim≤1024 — kein Framework nutzt das
3. **Parallele Units sind langsamer** — Memory-Bandwidth-Konkurrenz (-40%)
4. **MLX gewinnt** — 2,2× schneller als unsere beste Konfiguration
5. **Bandwidth > FLOPS** — die zentrale Erkenntnis für Apple Silicon LLMs

## Build

```bash
make
./fiber-inference --model models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --prompt "Hello" --tokens 32

# ANE Proof (benötigt stories110M.bin):
./fiber-inference --arch proof --prompt "Once upon a time"

# Hardware Sweep:
cd bench && make && ./bench-sweep

# Parallel Unit Test:
./parallel-test
```

## Verwandte Repositories

| Repo | Was | Commits |
|------|-----|---------|
| **Fiber-Inference** (dieses Repo) | Inference Engine + Paper + Benchmarks | 48 |
| **M4_RE** (`/Code/M4_RE`) | Hardware Reverse Engineering (33 Experimente) | 48 |
| **ANE-Training** (`/Code/ANE-Training`) | ANE Training + libane C API | 99 |

Archivierte Projekte (abgebrochen): `archive/Fiber-LLM`, `archive/gdc-lm`, `archive/fiber-train`

## Dokumentation

Alle Findings in `docs/`:
- `paper.md` — Vollständiges Research Paper
- `hardware-sweep.md` — ANE/GPU/AMX GFLOPS pro Dimension
- `proof.md` + `proof-sweep.md` — Korrektheitsbeweis
- `parallel-units-test.md` — Multi-Unit Test (langsamer!)
- `ane-vs-ollama-gemessen.md` — Fairer Vergleich
- `finale-erkenntnisse.md` — Was jede Unit kann und wann
- Und 20 weitere...

## Hardware

- Apple M4 Mac Mini, 16 GB Unified Memory
- macOS, Xcode Command Line Tools
- libane (symlinked aus ANE-Training)
