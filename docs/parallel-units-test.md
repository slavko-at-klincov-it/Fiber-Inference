# Parallel Units Test: GPU+AMX+ANE zusammen — LANGSAMER [GEMESSEN]

Date: 2026-04-05

## Die Frage

M4_RE Exp 06 zeigte 103% Effizienz wenn GPU+AMX+ANE gleichzeitig laufen.
MLX nutzt nur GPU (~7 TFLOPS). Liegen 11+ TFLOPS ungenutzt auf dem Tisch?
Kann man alle 3 Units auf dasselbe Modell gleichzeitig ansetzen?

## Ergebnis: NEIN. Parallel ist langsamer.

### dim=768, 12 Layers, seq=128

| Konfiguration | Wall Time | vs GPU allein |
|--------------|-----------|---------------|
| GPU allein | 8.6 ms | 1.0x |
| GPU + AMX parallel | 14.7 ms | **0.6x (langsamer!)** |
| GPU + ANE parallel | 108.5 ms | **0.08x (13x langsamer!)** |
| Alle 3 parallel | 109.9 ms | **0.08x** |

### dim=1024, 28 Layers, seq=128

| Konfiguration | Wall Time | vs GPU allein |
|--------------|-----------|---------------|
| GPU allein | 28.3 ms | 1.0x |
| GPU + AMX parallel | 40.4 ms | **0.7x (langsamer!)** |
| GPU + ANE parallel | 122.2 ms | **0.2x** |
| Alle 3 parallel | 128.1 ms | **0.2x** |

## Warum parallel LANGSAMER ist

### 1. ANE Compilation Overhead (~100ms)

ANE `ane_compile()` + erster `ane_eval()` kostet ~100ms beim ersten Aufruf in
einem neuen Thread. Das dominiert die gesamte Messung. Selbst wenn ANE danach
schnell waere — der Startup-Overhead ist 10-50x laenger als die eigentliche GPU-Arbeit.

### 2. Memory Bandwidth Konkurrenz

GPU und AMX teilen sich die 120 GB/s Unified Memory Bandwidth.
Wenn beide gleichzeitig grosse Matrizen lesen:
- GPU allein: ~565 GFLOPS (bei dim=768)
- GPU + AMX gleichzeitig: GPU faellt auf ~435 GFLOPS, AMX auf ~1100 GFLOPS
- Negative Overlap (-54%): sie stoeren sich, statt sich zu ergaenzen

### 3. M4_RE Exp 06 war ein ANDERER Test

Exp 06 zeigte 103% Effizienz bei **UNABHAENGIGEN** Workloads:
- GPU rechnete eine EIGENE Matmul
- AMX rechnete eine EIGENE Matmul
- ANE rechnete eine EIGENE Matmul
- Keine geteilten Daten, kein Bandwidth-Wettbewerb

Fuer LLM Inference lesen alle Units die GLEICHEN Weights → Bandwidth-Engpass.

## Fazit

**Parallel Multi-Unit auf demselben Modell ist keine Option.**

Die 103% Effizienz aus M4_RE gilt nur fuer unabhaengige Workloads.
Fuer LLM Inference wo alle Units dieselben Weights lesen: Bandwidth-limitiert.

MLX's Entscheidung, NUR die GPU zu nutzen, ist korrekt:
- Ein Accelerator = keine Bandwidth-Konkurrenz
- GPU allein: ~565-798 GFLOPS
- GPU+AMX zusammen: ~435-600 GFLOPS pro Unit (stoeren sich)

## Einziges Szenario wo Parallel helfen KOENNTE

Wenn GPU und AMX/ANE VERSCHIEDENE Modelle oder VERSCHIEDENE Requests rechnen:
- GPU: Modell A fuer User 1
- AMX: Modell B fuer User 2 (nur bei dim≤1024)
- ANE: Prefill fuer User 3 (nur bei dim≤1024)

Aber das ist ein Server-Szenario, nicht Single-User lokale Inference.
