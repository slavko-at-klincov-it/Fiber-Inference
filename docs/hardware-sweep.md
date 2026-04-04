# Hardware Sweep: Optimale Dimensionen fuer Custom LLM Architektur

Date: 2026-04-04
Hardware: Apple M4 Mac Mini (10-core GPU, 16-core ANE, AMX/SME)

## Key Findings

### 1. ANE Attention — Sweet Spot: dim=768-1024, seq≤256

| dim | heads | hd | seq | ANE ms | GFLOPS | SRAM Spill |
|-----|-------|----|-----|--------|--------|-----------|
| 256 | 4 | 64 | 128 | 0.11 | 383 | YES |
| 384 | 6 | 64 | 128 | 0.11 | 794 | YES |
| 512 | 8 | 64 | 128 | 0.14 | 1044 | YES |
| **768** | **12** | **64** | **128** | **0.21** | **1584** | YES |
| **1024** | **16** | **64** | **128** | **0.27** | **2136** | YES |
| 2048 | 32 | 64 | 128 | 0.93 | 2391 | YES |

**Beobachtungen:**
- GFLOPS steigt mit dim (mehr Parallelitaet fuer ANE)
- dim=1024 bietet bestes Verhaeltnis: 2136 GFLOPS bei nur 0.27ms
- dim=2048 hat hoechste GFLOPS aber 3.4x laengere Latenz
- **SRAM Spill auf allen Configs** — die grossen Weight-Tensoren (Wq=[dim,dim]) uebersteigen 32MB

**head_dim Vergleich (dim=512):**

| head_dim | heads | ms | GFLOPS |
|----------|-------|------|--------|
| 32 | 16 | 0.18 | 835 |
| 64 | 8 | 0.14 | 1057 |
| 128 | 4 | 0.14 | 1104 |

head_dim=64 und 128 sind gleichschnell, head_dim=32 ist langsamer.
**Empfehlung: head_dim=64 oder 128.**

**Sequenzlaengen-Skalierung (dim=768):**

| seq | ms | GFLOPS |
|-----|------|--------|
| 128 | 0.21 | 1584 |
| 256 | 0.41 | 1709 |
| **512** | **2.20** | **733** |

**Kritischer Kipppunkt bei seq=512** — GFLOPS brechen um 57% ein.
Attention Scores [32, 512, 512] = 32MB ≈ SRAM Limit.
**Empfehlung: seq ≤ 256 fuer ANE Attention, seq > 256 auf GPU.**

### 2. GPU FFN (MPS) — Sweet Spot: grosse Matrizen, seq≥128

**FFN-Ratio Sweep (dim=768, seq=128):**

| Ratio | ffn | ms | GFLOPS |
|-------|------|------|--------|
| 2.0x | 1536 | 1.22 | 741 |
| 2.67x | 2048 | 1.43 | 843 |
| 3.0x | 2304 | 1.55 | 878 |
| **4.0x** | **3072** | **1.62** | **1117** |
| **6.0x** | **4608** | **2.22** | **1226** |

**Beobachtung:** GFLOPS steigt mit FFN-Groesse! GPU wird effizienter bei groesseren Matmuls.
**Empfehlung: FFN-Ratio 4x oder hoeher bevorzugen fuer GPU.**

**Dim Sweep (ratio=3x, seq=128):**

| dim | ffn | ms | GFLOPS |
|-----|------|------|--------|
| 256 | 768 | 0.66 | 229 |
| 512 | 1536 | 0.99 | 613 |
| 768 | 2304 | 1.46 | 934 |
| 1024 | 3072 | 2.10 | 1152 |
| **2048** | **6144** | **4.19** | **2309** |

GPU wird bei grossem dim viel effizienter. **Peak 2309 GFLOPS bei dim=2048.**

**Seq Sweep (dim=768, ffn=2304):**

| seq | ms | GFLOPS |
|------|------|--------|
| 32 | 0.99 | 344 |
| 64 | 1.15 | 588 |
| 128 | 1.53 | 886 |
| 256 | 2.15 | 1263 |
| **512** | **2.04** | **2660** |

GPU wird SCHNELLER (pro Element) bei laengerem seq! **Peak Effizienz bei seq=512.**

### 3. AMX (cblas_sgemm) — Konstant ~1500 GFLOPS, seq-sensitiv

**FFN-aehnliche Matmuls:**

| M (ffn) | K (dim) | N (seq) | ms | GFLOPS |
|---------|---------|---------|-------|--------|
| 1536 | 768 | 128 | 0.192 | 1572 |
| 2304 | 768 | 128 | 0.288 | 1571 |
| 3072 | 768 | 128 | 0.374 | 1615 |
| 4608 | 768 | 128 | 0.652 | 1389 |

AMX ist konstant ~1500 GFLOPS unabhaengig von der FFN-Groesse.

**Seq Sweep:**

| seq | ms | GFLOPS |
|------|-------|--------|
| 32 | 0.113 | 1002 |
| 64 | 0.170 | 1336 |
| 128 | 0.282 | 1604 |
| 256 | 0.508 | 1783 |
| 512 | 0.854 | 2121 |

AMX wird effizienter mit groesserem seq (mehr Daten pro Dispatch).

### 4. GPU vs AMX Crossover

| dim | ffn | seq | GPU (GFLOPS) | AMX (GFLOPS) | Winner |
|-----|------|-----|-------------|-------------|--------|
| 256 | 768 | 128 | 229 | 1497 | **AMX 6.5x** |
| 512 | 1536 | 128 | 613 | 1451 | **AMX 2.4x** |
| 768 | 2304 | 128 | 886 | 1604 | **AMX 1.8x** |
| 1024 | 3072 | 128 | 1152 | 1343 | **AMX 1.2x** |
| 2048 | 6144 | 128 | 2309 | 1422 | **GPU 1.6x** |

**Crossover bei dim ≈ 1024-1500.** Unter 1024: AMX gewinnt. Ueber 1500: GPU gewinnt.

Bei seq=512: GPU=2660, AMX=2121 → **GPU gewinnt immer bei langem seq.**

---

## Architektur-Empfehlung

Basierend auf den Sweep-Ergebnissen, optimale Konfiguration fuer M4:

### Option A: "Balanced" (dim=768, 12 heads)
- ANE Attention: 0.21ms/Layer, 1584 GFLOPS
- GPU FFN (3x=2304): 1.46ms/Layer, 934 GFLOPS
- AMX FFN: 0.28ms, 1604 GFLOPS → **AMX gewinnt fuer FFN bei dim=768!**
- Seq sweet spot: 128-256

### Option B: "Wide" (dim=1024, 16 heads)
- ANE Attention: 0.27ms/Layer, 2136 GFLOPS
- GPU FFN (3x=3072): 2.10ms/Layer, 1152 GFLOPS
- AMX FFN: 0.60ms, 1343 GFLOPS → **AMX leicht besser**
- Am GPU/AMX Crossover

### Option C: "ANE-Optimized" (dim=384, 6 heads, ffn=4x=1536)
- ANE Attention: 0.11ms/Layer (ultraschnell)
- AMX FFN (4x=1536): 0.19ms, 1572 GFLOPS → **AMX fuer FFN**
- GPU: frei fuer andere Arbeit (Decode, Klassifikation)
- Kleineres Modell, aber alle 3 Units parallel nutzbar

### Kritische Einsichten

1. **ANE dominiert Attention** bei allen Dimensionen (0.11-0.93ms vs GPU mehrere ms)
2. **AMX schlaegt GPU fuer FFN** bei dim ≤ 1024 — das haben wir bisher ignoriert!
3. **GPU wird erst bei dim > 1500 effizienter als AMX** fuer FFN
4. **seq > 256 bricht ANE Attention** — fuer lange Kontexte muss Attention anders gehandhabt werden
5. **FFN-Ratio 4x+ ist GPU-effizienter** als 3x — breitere FFN bevorzugen

### Empfohlene Architektur: "Fiber-Native"

```
dim=768, heads=12, head_dim=64, ffn=3072 (4x), max_seq=256
22-28 Layers fuer ~1B-2B Parameter

Pipeline pro Layer:
  ANE: Fused SDPA (0.21ms)         — 19 TFLOPS, Attention-Spezialist
  AMX: FFN W1+W3 Matmul (0.37ms)   — 1.6 TFLOPS, schlaegt GPU bei dim=768!
  GPU: FFN W2 + SiLU + Residual     — parallel zu AMX
  CPU: Orchestrierung, KV Cache
```

Geschaetzte Layer-Zeit: ~0.6ms (ANE+AMX parallel, GPU parallel)
= **1700+ tok/s Prefill** fuer seq=128 (theoretisch)

### Offene Fragen

- Kann AMX FFN parallel zu ANE Attention laufen? (M4_RE Exp 06 sagt ja: 103% Effizienz)
- Wie gut funktioniert AMX mit FP16 Weights + FP32 Compute? (Dequant-Overhead?)
- Lohnt sich INT8 auf AMX/SME? (4 TOPS vs 2 TFLOPS FP32)
