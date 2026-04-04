# Limit Tests: Wo bricht die Fiber-Architektur ein?

Date: 2026-04-04

## Test 1: Sequenzlaenge [MEASURED]

dim=768, heads=12, MHA, FFN=2048

| seq | Attn ms | Attn tok/s | FFN ms | FFN tok/s | Layer tok/s |
|-----|---------|-----------|--------|----------|-------------|
| 128 | 0.20 | 642,678 | 0.24 | 537,157 | 292,599 |
| 192 | 0.27 | 718,989 | 0.26 | 745,872 | 366,092 |
| **256** | **0.42** | **608,859** | **0.27** | **962,858** | **372,997** |
| 384 | 0.97 | 394,656 | 0.36 | 1,062,241 | 287,748 |
| 512 | 1.54 | 331,553 | 0.44 | 1,166,287 | 258,162 |
| 768 | 5.43 | 141,382 | 0.69 | 1,116,009 | 125,485 |
| 1024 | 7.06 | 145,123 | 0.86 | 1,190,755 | 129,358 |

**Erkenntnisse:**
- **Attention bricht ab seq=384** — 0.42ms → 0.97ms (2.3x Sprung). SRAM Limit.
- **FFN skaliert linear** — kein Einbruch bis seq=1024. FFN ist SRAM-freundlicher.
- **Sweet Spot: seq=192-256** — bester Layer-Durchsatz (366K-373K tok/s/Layer)
- **seq=768-1024 noch funktional** aber Attention 10-17x langsamer als bei seq=128

## Test 2: Dimension [MEASURED]

seq=256, FFN=2.67x dim, MHA

| dim | heads | Attn ms | Attn tok/s | FFN ms | FFN tok/s | RSS MB |
|-----|-------|---------|-----------|--------|----------|--------|
| **256** | 4 | **0.16** | **1,599,167** | **0.07** | **3,753,207** | 54.5 |
| 384 | 6 | 0.21 | 1,196,262 | 0.12 | 2,122,280 | 54.6 |
| 512 | 8 | 0.29 | 892,245 | 0.17 | 1,483,341 | 54.6 |
| **768** | **12** | **0.42** | **610,069** | **0.28** | **913,606** | **55.7** |
| 1024 | 16 | 0.63 | 404,850 | 0.40 | 642,476 | 82.5 |
| 1536 | 24 | 1.10 | 232,771 | 1.25 | 204,807 | 171.2 |
| **2048** | **32** | **2.91** | **87,979** | **3.18** | **80,526** | **278.0** |

**Erkenntnisse:**
- **dim=256 ist am schnellsten** (1.6M tok/s Attention, 3.7M tok/s FFN pro Layer!)
- **dim ≤ 768 passt in ~56MB RSS** — darüber steigt RAM schnell
- **dim=1536+ wird langsam** — ANE Attention und FFN beide > 1ms/Layer
- **dim=2048 ist 7x langsamer als dim=768** — nicht empfohlen fuer ANE-native

**Fuer maximale Speed: dim=256-384. Fuer beste Qualitaet/Speed Balance: dim=768.**

## Test 3: GQA Ratio [MEASURED]

dim=768, 12 Q-Heads, seq=256

| KV Heads | Typ | Attn ms | tok/s |
|----------|-----|---------|-------|
| 12 | MHA | 0.41 | 621,988 |
| 6 | GQA 2:1 | 0.39 | 649,403 |
| 4 | GQA 3:1 | 0.39 | 653,061 |
| 3 | GQA 4:1 | 0.40 | 646,805 |
| 2 | GQA 6:1 | 0.37 | 685,791 |
| **1** | **GQA 12:1** | **0.36** | **704,103** |

**Erkenntnisse:**
- **GQA hat keinen nennenswerten Speed-Impact auf ANE** — alle Varianten ~0.36-0.42ms
- GQA 12:1 (MQA) ist marginal am schnellsten (13% ueber MHA)
- Der Tile-Overhead im MIL ist vernachlaessigbar
- **GQA lohnt sich trotzdem: spart KV-Cache Memory** (12x weniger bei MQA)

## Test 4: Compile Budget [MEASURED]

Nach allen Tests: 34/119 Kernels kompiliert. **85 verbleibend.**
= Genug fuer ~3 verschiedene Modellkonfigurationen à 24 Layers.

## Zusammenfassung: Grenzen der Architektur

| Grenze | Wert | Was passiert |
|--------|------|-------------|
| **seq > 256** | Attention 2-17x langsamer | SRAM Spill bei Attention Scores |
| **dim > 1024** | Beide Ops > 1ms/Layer | Weight-Tensoren zu gross fuer SRAM |
| **dim > 2048** | ~3ms/Layer, 278MB RSS | Unpraktisch fuer ANE-native |
| **Compile Budget** | 119 pro Prozess | ~3 Modelle à 24 Layers |
| **FFN** | Skaliert linear bis seq=1024 | Kein SRAM-Problem bei FFN |
| **GQA** | Kein Speed-Impact | Nur Memory-Vorteil |

## Empfehlung fuer Produktion

Basierend auf allen Tests:

```
Maximale Speed:   dim=256, 4 heads, seq=256 → 1.6M tok/s pro Attention-Layer
Balanced:         dim=768, 12 heads, seq=256 → 373K tok/s pro Layer
Qualitaet:        dim=1024, 16 heads, seq=256 → 240K tok/s pro Layer
Nicht empfohlen:  dim > 1536 oder seq > 512 auf ANE
```

Fuer Kontexte > 256 Tokens: Sliding Window Attention oder Chunked Attention
noetig (ANE fuer Chunks ≤ 256, GPU fuer Cross-Chunk Attention).
