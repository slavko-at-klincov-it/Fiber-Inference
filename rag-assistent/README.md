# RAG-Assistent auf Apple Silicon

Ein spezialisierter Retrieval Augmented Generation Assistent der die
Apple M4 Neural Engine fuer schnellstes Time-to-First-Token nutzt.

## Warum RAG + ANE zusammenpassen

RAG-Workload = **langer Prompt** (Kontext-Dokumente) + **kurze Antwort** (Extrakt/Zusammenfassung).

| Phase | Anteil | Bester Accelerator |
|-------|--------|-------------------|
| Retrieval (Embedding, Vektorsuche) | ~100ms | CPU/GPU |
| **Prefill (Kontext verarbeiten)** | **~50-150ms** | **ANE (2-3x schneller als GPU)** |
| Decode (Antwort generieren) | ~200ms (kurz!) | GPU |

ANE-Vorteil: Prefill ist der groesste Teil, und genau da ist ANE 2-3x schneller.
Kurzer Decode (2-3 Saetze) minimiert den GPU-Nachteil.

## Architektur

```
┌─────────────────────────────────────────────┐
│  User-Frage                                  │
│  "Was steht im Vertrag ueber Kuendigung?"   │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌──────────────────────────┐
│  1. RETRIEVAL (CPU/GPU)  │
│  Embedding der Frage     │
│  Vektorsuche in Docs     │
│  Top-5 relevante Chunks  │
│  ~100ms                  │
└──────────────┬───────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  2. PROMPT BAUEN                              │
│  System: "Du bist ein Vertrags-Experte."     │
│  Kontext: [5 relevante Absaetze, ~500 Tokens] │
│  Frage: "Was steht ueber Kuendigung?"        │
│  Total: ~600 Tokens Prompt                    │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  3. ANE PREFILL                               │
│  600 Tokens durch 12-24 Layers               │
│  ANE: ~50ms (vs GPU: ~150ms)                 │
│  2-3x schneller als Standard                 │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  4. GPU DECODE                                │
│  Generiert 50-100 Tokens Antwort             │
│  GPU: ~1-2 Sekunden                          │
│  (kurze Antwort = wenig Decode noetig)       │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  Antwort                                      │
│  "Die Kuendigungsfrist betraegt 3 Monate     │
│   zum Quartalsende (§12 Abs. 3)."            │
│                                               │
│  Total: ~350ms statt ~550ms                  │
│  Time-to-First-Token: ~150ms statt ~250ms    │
└──────────────────────────────────────────────┘
```

## Komponenten

### Was wir schon haben (aus Fiber-Inference)

| Komponente | Status | Wo |
|-----------|--------|-----|
| ANE Attention Kernel (MIL) | Fertig | `src/ane_mil.h` |
| ANE FFN Kernel (MIL) | Fertig | `src/ane_mil.h` |
| ANE Prefill Pipeline | Fertig, 1,219 tok/s | `src/ane_attn.m` |
| GPU Decode (Multi-Row MatVec) | Fertig, 52 tok/s | `src/gpu_ffn.m`, `metal/kernels.metal` |
| Tokenizer (Llama2 BPE) | Fertig | `src/tokenizer.c` |
| GGUF Model Loading | Fertig | `src/model.m`, `src/gguf.c` |
| karpathy Model Loading | Fertig | `src/fiber_ckpt.m` |

### Was noch gebaut werden muss

| Komponente | Beschreibung | Aufwand |
|-----------|-------------|--------|
| **Embedding-Modell** | Text → Vektor (fuer Retrieval). Kann ein kleines Modell sein (z.B. all-MiniLM-L6-v2, dim=384) | Mittel |
| **Vektorstore** | In-Memory Vektor-Datenbank. Einfachste Version: Brute-Force Cosine Similarity | Klein |
| **Dokument-Loader** | Text/PDF/Markdown → Chunks (~200 Tokens pro Chunk) | Klein |
| **Prompt-Template** | System + Kontext + Frage zusammenbauen | Trivial |
| **LLM (dim≤1024)** | Qwen3-0.6B (dim=1024, GGUF) oder finetuned Stories-110M | Download/Training |
| **API/CLI** | Frage eingeben → Antwort bekommen | Klein |

## Modell-Optionen

| Modell | dim | Parameter | Qualitaet | ANE Prefill |
|--------|-----|-----------|-----------|-------------|
| Stories-110M | 768 | 110M | Nur Kindergeschichten | 1,219 tok/s |
| **Qwen3-0.6B** | **1024** | **600M** | **Brauchbar fuer einfache QA** | **~800 tok/s (geschaetzt)** |
| Finetuned 768-dim | 768 | ~400M | Spezialisiert auf DEINE Daten | 1,219 tok/s |
| Phi-1.5 | 2048 | 1.3B | Gut fuer Code | ~200 tok/s (ANE limitiert) |

**Empfehlung: Qwen3-0.6B (dim=1024)** — beste Balance aus Qualitaet und ANE-Speed.

## Erwartete Performance

| Metrik | RAG + ANE | RAG + GPU (Standard) | Vorteil |
|--------|----------|---------------------|---------|
| Time-to-First-Token | **~150ms** | ~250ms | **40% schneller** |
| Total Response Time | **~350ms** | ~550ms | **36% schneller** |
| Prefill (600 Tokens) | **~50ms** | ~150ms | **3x schneller** |
| Decode (50 Tokens) | ~500ms | ~500ms | gleich |
| RAM | ~2 GB | ~1 GB | mehr (FP16 vs Q4) |
| Power | ~5W avg | ~15W avg | **3x effizienter** |

## Anwendungsbeispiele

### 1. Vertrags-Assistent
- Dokumente: Vertraege, AGB, Datenschutzerklaerungen
- Frage: "Was sind die Kuendigungsfristen?"
- Antwort: Exakte Stelle + Zusammenfassung

### 2. Code-Dokumentation
- Dokumente: README, API Docs, Quellcode-Kommentare
- Frage: "Wie initialisiere ich die Datenbank-Verbindung?"
- Antwort: Code-Snippet + Erklaerung

### 3. Persoenlicher Wissens-Assistent
- Dokumente: Notizen, Bookmarks, gespeicherte Artikel
- Frage: "Was habe ich letzte Woche ueber X gelesen?"
- Antwort: Zusammenfassung der relevanten Notizen

### 4. Kunden-Support-Bot
- Dokumente: FAQ, Produktbeschreibungen, Support-Tickets
- Frage: "Wie setze ich mein Passwort zurueck?"
- Antwort: Schritt-fuer-Schritt Anleitung

## Naechste Schritte

1. **Qwen3-0.6B GGUF downloaden** und mit unserer Engine laden
2. **Embedding-Modell** integrieren (all-MiniLM-L6 oder aehnlich)
3. **Einfacher Vektorstore** (Cosine Similarity, In-Memory)
4. **Prompt-Template** fuer RAG bauen
5. **End-to-End Pipeline** testen: Dokument → Frage → Antwort
6. **Benchmark** gegen Ollama + gleiches Modell
