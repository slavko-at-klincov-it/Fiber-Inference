# Einsatzmoeglichkeiten: Wo kleine dim-Modelle auf ANE glaenzen

Date: 2026-04-05

## Die Erkenntnis

Kleine Modelle (dim≤1024, ≤1B Parameter) sind nicht "schlechtes GPT" —
sie haben eigene Staerken. Und genau dort hat unsere ANE-Architektur
einen echten, gemessenen Vorteil gegenueber GPU-only Engines.

## 5 Einsatzgebiete

### 1. RAG (Retrieval Augmented Generation) — Sweet Spot

Bei RAG gibt das Modell nicht "aus dem Kopf" Antworten — es bekommt
den relevanten Text als Input und muss ihn nur zusammenfassen oder
die richtige Stelle finden. Dafuer braucht man kein 8B Modell.

RAG hat **lange Prompts** (injizierter Kontext) + **kurze Antworten**.
ANE Prefill ist 2-3x schneller → User wartet weniger auf erste Antwort.

### 2. Spezialisierte Assistenten (Finetuned)

Ein 600M Modell das auf EINE Aufgabe finetuned ist, schlaegt ein
generalistisches 8B Modell — fuer genau diese Aufgabe:

- E-Mail Zusammenfassung
- Code-Completion (einzelne Zeilen)
- Formular-/PDF-Extraktion
- Sentiment-Analyse
- FAQ-Chatbot
- Einfache Uebersetzung

### 3. On-Device Privacy

Alles lokal, nichts an Cloud:
- Medizinische Daten
- Juristische Dokumente
- Firmeninterne Informationen
- Persoenliche Notizen

600M Modell = ~1.2 GB RAM. Passt auf jedes iPhone/Mac.

### 4. Latenz-kritische Anwendungen

Antwort in unter 50ms:
- Autocomplete waehrend dem Tippen
- Real-time Uebersetzung
- IDE Code Suggestions
- Sprach-Assistent

ANE: 100-Token Prompt in 82ms. GPU: 200ms. Spuerbarer Unterschied.

### 5. Batch-Verarbeitung

1000 Dokumente klassifizieren/zusammenfassen/taggen:
- ANE: 10,000+ tok/s Prefill → Sekunden
- GPU: 1,400 tok/s → 7x langsamer

## Warum ANE hier gewinnt [GEMESSEN]

| Szenario | ANE Vorteil | Grund |
|----------|------------|-------|
| Langer Prompt, kurze Antwort (RAG) | **2-3x schnellere Antwort** | Prefill dominiert, Decode ist kurz |
| Batch-Verarbeitung | **5-10x Durchsatz** | Nur Prefill, kein Decode |
| Latenz-kritisch | **50% schnelleres TTFT** | ANE Prefill ist 2x schneller |
| On-Device | **Gleich schnell, weniger Power** | ANE: ~2W vs GPU: ~20W |

## Groesste Chance: Spezialisierter RAG-Assistent

Nicht "LLM das alles kann" — sondern **schneller Spezialist**
der eine Sache richtig gut macht. Details: siehe `rag-assistent/` Ordner.
