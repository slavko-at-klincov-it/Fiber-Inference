# LinkedIn Post

---

Ich habe alle 5 Compute Units des Apple M4 Chips systematisch fuer LLM Inference getestet. 44 Commits, ueber 200 Messungen, ein klares Ergebnis.

Der M4 hat neben der GPU noch 4 weitere programmierbare Einheiten — die Apple Neural Engine (19 TFLOPS), den AMX Matrix Coprocessor (2 TFLOPS), Performance Cores und Efficiency Cores. Zusammen ueber 27 TFLOPS an ungenutztem Potenzial.

Die Frage: Kann man diese fuer schnellere lokale KI nutzen?

Die kurze Antwort: Nein — und hier ist warum.

Meine wichtigsten Findings:

1. Die Neural Engine (ANE) kann korrekte Transformer-Layer ausfuehren. Wir haben kohaerenten Text damit generiert — eine Premiere ueber die private API. Aber sie verliert gegen Apple's eigene MLX-Engine weil MLX quantisierte Weights nutzt (4x weniger Memory-Transfer).

2. Der AMX Coprozessor ist 1.8x schneller als die GPU fuer bestimmte Matmuls bei dim<=1024. Kein LLM-Framework nutzt das — aber der Vorteil ist auf kleine Modelle beschraenkt.

3. Alle Units parallel auf dasselbe Modell ansetzen? Katastrophal. Sie konkurrieren um die 120 GB/s Shared Memory Bandwidth und werden 30-40% LANGSAMER statt schneller.

4. Apple's MLX (das auch Ollama nutzt) ist 2.2x schneller als unsere beste Multi-Unit Konfiguration. Vertikale Integration (Hardware + Software vom selben Hersteller) schlaegt Reverse Engineering.

Die Erkenntnis: Fuer lokale LLMs auf Apple Silicon ist Bandwidth > TFLOPS. Mehr Rechenkerne helfen nicht wenn alle dieselben Gewichte aus dem Speicher lesen muessen. Deshalb setzt Apple auf bessere Quantisierung und hoehere Bandwidth (M5: 153 GB/s) — nicht auf mehr Units.

Alle Benchmarks, Code und Messdaten sind Open Source (Link im PDF).

Geschrieben mit Unterstuetzung von Claude (Anthropic) als technischem Sparringspartner und Co-Autor fuer Code und Benchmarks.

#AppleSilicon #MachineLearning #LLM #LocalAI #MLX #NeuralEngine #Hardware #Performance #Research

---
