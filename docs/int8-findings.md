# INT8 auf ANE: Compile-Fehler [MEASURED]

Date: 2026-04-04

## Test

Versuch: FP16 MIL Kernel mit INT8 Weight-Blobs kompilieren.
Weights via `ane_weight_int8()` statt `ane_weight_fp16()`.
MIL bleibt gleich (deklariert `tensor<fp16, ...>` fuer Weights).

## Ergebnis

```
FP16 Attention: compile OK, 0.21 ms
INT8 Attention: compile FAILED
FP16 FFN: compile OK, 0.23 ms
INT8 FFN: compile FAILED
```

## Analyse

ANE Compiler lehnt INT8 Weight-Blobs ab wenn MIL FP16 Tensoren deklariert.
Das Weight-Format muss zum MIL-Datentyp passen.

Fuer echtes INT8 muesste der MIL Generator angepasst werden:
- `tensor<fp16, ...>` → `tensor<int8, ...>` fuer Weights
- Zusaetzliche `cast()` Ops im MIL fuer FP16↔INT8 Konvertierung
- Oder: `quantized_weight` MIL Ops (undokumentiert, moeglicherweise in CoreML)

## Fazit

INT8 auf ANE ist **nicht trivial** — braucht MIL-Umbau, nicht nur Weight-Format-Aenderung.
Aufwand: Hoch. Impact: Theoretisch 2x Speed (38 TOPS vs 19 TFLOPS).
Empfehlung: Spaeter angehen, nach Dynamic Weights (das ist der Enabler).
