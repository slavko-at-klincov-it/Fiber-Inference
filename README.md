# Fiber-Inference

Multi-accelerator LLM inference engine for Apple M4, combining ANE attention + GPU FFN
with mmap/SSD weight offloading.

**What this enables on M4 16GB that no existing framework can do:**
- 30B Q4 models (normally: max ~14B)
- 128K context on 7B (normally: max ~32K)
- Higher tok/s through parallel ANE+GPU execution

## Status

Phase 0: GGUF Loader + Skeleton

## Build

```bash
make
./fiber-inference --model models/your-model.gguf
```

## Architecture

```
Per token, per layer:
  ANE  ──── Fused SDPA (7.5x faster than GPU for attention) ────┐
  GPU  ──── FFN (dequant Q4→FP16 + MPS matmul + SiLU) ─────────┤
  SSD  ──── Prefetch next layer weights (E-Core background) ────┤
  CPU  ──── Residual add + orchestration ────────────────────────┘
                                                                  ↓
                                                            Next layer
```

All accelerators share Unified Memory (zero-copy via IOSurface).

## Requirements

- macOS 26.x (Apple Silicon M4)
- libane (symlinked from ANE-Training project)
- GGUF model files (Q4_K_M recommended)
