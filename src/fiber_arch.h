// fiber_arch.h — Fiber-768: Apple-Silicon-Native LLM Architecture
// Dimensions derived from hardware sweep (docs/hardware-sweep.md)
#pragma once

// Architecture constants
#define FIBER_DIM         768    // ANE sweet spot: 1584 GFLOPS attention
#define FIBER_HEADS       12     // head_dim = 64
#define FIBER_KV_HEADS    4      // GQA 3:1 — saves KV cache memory
#define FIBER_HEAD_DIM    64     // FIBER_DIM / FIBER_HEADS
#define FIBER_FFN_DIM     3072   // 4x ratio — GPU-efficient per sweep
#define FIBER_LAYERS      24     // ~800M parameters
#define FIBER_VOCAB       32000  // Standard
#define FIBER_MAX_SEQ     256    // ANE SRAM limit — beyond this perf drops 57%
#define FIBER_RMS_EPS     1e-5f
#define FIBER_ROPE_BASE   10000.0f

// Derived
#define FIBER_KV_DIM      (FIBER_KV_HEADS * FIBER_HEAD_DIM)  // 256
#define FIBER_GQA_RATIO   (FIBER_HEADS / FIBER_KV_HEADS)     // 3

// Estimated parameters: LAYERS * (4*DIM^2 + 3*DIM*FFN_DIM) ≈ 800M
