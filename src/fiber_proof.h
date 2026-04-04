// fiber_proof.h — A/B proof: same model, GPU vs ANE, compare text + speed
#ifndef FIBER_PROOF_H
#define FIBER_PROOF_H

// Run the full proof:
// 1. Load Stories-110M from BLZT checkpoint
// 2. Load tokenizer from TinyLlama GGUF
// 3. GPU-only forward → generate text
// 4. ANE-only forward → generate text
// 5. Compare output + measure speed
void fiber_proof(const char *ckpt_path, const char *gguf_path, const char *prompt);

// Extended proof: sweep dim/heads/layers with synthetic weights,
// CPU vs ANE for each config, verify token match.
void fiber_proof_sweep(void);

#endif
