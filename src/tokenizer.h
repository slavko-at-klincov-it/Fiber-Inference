// tokenizer.h — BPE tokenizer reading vocab from GGUF metadata

#ifndef TOKENIZER_H
#define TOKENIZER_H

#include "gguf.h"

typedef struct tokenizer tokenizer_t;

// Initialize tokenizer from GGUF metadata (tokenizer.ggml.*)
tokenizer_t *tokenizer_init(const gguf_file_t *gf);
void tokenizer_free(tokenizer_t *tok);

// Encode text to token IDs. Returns number of tokens written.
int tokenizer_encode(const tokenizer_t *tok, const char *text,
                     int *tokens, int max_tokens);

// Decode single token to string (valid until next decode call)
const char *tokenizer_decode(const tokenizer_t *tok, int token_id);

// Special token IDs
int tokenizer_bos(const tokenizer_t *tok);
int tokenizer_eos(const tokenizer_t *tok);
int tokenizer_vocab_size(const tokenizer_t *tok);

#endif // TOKENIZER_H
