// tokenizer.c — SentencePiece BPE tokenizer reading vocab from GGUF metadata

#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_TOKEN_LEN 128
#define MAX_ENCODE_TOKENS 8192

// --- Tokenizer structure ---

struct tokenizer {
    int vocab_size;
    char **vocab;       // [vocab_size] null-terminated strings
    float *scores;      // [vocab_size]
    int bos_id;
    int eos_id;
    int nl_id;          // newline token

    // Hash table: string → token ID (open addressing)
    uint32_t *ht_hash;
    int *ht_id;
    int ht_cap;         // power of 2
};

// --- FNV-1a hash ---

static uint32_t fnv1a(const char *s, int len) {
    uint32_t h = 2166136261u;
    for (int i = 0; i < len; i++) {
        h ^= (uint8_t)s[i];
        h *= 16777619u;
    }
    return h | 1;  // never 0 (0 = empty slot)
}

static void ht_insert(tokenizer_t *tok, const char *s, int len, int id) {
    uint32_t h = fnv1a(s, len);
    int mask = tok->ht_cap - 1;
    int idx = (int)(h & (uint32_t)mask);
    while (tok->ht_hash[idx] != 0) {
        idx = (idx + 1) & mask;
    }
    tok->ht_hash[idx] = h;
    tok->ht_id[idx] = id;
}

static int ht_lookup(const tokenizer_t *tok, const char *s, int len) {
    uint32_t h = fnv1a(s, len);
    int mask = tok->ht_cap - 1;
    int idx = (int)(h & (uint32_t)mask);
    while (tok->ht_hash[idx] != 0) {
        if (tok->ht_hash[idx] == h) {
            int id = tok->ht_id[idx];
            if ((int)strlen(tok->vocab[id]) == len &&
                memcmp(tok->vocab[id], s, len) == 0) {
                return id;
            }
        }
        idx = (idx + 1) & mask;
    }
    return -1;
}

// --- UTF-8 helpers ---

static int utf8_len(uint8_t byte) {
    if ((byte & 0x80) == 0)    return 1;
    if ((byte & 0xE0) == 0xC0) return 2;
    if ((byte & 0xF0) == 0xE0) return 3;
    if ((byte & 0xF8) == 0xF0) return 4;
    return 1;
}

// --- Init ---

tokenizer_t *tokenizer_init(const gguf_file_t *gf) {
    // Find vocab array
    const gguf_kv_t *vocab_kv = gguf_find_kv(gf, "tokenizer.ggml.tokens");
    if (!vocab_kv || vocab_kv->type != GGUF_TYPE_ARRAY ||
        vocab_kv->value.arr.elem_type != GGUF_TYPE_STRING) {
        fprintf(stderr, "tokenizer: missing tokenizer.ggml.tokens\n");
        return NULL;
    }

    int vocab_size = (int)vocab_kv->value.arr.count;

    // Find scores array
    const gguf_kv_t *scores_kv = gguf_find_kv(gf, "tokenizer.ggml.scores");
    const float *raw_scores = NULL;
    if (scores_kv && scores_kv->type == GGUF_TYPE_ARRAY &&
        scores_kv->value.arr.elem_type == GGUF_TYPE_FLOAT32) {
        raw_scores = (const float *)scores_kv->value.arr.data;
    }

    // Allocate tokenizer
    tokenizer_t *tok = calloc(1, sizeof(tokenizer_t));
    tok->vocab_size = vocab_size;
    tok->vocab = calloc(vocab_size, sizeof(char *));
    tok->scores = calloc(vocab_size, sizeof(float));

    // Hash table: 4x vocab size for low collision rate
    tok->ht_cap = 1;
    while (tok->ht_cap < vocab_size * 4) tok->ht_cap <<= 1;
    tok->ht_hash = calloc(tok->ht_cap, sizeof(uint32_t));
    tok->ht_id = calloc(tok->ht_cap, sizeof(int));

    // Extract vocab strings from GGUF string array
    const uint8_t *ptr = (const uint8_t *)vocab_kv->value.arr.data;
    for (int i = 0; i < vocab_size; i++) {
        uint64_t len;
        memcpy(&len, ptr, 8);
        ptr += 8;

        tok->vocab[i] = malloc(len + 1);
        memcpy(tok->vocab[i], ptr, len);
        tok->vocab[i][len] = '\0';
        ptr += len;

        tok->scores[i] = raw_scores ? raw_scores[i] : 0.0f;

        // Insert into hash table
        ht_insert(tok, tok->vocab[i], (int)len, i);
    }

    // Special tokens
    tok->bos_id = (int)gguf_get_u32(gf, "tokenizer.ggml.bos_token_id", 1);
    tok->eos_id = (int)gguf_get_u32(gf, "tokenizer.ggml.eos_token_id", 2);

    // Find newline token
    tok->nl_id = ht_lookup(tok, "\n", 1);

    printf("Tokenizer: %d tokens, BOS=%d, EOS=%d\n", vocab_size, tok->bos_id, tok->eos_id);
    return tok;
}

void tokenizer_free(tokenizer_t *tok) {
    if (!tok) return;
    for (int i = 0; i < tok->vocab_size; i++) free(tok->vocab[i]);
    free(tok->vocab);
    free(tok->scores);
    free(tok->ht_hash);
    free(tok->ht_id);
    free(tok);
}

// --- BPE Encode ---

typedef struct {
    int id;
    char text[MAX_TOKEN_LEN];
    int len;
} bpe_tok_t;

int tokenizer_encode(const tokenizer_t *tok, const char *text,
                     int *tokens, int max_tokens) {
    if (!text || !text[0]) return 0;

    // Step 1: Build processed text (prepend ▁, replace spaces with ▁)
    int text_len = (int)strlen(text);
    int buf_size = text_len * 3 + 16;
    char *processed = malloc(buf_size);
    int pos = 0;

    // Prepend ▁
    processed[pos++] = (char)0xE2;
    processed[pos++] = (char)0x96;
    processed[pos++] = (char)0x81;

    for (int i = 0; i < text_len; i++) {
        if (text[i] == ' ') {
            processed[pos++] = (char)0xE2;
            processed[pos++] = (char)0x96;
            processed[pos++] = (char)0x81;
        } else {
            processed[pos++] = text[i];
        }
    }
    processed[pos] = '\0';

    // Step 2: Split into UTF-8 characters, map to initial tokens
    bpe_tok_t *toks = malloc(MAX_ENCODE_TOKENS * sizeof(bpe_tok_t));
    int n = 0;

    int i = 0;
    while (i < pos && n < MAX_ENCODE_TOKENS - 1) {
        int clen = utf8_len((uint8_t)processed[i]);
        if (i + clen > pos) clen = 1;

        // Try to find this character in vocab
        int id = ht_lookup(tok, processed + i, clen);
        if (id >= 0) {
            toks[n].id = id;
            memcpy(toks[n].text, processed + i, clen);
            toks[n].text[clen] = '\0';
            toks[n].len = clen;
            n++;
        } else {
            // Byte fallback: each byte becomes a <0xNN> token
            for (int b = 0; b < clen && n < MAX_ENCODE_TOKENS - 1; b++) {
                char hex[8];
                snprintf(hex, sizeof(hex), "<0x%02X>", (uint8_t)processed[i + b]);
                int bid = ht_lookup(tok, hex, 6);
                if (bid >= 0) {
                    toks[n].id = bid;
                    memcpy(toks[n].text, hex, 6);
                    toks[n].text[6] = '\0';
                    toks[n].len = 6;
                } else {
                    // Unknown byte, use id 0 (unk)
                    toks[n].id = 0;
                    toks[n].text[0] = processed[i + b];
                    toks[n].text[1] = '\0';
                    toks[n].len = 1;
                }
                n++;
            }
        }
        i += clen;
    }
    free(processed);

    // Step 3: BPE merge loop — iteratively merge highest-scoring pair
    while (n > 1) {
        float best_score = -1e30f;
        int best_pos = -1;
        int best_id = -1;

        for (int j = 0; j < n - 1; j++) {
            int mlen = toks[j].len + toks[j + 1].len;
            if (mlen >= MAX_TOKEN_LEN) continue;

            char merged[MAX_TOKEN_LEN];
            memcpy(merged, toks[j].text, toks[j].len);
            memcpy(merged + toks[j].len, toks[j + 1].text, toks[j + 1].len);

            int id = ht_lookup(tok, merged, mlen);
            if (id >= 0 && tok->scores[id] > best_score) {
                best_score = tok->scores[id];
                best_pos = j;
                best_id = id;
            }
        }

        if (best_pos < 0) break;

        // Merge tokens[best_pos] and tokens[best_pos+1]
        int mlen = toks[best_pos].len + toks[best_pos + 1].len;
        memcpy(toks[best_pos].text + toks[best_pos].len,
               toks[best_pos + 1].text, toks[best_pos + 1].len);
        toks[best_pos].len = mlen;
        toks[best_pos].text[mlen] = '\0';
        toks[best_pos].id = best_id;

        // Shift remaining
        for (int j = best_pos + 1; j < n - 1; j++) {
            toks[j] = toks[j + 1];
        }
        n--;
    }

    // Step 4: Output
    int out = 0;
    for (int j = 0; j < n && out < max_tokens; j++) {
        tokens[out++] = toks[j].id;
    }

    free(toks);
    return out;
}

// --- Decode ---

const char *tokenizer_decode(const tokenizer_t *tok, int token_id) {
    if (token_id < 0 || token_id >= tok->vocab_size) return "";

    const char *s = tok->vocab[token_id];
    int slen = (int)strlen(s);

    // Handle byte fallback tokens: <0xNN>
    if (slen == 6 && s[0] == '<' && s[1] == '0' && s[2] == 'x' && s[5] == '>') {
        static char byte_buf[2];
        byte_buf[0] = (char)strtol(s + 3, NULL, 16);
        byte_buf[1] = '\0';
        return byte_buf;
    }

    // Replace ▁ (0xE2 0x96 0x81) with space
    static char decode_buf[512];
    int j = 0;
    for (int i = 0; i < slen && j < 510;) {
        if (i + 2 < slen &&
            (uint8_t)s[i] == 0xE2 && (uint8_t)s[i+1] == 0x96 && (uint8_t)s[i+2] == 0x81) {
            decode_buf[j++] = ' ';
            i += 3;
        } else {
            decode_buf[j++] = s[i++];
        }
    }
    decode_buf[j] = '\0';
    return decode_buf;
}

// --- Special tokens ---

int tokenizer_bos(const tokenizer_t *tok) { return tok->bos_id; }
int tokenizer_eos(const tokenizer_t *tok) { return tok->eos_id; }
int tokenizer_vocab_size(const tokenizer_t *tok) { return tok->vocab_size; }
