// sampler.c — Temperature / Top-k / Top-p sampling

#include "sampler.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

// Simple xorshift64 RNG
static float rng_float(unsigned long long *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (float)((*state * 0x2545F4914F6CDD1DULL) >> 40) / (float)(1 << 24);
}

void sampler_init(sampler_t *s, int vocab_size, float temp, int top_k, float top_p) {
    s->vocab_size = vocab_size;
    s->temperature = temp;
    s->top_k = top_k > 0 ? top_k : vocab_size;
    s->top_p = top_p;
    s->rng_state = 42;
}

// Sort indices by value descending (for top-k/top-p)
typedef struct { float val; int idx; } scored_t;

static int cmp_scored_desc(const void *a, const void *b) {
    float va = ((const scored_t *)a)->val;
    float vb = ((const scored_t *)b)->val;
    return (va < vb) - (va > vb);
}

int sampler_sample(sampler_t *s, const float *logits) {
    int n = s->vocab_size;
    float temp = s->temperature;

    if (temp <= 0.0f || temp < 1e-6f) {
        // Argmax (greedy)
        int best = 0;
        float best_val = logits[0];
        for (int i = 1; i < n; i++) {
            if (logits[i] > best_val) { best_val = logits[i]; best = i; }
        }
        return best;
    }

    // Apply temperature
    float *probs = malloc(n * sizeof(float));
    float inv_temp = 1.0f / temp;
    for (int i = 0; i < n; i++) {
        probs[i] = logits[i] * inv_temp;
    }

    // Top-k: keep only top_k highest logits
    int k = s->top_k < n ? s->top_k : n;

    scored_t *scored = malloc(n * sizeof(scored_t));
    for (int i = 0; i < n; i++) {
        scored[i].val = probs[i];
        scored[i].idx = i;
    }

    // Partial sort: only need top k
    if (k < n) {
        qsort(scored, n, sizeof(scored_t), cmp_scored_desc);
    }

    // Softmax over top-k
    float max_val = scored[0].val;
    float sum = 0.0f;
    for (int i = 0; i < k; i++) {
        scored[i].val = expf(scored[i].val - max_val);
        sum += scored[i].val;
    }
    for (int i = 0; i < k; i++) {
        scored[i].val /= sum;
    }

    // Top-p (nucleus sampling): truncate to cumulative probability p
    int cutoff = k;
    if (s->top_p < 1.0f) {
        float cumsum = 0.0f;
        for (int i = 0; i < k; i++) {
            cumsum += scored[i].val;
            if (cumsum >= s->top_p) {
                cutoff = i + 1;
                break;
            }
        }
        // Re-normalize
        sum = 0.0f;
        for (int i = 0; i < cutoff; i++) sum += scored[i].val;
        for (int i = 0; i < cutoff; i++) scored[i].val /= sum;
    }

    // Sample from distribution
    float r = rng_float(&s->rng_state);
    float cumsum = 0.0f;
    int selected = scored[0].idx;
    for (int i = 0; i < cutoff; i++) {
        cumsum += scored[i].val;
        if (r <= cumsum) {
            selected = scored[i].idx;
            break;
        }
    }

    free(probs);
    free(scored);
    return selected;
}
