// sampler.h — Temperature / Top-k / Top-p sampling

#ifndef SAMPLER_H
#define SAMPLER_H

typedef struct {
    int vocab_size;
    float temperature;
    int top_k;
    float top_p;
    unsigned long long rng_state;
} sampler_t;

// Initialize sampler with given parameters
void sampler_init(sampler_t *s, int vocab_size, float temp, int top_k, float top_p);

// Sample next token from F32 logits. Returns token ID.
int sampler_sample(sampler_t *s, const float *logits);

#endif // SAMPLER_H
