// gguf.c — GGUF v3 parser implementation
// Reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

// --- Type size tables ---

typedef struct {
    size_t type_size;   // bytes per block
    size_t block_size;  // elements per block
    const char *name;
} type_info_t;

static const type_info_t TYPE_INFO[] = {
    [GGML_TYPE_F32]     = { 4,   1, "F32" },
    [GGML_TYPE_F16]     = { 2,   1, "F16" },
    [GGML_TYPE_Q4_0]    = { 18,  32, "Q4_0" },
    [GGML_TYPE_Q4_1]    = { 20,  32, "Q4_1" },
    [GGML_TYPE_Q5_0]    = { 22,  32, "Q5_0" },
    [GGML_TYPE_Q5_1]    = { 24,  32, "Q5_1" },
    [GGML_TYPE_Q8_0]    = { 34,  32, "Q8_0" },
    [GGML_TYPE_Q8_1]    = { 36,  32, "Q8_1" },
    [GGML_TYPE_Q2_K]    = { 84,  256, "Q2_K" },
    [GGML_TYPE_Q3_K]    = { 110, 256, "Q3_K" },
    [GGML_TYPE_Q4_K]    = { 144, 256, "Q4_K" },
    [GGML_TYPE_Q5_K]    = { 176, 256, "Q5_K" },
    [GGML_TYPE_Q6_K]    = { 210, 256, "Q6_K" },
    [GGML_TYPE_Q8_K]    = { 292, 256, "Q8_K" },
    [GGML_TYPE_IQ2_XXS] = { 66,  256, "IQ2_XXS" },
    [GGML_TYPE_IQ2_XS]  = { 74,  256, "IQ2_XS" },
    [GGML_TYPE_IQ3_XXS] = { 98,  256, "IQ3_XXS" },
    [GGML_TYPE_IQ1_S]   = { 50,  256, "IQ1_S" },
    [GGML_TYPE_IQ4_NL]  = { 18,  32,  "IQ4_NL" },
    [GGML_TYPE_IQ3_S]   = { 110, 256, "IQ3_S" },
    [GGML_TYPE_IQ2_S]   = { 82,  256, "IQ2_S" },
    [GGML_TYPE_IQ4_XS]  = { 36,  32,  "IQ4_XS" },
    [GGML_TYPE_I8]      = { 1,   1, "I8" },
    [GGML_TYPE_I16]     = { 2,   1, "I16" },
    [GGML_TYPE_I32]     = { 4,   1, "I32" },
    [GGML_TYPE_I64]     = { 8,   1, "I64" },
    [GGML_TYPE_F64]     = { 8,   1, "F64" },
    [GGML_TYPE_IQ1_M]   = { 56,  256, "IQ1_M" },
    [GGML_TYPE_BF16]    = { 2,   1, "BF16" },
};

#define N_TYPE_INFO (sizeof(TYPE_INFO) / sizeof(TYPE_INFO[0]))

size_t ggml_type_size(ggml_type_t type) {
    if ((size_t)type < N_TYPE_INFO) return TYPE_INFO[type].type_size;
    return 0;
}

size_t ggml_block_size(ggml_type_t type) {
    if ((size_t)type < N_TYPE_INFO) return TYPE_INFO[type].block_size;
    return 0;
}

const char *ggml_type_name(ggml_type_t type) {
    if ((size_t)type < N_TYPE_INFO && TYPE_INFO[type].name) return TYPE_INFO[type].name;
    return "UNKNOWN";
}

static const char *GGUF_TYPE_NAMES[] = {
    "UINT8", "INT8", "UINT16", "INT16", "UINT32", "INT32",
    "FLOAT32", "BOOL", "STRING", "ARRAY", "UINT64", "INT64", "FLOAT64"
};

const char *gguf_type_name(gguf_type_t type) {
    if ((size_t)type < sizeof(GGUF_TYPE_NAMES)/sizeof(GGUF_TYPE_NAMES[0]))
        return GGUF_TYPE_NAMES[type];
    return "UNKNOWN";
}

// --- Reader helpers ---

typedef struct {
    const uint8_t *base;
    size_t         pos;
    size_t         size;
} reader_t;

static bool reader_has(const reader_t *r, size_t n) {
    return r->pos + n <= r->size;
}

static uint8_t read_u8(reader_t *r) {
    if (!reader_has(r, 1)) return 0;
    return r->base[r->pos++];
}

static uint16_t read_u16(reader_t *r) {
    if (!reader_has(r, 2)) return 0;
    uint16_t v;
    memcpy(&v, r->base + r->pos, 2);
    r->pos += 2;
    return v;
}

static uint32_t read_u32(reader_t *r) {
    if (!reader_has(r, 4)) return 0;
    uint32_t v;
    memcpy(&v, r->base + r->pos, 4);
    r->pos += 4;
    return v;
}

static uint64_t read_u64(reader_t *r) {
    if (!reader_has(r, 8)) return 0;
    uint64_t v;
    memcpy(&v, r->base + r->pos, 8);
    r->pos += 8;
    return v;
}

static float read_f32(reader_t *r) {
    if (!reader_has(r, 4)) return 0;
    float v;
    memcpy(&v, r->base + r->pos, 4);
    r->pos += 4;
    return v;
}

static double read_f64(reader_t *r) {
    if (!reader_has(r, 8)) return 0;
    double v;
    memcpy(&v, r->base + r->pos, 8);
    r->pos += 8;
    return v;
}

static gguf_string_t read_string(reader_t *r) {
    gguf_string_t s = {0};
    s.len = read_u64(r);
    if (!reader_has(r, s.len)) { s.len = 0; return s; }
    s.str = (const char *)(r->base + r->pos);
    r->pos += s.len;
    return s;
}

static size_t gguf_value_type_size(gguf_type_t type) {
    switch (type) {
        case GGUF_TYPE_UINT8:   return 1;
        case GGUF_TYPE_INT8:    return 1;
        case GGUF_TYPE_UINT16:  return 2;
        case GGUF_TYPE_INT16:   return 2;
        case GGUF_TYPE_UINT32:  return 4;
        case GGUF_TYPE_INT32:   return 4;
        case GGUF_TYPE_FLOAT32: return 4;
        case GGUF_TYPE_BOOL:    return 1;
        case GGUF_TYPE_UINT64:  return 8;
        case GGUF_TYPE_INT64:   return 8;
        case GGUF_TYPE_FLOAT64: return 8;
        default: return 0;
    }
}

// Read a KV value (non-array, non-string types read inline)
static bool read_kv_value(reader_t *r, gguf_kv_t *kv) {
    switch (kv->type) {
        case GGUF_TYPE_UINT8:   kv->value.u8  = read_u8(r);  break;
        case GGUF_TYPE_INT8:    kv->value.i8  = (int8_t)read_u8(r); break;
        case GGUF_TYPE_UINT16:  kv->value.u16 = read_u16(r); break;
        case GGUF_TYPE_INT16:   kv->value.i16 = (int16_t)read_u16(r); break;
        case GGUF_TYPE_UINT32:  kv->value.u32 = read_u32(r); break;
        case GGUF_TYPE_INT32:   kv->value.i32 = (int32_t)read_u32(r); break;
        case GGUF_TYPE_FLOAT32: kv->value.f32 = read_f32(r); break;
        case GGUF_TYPE_BOOL:    kv->value.b   = read_u8(r) != 0; break;
        case GGUF_TYPE_UINT64:  kv->value.u64 = read_u64(r); break;
        case GGUF_TYPE_INT64:   kv->value.i64 = (int64_t)read_u64(r); break;
        case GGUF_TYPE_FLOAT64: kv->value.f64 = read_f64(r); break;
        case GGUF_TYPE_STRING:  kv->value.str = read_string(r); break;
        case GGUF_TYPE_ARRAY: {
            kv->value.arr.elem_type = (gguf_type_t)read_u32(r);
            kv->value.arr.count     = read_u64(r);
            kv->value.arr.data      = r->base + r->pos;
            // Skip array data
            if (kv->value.arr.elem_type == GGUF_TYPE_STRING) {
                for (uint64_t i = 0; i < kv->value.arr.count; i++) {
                    read_string(r);  // advance past each string
                }
            } else {
                size_t elem_sz = gguf_value_type_size(kv->value.arr.elem_type);
                r->pos += kv->value.arr.count * elem_sz;
            }
            break;
        }
        default:
            fprintf(stderr, "gguf: unknown KV type %d\n", kv->type);
            return false;
    }
    return reader_has(r, 0);  // check we didn't overrun
}

// --- Compute tensor sizes ---

static size_t compute_tensor_size(const gguf_tensor_info_t *ti) {
    uint64_t n_elem = 1;
    for (uint32_t d = 0; d < ti->n_dims; d++) {
        n_elem *= ti->dims[d];
    }
    size_t ts = ggml_type_size(ti->type);
    size_t bs = ggml_block_size(ti->type);
    if (bs == 0) return 0;
    return (n_elem / bs) * ts;
}

// --- Public API ---

gguf_file_t *gguf_open(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        perror("gguf: open");
        return NULL;
    }

    struct stat st;
    if (fstat(fd, &st) < 0) {
        perror("gguf: fstat");
        close(fd);
        return NULL;
    }

    size_t file_size = (size_t)st.st_size;
    if (file_size < 24) {
        fprintf(stderr, "gguf: file too small (%zu bytes)\n", file_size);
        close(fd);
        return NULL;
    }

    void *data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) {
        perror("gguf: mmap");
        close(fd);
        return NULL;
    }

    // Advise sequential access for header parsing
    madvise(data, file_size, MADV_SEQUENTIAL);

    reader_t r = { .base = (const uint8_t *)data, .pos = 0, .size = file_size };

    // Read header
    uint32_t magic = read_u32(&r);
    if (magic != GGUF_MAGIC) {
        fprintf(stderr, "gguf: bad magic 0x%08x (expected 0x%08x)\n", magic, GGUF_MAGIC);
        munmap(data, file_size);
        close(fd);
        return NULL;
    }

    uint32_t version   = read_u32(&r);
    uint64_t n_tensors = read_u64(&r);
    uint64_t n_kv      = read_u64(&r);

    if (version < 2 || version > 3) {
        fprintf(stderr, "gguf: unsupported version %u (need 2 or 3)\n", version);
        munmap(data, file_size);
        close(fd);
        return NULL;
    }

    gguf_file_t *gf = calloc(1, sizeof(gguf_file_t));
    gf->fd        = fd;
    gf->data      = data;
    gf->file_size = file_size;
    gf->version   = version;
    gf->n_tensors = n_tensors;
    gf->n_kv      = n_kv;

    // Parse metadata KV pairs
    gf->kv = calloc(n_kv, sizeof(gguf_kv_t));
    for (uint64_t i = 0; i < n_kv; i++) {
        gf->kv[i].key  = read_string(&r);
        gf->kv[i].type = (gguf_type_t)read_u32(&r);
        if (!read_kv_value(&r, &gf->kv[i])) {
            fprintf(stderr, "gguf: error parsing KV %llu\n", i);
            gguf_close(gf);
            return NULL;
        }
    }

    // Parse tensor info
    gf->tensors = calloc(n_tensors, sizeof(gguf_tensor_info_t));
    for (uint64_t i = 0; i < n_tensors; i++) {
        gguf_tensor_info_t *ti = &gf->tensors[i];
        ti->name   = read_string(&r);
        ti->n_dims = read_u32(&r);
        if (ti->n_dims > 4) {
            fprintf(stderr, "gguf: tensor %llu has %u dims (max 4)\n", i, ti->n_dims);
            gguf_close(gf);
            return NULL;
        }
        ti->n_elements = 1;
        for (uint32_t d = 0; d < ti->n_dims; d++) {
            ti->dims[d] = read_u64(&r);
            ti->n_elements *= ti->dims[d];
        }
        for (uint32_t d = ti->n_dims; d < 4; d++) {
            ti->dims[d] = 1;
        }
        ti->type   = (ggml_type_t)read_u32(&r);
        ti->offset = read_u64(&r);
        ti->size_bytes = compute_tensor_size(ti);
    }

    // Data section starts after header, aligned to 32 bytes (GGUF spec)
    size_t alignment = 32;
    const gguf_kv_t *align_kv = gguf_find_kv(gf, "general.alignment");
    if (align_kv && align_kv->type == GGUF_TYPE_UINT32) {
        alignment = align_kv->value.u32;
    }
    gf->data_offset = (r.pos + alignment - 1) & ~(alignment - 1);

    // Switch to random access for tensor data
    madvise(data, file_size, MADV_RANDOM);

    return gf;
}

void gguf_close(gguf_file_t *gf) {
    if (!gf) return;
    if (gf->data) munmap(gf->data, gf->file_size);
    if (gf->fd >= 0) close(gf->fd);
    free(gf->kv);
    free(gf->tensors);
    free(gf);
}

const void *gguf_tensor_data(const gguf_file_t *gf, const gguf_tensor_info_t *ti) {
    return (const uint8_t *)gf->data + gf->data_offset + ti->offset;
}

const gguf_tensor_info_t *gguf_find_tensor(const gguf_file_t *gf, const char *name) {
    size_t name_len = strlen(name);
    for (uint64_t i = 0; i < gf->n_tensors; i++) {
        if (gf->tensors[i].name.len == name_len &&
            memcmp(gf->tensors[i].name.str, name, name_len) == 0) {
            return &gf->tensors[i];
        }
    }
    return NULL;
}

const gguf_kv_t *gguf_find_kv(const gguf_file_t *gf, const char *key) {
    size_t key_len = strlen(key);
    for (uint64_t i = 0; i < gf->n_kv; i++) {
        if (gf->kv[i].key.len == key_len &&
            memcmp(gf->kv[i].key.str, key, key_len) == 0) {
            return &gf->kv[i];
        }
    }
    return NULL;
}

uint32_t gguf_get_u32(const gguf_file_t *gf, const char *key, uint32_t default_val) {
    const gguf_kv_t *kv = gguf_find_kv(gf, key);
    if (!kv) return default_val;
    if (kv->type == GGUF_TYPE_UINT32) return kv->value.u32;
    if (kv->type == GGUF_TYPE_INT32)  return (uint32_t)kv->value.i32;
    return default_val;
}

uint64_t gguf_get_u64(const gguf_file_t *gf, const char *key, uint64_t default_val) {
    const gguf_kv_t *kv = gguf_find_kv(gf, key);
    if (!kv) return default_val;
    if (kv->type == GGUF_TYPE_UINT64) return kv->value.u64;
    if (kv->type == GGUF_TYPE_UINT32) return kv->value.u32;
    return default_val;
}

float gguf_get_f32(const gguf_file_t *gf, const char *key, float default_val) {
    const gguf_kv_t *kv = gguf_find_kv(gf, key);
    if (!kv || kv->type != GGUF_TYPE_FLOAT32) return default_val;
    return kv->value.f32;
}

const char *gguf_get_str(const gguf_file_t *gf, const char *key) {
    const gguf_kv_t *kv = gguf_find_kv(gf, key);
    if (!kv || kv->type != GGUF_TYPE_STRING) return NULL;
    // Need to copy to null-terminated string (GGUF strings are length-prefixed)
    static char buf[4096];
    size_t len = kv->value.str.len < sizeof(buf)-1 ? kv->value.str.len : sizeof(buf)-1;
    memcpy(buf, kv->value.str.str, len);
    buf[len] = '\0';
    return buf;
}
