// gguf.h — GGUF v3 parser with mmap support
// Reads model metadata and tensor index from GGUF files.
// All tensor data stays mmap'd (zero-copy, OS manages paging).

#ifndef GGUF_H
#define GGUF_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#define GGUF_MAGIC 0x46554747  // "GGUF" as little-endian uint32

// GGUF value types
typedef enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
} gguf_type_t;

// GGUF quantization types (subset we care about)
typedef enum {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
    GGML_TYPE_I64     = 27,
    GGML_TYPE_F64     = 28,
    GGML_TYPE_IQ1_M   = 29,
    GGML_TYPE_BF16    = 30,
} ggml_type_t;

// String stored in GGUF
typedef struct {
    uint64_t len;
    const char *str;  // points into mmap'd data (NOT null-terminated)
} gguf_string_t;

// Metadata key-value pair
typedef struct {
    gguf_string_t key;
    gguf_type_t   type;
    union {
        uint8_t   u8;
        int8_t    i8;
        uint16_t  u16;
        int16_t   i16;
        uint32_t  u32;
        int32_t   i32;
        float     f32;
        bool      b;
        uint64_t  u64;
        int64_t   i64;
        double    f64;
        gguf_string_t str;
        struct {
            gguf_type_t elem_type;
            uint64_t    count;
            const void *data;
        } arr;
    } value;
} gguf_kv_t;

// Tensor info from the GGUF header
typedef struct {
    gguf_string_t name;
    uint32_t  n_dims;
    uint64_t  dims[4];
    ggml_type_t type;
    uint64_t  offset;   // relative to data section start
    // computed
    size_t    size_bytes;
    uint64_t  n_elements;
} gguf_tensor_info_t;

// Parsed GGUF file
typedef struct {
    // File mapping
    int          fd;
    void        *data;       // mmap base
    size_t       file_size;

    // Header
    uint32_t     version;
    uint64_t     n_tensors;
    uint64_t     n_kv;

    // Metadata
    gguf_kv_t   *kv;         // array of n_kv entries

    // Tensor index
    gguf_tensor_info_t *tensors;  // array of n_tensors entries

    // Data section
    size_t       data_offset; // absolute file offset where tensor data begins
} gguf_file_t;

// Open and parse a GGUF file (mmap'd, read-only)
// Returns NULL on error, prints to stderr.
gguf_file_t *gguf_open(const char *path);

// Close and unmap
void gguf_close(gguf_file_t *gf);

// Get pointer to tensor data (into mmap'd region)
const void *gguf_tensor_data(const gguf_file_t *gf, const gguf_tensor_info_t *ti);

// Find tensor by name (linear scan, OK for init)
const gguf_tensor_info_t *gguf_find_tensor(const gguf_file_t *gf, const char *name);

// Find metadata value by key
const gguf_kv_t *gguf_find_kv(const gguf_file_t *gf, const char *key);

// Convenience: get uint32 metadata value, returns default_val if not found
uint32_t gguf_get_u32(const gguf_file_t *gf, const char *key, uint32_t default_val);
uint64_t gguf_get_u64(const gguf_file_t *gf, const char *key, uint64_t default_val);
float    gguf_get_f32(const gguf_file_t *gf, const char *key, float default_val);
const char *gguf_get_str(const gguf_file_t *gf, const char *key);

// Type info helpers
size_t      ggml_type_size(ggml_type_t type);     // bytes per block
size_t      ggml_block_size(ggml_type_t type);     // elements per block
const char *ggml_type_name(ggml_type_t type);
const char *gguf_type_name(gguf_type_t type);

#endif // GGUF_H
