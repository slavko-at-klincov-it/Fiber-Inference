ANE_DIR = /Users/slavkoklincov/Code/ANE-Training
LIBANE_DIR = $(ANE_DIR)/libane
TRAINING_DIR = $(ANE_DIR)/training

CC = clang
CFLAGS = -O2 -fobjc-arc -Wall -Wno-deprecated-declarations
INCLUDES = -Iinclude -Isrc -I$(TRAINING_DIR)
# libane linked only when needed (Phase 2+). For now, just GGUF + Metal.
LDFLAGS_ANE = -L$(LIBANE_DIR) -lane -Wl,-rpath,$(LIBANE_DIR)
LDFLAGS = $(LDFLAGS_ANE)
FRAMEWORKS = -framework Foundation -framework Metal -framework MetalPerformanceShaders \
             -framework IOSurface -framework Accelerate

METAL_SDK = $(shell xcrun --sdk macosx --show-sdk-path)

SRC = src/main.m src/gguf.c src/model.m src/tokenizer.c src/sampler.c \
      src/kv_cache.m src/gpu_ffn.m src/ane_attn.m
TARGET = fiber-inference

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(INCLUDES) $(FRAMEWORKS) $(LDFLAGS) -o $@ $^
	install_name_tool -change libane.dylib @rpath/libane.dylib $@

clean:
	rm -f $(TARGET)

.PHONY: all clean
