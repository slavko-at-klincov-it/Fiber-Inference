ANE_DIR = /Users/slavkoklincov/Code/ANE-Training
LIBANE_DIR = $(ANE_DIR)/libane

CC = clang
CFLAGS = -O2 -fobjc-arc -Wall -Wno-deprecated-declarations
INCLUDES = -I../include -I../src -I$(ANE_DIR)/training
LDFLAGS = -L$(LIBANE_DIR) -lane -Wl,-rpath,$(LIBANE_DIR)
FRAMEWORKS = -framework Foundation -framework Metal -framework MetalPerformanceShaders \
             -framework IOSurface -framework Accelerate

all: bench-sweep

bench-sweep: sweep.m
	$(CC) $(CFLAGS) $(INCLUDES) $(FRAMEWORKS) $(LDFLAGS) -o $@ $^
	install_name_tool -change libane.dylib @rpath/libane.dylib $@

clean:
	rm -f bench-sweep

.PHONY: all clean
