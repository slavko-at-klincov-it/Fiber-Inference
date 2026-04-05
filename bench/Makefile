ANE_DIR ?= ../../ANE-Training
LIBANE_DIR = $(ANE_DIR)/libane

CC = clang
CFLAGS = -O2 -fobjc-arc -Wall -Wno-deprecated-declarations
INCLUDES = -I../include -I../src -I$(ANE_DIR)/training
LDFLAGS = -L$(LIBANE_DIR) -lane -Wl,-rpath,$(LIBANE_DIR)
FRAMEWORKS = -framework Foundation -framework Metal -framework MetalPerformanceShaders \
             -framework IOSurface -framework Accelerate

all: bench-sweep bench-limits bench-large

bench-sweep: sweep.m
	$(CC) $(CFLAGS) $(INCLUDES) $(FRAMEWORKS) $(LDFLAGS) -o $@ $^
	install_name_tool -change libane.dylib @rpath/libane.dylib $@

bench-limits: limits.m
	$(CC) $(CFLAGS) $(INCLUDES) $(FRAMEWORKS) $(LDFLAGS) -o $@ $^
	install_name_tool -change libane.dylib @rpath/libane.dylib $@

bench-large: large_ane.m
	$(CC) $(CFLAGS) $(INCLUDES) $(FRAMEWORKS) $(LDFLAGS) -o $@ $^
	install_name_tool -change libane.dylib @rpath/libane.dylib $@

clean:
	rm -f bench-sweep bench-limits bench-large

.PHONY: all clean
