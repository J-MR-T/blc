# standard cpp makefile

CXXC=g++
CFLAGS=-Wall -Wextra -Wpedantic -O3 -std=c++2b -fno-rtti -lz

LLVM_CONFIG=llvm-config

OUT=blc

LLVM_CFLAGS=$(shell $(LLVM_CONFIG) --cppflags --ldflags --libs --system-libs --libs all) -DLLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING=1 

ALL_FLAGS=$(CFLAGS) $(LLVM_CFLAGS)

SOURCES=$(wildcard *.cpp)

.PHONY: all debug clean
all: $(SOURCES)
	$(CXXC) $^ $(ALL_FLAGS) -DNDEBUG -o $(OUT)

debug: $(SOURCES)
	$(CXXC) $^ $(ALL_FLAGS) -fsanitize=address -fsanitize=undefined -fsanitize=leak -O0 -g -o $(OUT)

test: debug
	lit -j1 -sv .

clean:
	rm $(OUT)
