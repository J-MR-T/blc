# standard cpp makefile

CPPC=g++
CFLAGS=-Wall -Wextra -Wpedantic -O3 -std=c++2b -fno-rtti -lz

LLVM_CONFIG=llvm-config

OUT=main

LLVM_CFLAGS=$(shell $(LLVM_CONFIG) --cppflags --ldflags --libs) -DLLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING=1 
.PHONY: all debug clean
all: main.cpp
	$(CPPC) $^ $(CFLAGS) $(LLVM_CFLAGS) -DNDEBUG -o $(OUT)

debug: main.cpp
	$(CPPC) $^ $(CFLAGS) $(LLVM_CFLAGS) -O0 -g -o $(OUT)

clean:
	rm $(OUT)
