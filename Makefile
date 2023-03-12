# standard cpp makefile

CXX=g++
OUT=blc

LLVM_CONFIG=llvm-config
LLVM_FLAGS=$(shell $(LLVM_CONFIG) --cppflags --ldflags --system-libs --libs all) -DLLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING=1 -lLLVM-15

CXXFLAGS=$(LLVM_FLAGS) -Wall -Wextra -Wpedantic -O3 -std=c++2b -fno-rtti -lz
DEBUGFLAGS=-fsanitize=address -fsanitize=undefined -fsanitize=leak -O0 -g

SOURCES=$(wildcard src/*.cpp)
OBJECTS=$(SOURCES:src/%.cpp=build/%.o)

.PHONY: all debug clean

all: setup $(SOURCES)
	export CXXFLAGS="$(CXXFLAGS) -DNDEBUG"
	$(MAKE) $(OUT)

debug: setup $(SOURCES)
	export CXXFLAGS="$(CXXFLAGS) $(DEBUGFLAGS)"
	$(MAKE) $(OUT)

setup:
	mkdir -p build

build/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# link
$(OUT): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@

test: debug
	lit -j1 -sv .

clean:
	rm -rf build
	rm -f $(OUT)
