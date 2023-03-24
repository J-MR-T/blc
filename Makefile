#LLVM_BUILD_DIR=~/programming/Libs/Cpp/clang+llvm-15.0.2-x86_64-unknown-linux-gnu
# TODO make this modular
LLVM_BUILD_DIR=~/programming/Libs/Cpp/llvm-project/buildSchlepptop

.phony: release debug makeCMakeBearable clean setup

release: setup
	[ ! -f build/isDebug ] || $(MAKE) clean && $(MAKE) setup
	touch build/isRelease
	$(MAKE) cmake_build_type=Release makeCMakeBearable

debug: setup
	[ ! -f build/isRelease ] || $(MAKE) clean && $(MAKE) setup
	touch build/isDebug
	$(MAKE) cmake_build_type=Debug makeCMakeBearable

makeCMakeBearable: setup
	cd build                                                                                                                                && \
	cmake .. -DCMAKE_BUILD_TYPE=$(cmake_build_type) -DLLVM_DIR=$(LLVM_BUILD_DIR)/lib/cmake/llvm -DMLIR_DIR=$(LLVM_BUILD_DIR)/lib/cmake/mlir && \
	cmake --build .                                                                                                                         && \
	cd ..

setup:
	mkdir -p build

clean:
	rm -rf build
