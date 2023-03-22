#LLVM_BUILD_DIR=~/programming/Libs/Cpp/clang+llvm-15.0.2-x86_64-unknown-linux-gnu
# TODO make this modular
LLVM_BUILD_DIR=~/programming/Libs/Cpp/llvm-project/buildSchlepptop

.phony: makeCMakeBearable clean

makeCMakeBearable:
	mkdir -p build
	cd build                                                                                         && \
	cmake .. -DLLVM_DIR=$(LLVM_BUILD_DIR)/lib/cmake/llvm -DMLIR_DIR=$(LLVM_BUILD_DIR)/lib/cmake/mlir && \
	cmake --build .                                                                                  && \
	cd ..

clean:
	rm -rf build
