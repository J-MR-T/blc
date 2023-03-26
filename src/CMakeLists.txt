get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

#llvm_map_components_to_libnames(llvm_libs support core irreader object native target)
#llvm_map_components_to_libnames(llvm_libs aarch64asmparser aarch64codegen aarch64desc  aarch64info aarch64utils aggressiveinstcombine all amdgpuasmparser amdgpucodegen amdgpudesc  amdgpuinfo  amdgpuutils analysis armasmparser armcodegen armdesc  arminfo armutils asmparser asmprinter avrasmparser avrcodegen avrdesc  avrinfo binaryformat bitreader bitstreamreader bitwriter bpfasmparser bpfcodegen bpfdesc  bpfinfo cfguard codegen core coroutines  debuginfocodeview debuginfodwarf debuginfomsf debuginfopdb demangle engine executionengine hexagonasmparser hexagoncodegen hexagondesc  hexagoninfo instcombine instrumentation irreader lanaiasmparser lanaicodegen lanaidesc  lanaiinfo mipsasmparser mipscodegen mipsdesc  mipsinfo mirparser msp430asmparser msp430codegen msp430desc  msp430info native nativecodegen nvptxcodegen nvptxdesc nvptxinfo object option powerpcasmparser powerpccodegen powerpcdesc  powerpcinfo riscvasmparser riscvcodegen riscvdesc  riscvinfo sparcasmparser sparccodegen sparcdesc  sparcinfo support symbolize systemzasmparser systemzcodegen systemzdesc  systemzinfo tablegen target textapi transformutils veasmparser vecodegen vectorize vedesc  veinfo webassemblyasmparser webassemblycodegen webassemblydesc  webassemblyinfo webassemblyutils windowsdriver x86asmparser x86codegen x86desc  x86info xcorecodegen xcoredesc  xcoreinfo)

# i would prefer this, if I knew how to include llvm-config in the original LLVM build...
#execute_process(COMMAND ~/programming/Libs/Cpp/llvm-project/buildSchlepptop/bin/llvm-config --components OUTPUT_VARIABLE llvm_components)
#llvm_map_components_to_libnames(llvm_libs ${llvm_components})
#execute_process(COMMAND ~/programming/Libs/Cpp/llvm-project/buildSchlepptop/bin/llvm-config --libs all OUTPUT_VARIABLE llvm_libs)
message(STATUS "LLVM libs: ${llvm_libs}")

set(LIBS
		${llvm_libs}
        ${dialect_libs}
        ${conversion_libs}
		MLIRB

        MLIRAnalysis
        MLIRCallInterfaces
        MLIRCastInterfaces
        MLIRExecutionEngine
        MLIRIR
        MLIRLLVMCommonConversion
        MLIRLLVMToLLVMIRTranslation
        MLIRMemRefDialect
        MLIRLLVMDialect
        MLIRParser
        MLIRPass
        MLIRSideEffectInterfaces
        MLIRSupport
        MLIRTargetLLVMIRExport
        MLIRTransforms
        MLIROptLib
        rt
        dl
        m
        z
        zstd
        tinfo
        xml2
        )

# this solves the linker errors with the target specific initialize methods
set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  )
message(STATUS "LLVM link components: ${LLVM_LINK_COMPONENTS}")
#
add_llvm_executable(blc main.cpp frontend.cpp util.cpp mlir.cpp)

llvm_update_compile_flags(blc)
target_link_libraries(blc PRIVATE ${LIBS})

target_link_directories(blc PUBLIC /lib)

target_compile_options(blc PUBLIC -fexceptions)

mlir_check_all_link_libraries(blc)
