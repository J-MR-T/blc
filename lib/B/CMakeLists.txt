add_mlir_dialect_library(MLIRB
		BDialect.cpp
		BOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/B

		DEPENDS
		MLIRBOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
        MLIRInferTypeOpInterface
	)
