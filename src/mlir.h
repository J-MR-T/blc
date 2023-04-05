#pragma once
#include <mlir/IR/BuiltinOps.h>

#include "frontend.h"

namespace Codegen::MLIR{
    extern bool warningsGenerated;
	extern mlir::ModuleOp mod;

	bool generate(AST& ast) noexcept;
}
