#pragma once
#include <mlir/IR/BuiltinOps.h>

#include "frontend.h"

namespace Codegen::MLIR{
    /// yes im going this far to avoid the cpp class decl/impl nonsense
    std::tuple<bool /* failed? */, bool /* warnings generated? */, mlir::OwningOpRef<mlir::ModuleOp>> generate(mlir::MLIRContext&, AST&) noexcept;

    void canonicalizerTest() noexcept;
    bool runCanonicalizer(mlir::ModuleOp mod) noexcept;

    mlir::LogicalResult lowerToLLVM(mlir::ModuleOp mod) noexcept;
    void workaroundAutomaticFreeMallocdecls(mlir::ModuleOp mod) noexcept;
}
