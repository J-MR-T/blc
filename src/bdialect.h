/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#include <mlir/IR/Dialect.h>

namespace b {

class BDialect : public ::mlir::Dialect {
  explicit BDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~BDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("b");
  }
};
} // namespace b
MLIR_DECLARE_EXPLICIT_TYPE_ID(b::BDialect)
