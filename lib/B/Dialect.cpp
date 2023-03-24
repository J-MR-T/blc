#include "B/Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace b;

#include "B/Dialect.cpp.inc"

void BDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "B/Ops.cpp.inc"
      >();
}

