#ifndef B_DIALECT_H
#define B_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "B/Dialect.h.inc"

#define GET_OP_CLASSES
#include "B/Ops.h.inc"

#endif
