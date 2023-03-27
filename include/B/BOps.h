#ifndef B_OPS_H
#define B_OPS_H

#include "mlir/IR/OpDefinition.h"

#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/IR/Builders.h"

#include "mlir/IR/BuiltinTypes.h"
//#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
//#include "mlir/IR/BuiltinTypeInterfaces.h"

#include "B/BTypes.h"

#define GET_OP_CLASSES
#include "B/BOps.h.inc"


#endif
