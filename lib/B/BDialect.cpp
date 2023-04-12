
#include "B/BDialect.h"
#include "B/BOps.h"

using namespace mlir;
using namespace mlir::b;

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_TYPEDEF_CLASSES
#include "B/BOpsTypes.cpp.inc"

#include "B/BOpsDialect.cpp.inc"

void BDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "B/BOps.cpp.inc"
      >();


  addAttributes<
#define GET_ATTRDEF_LIST
#include "B/BOps.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "B/BOpsTypes.cpp.inc"
      >();
  // fallback, in case this thing above doesn't work
  //addTypes<mlir::b::PointerType>();
}

//void b::ConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, double value) {
  //auto dataType = RankedTensorType::get({}, builder.getF64Type());
  //auto dataAttribute = DenseElementsAttr::get(dataType, value);
  //b::ConstantOp::build(builder, state, dataType, dataAttribute);
//}

/*
mlir::Operation *BDialect::materializeConstant(mlir::OpBuilder &builder,
                                                 mlir::Attribute value,
                                                 mlir::Type type,
                                                 mlir::Location loc) {
    return builder.create<b::ConstantOp>(loc, type,
                                      value.cast<mlir::DenseElementsAttr>());
}
*/
