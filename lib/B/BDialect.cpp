
#include "B/BDialect.h"
#include "B/BOps.h"

using namespace mlir;
using namespace mlir::b;

#include "B/BOpsDialect.cpp.inc"

void BDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "B/BOps.cpp.inc"
      >();
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
