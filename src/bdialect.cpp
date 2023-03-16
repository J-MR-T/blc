#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include "bdialect.h"
#include "frontend.h"

namespace b{

    class SubscriptOp;

    void BDialect::initialize(){
        addOperations<SubscriptOp>();
    }

    class SubscriptOp : public mlir::Op<SubscriptOp, 
        mlir::OpTrait::NOperands<3>::Impl,
        mlir::OpTrait::OneResult,
        mlir::OpTrait::OneTypedResult<mlir::IntegerType>::Impl
    >{
         public:
         /// Inherit the constructors from the base Op class.
         using Op::Op;

         /// Provide the unique name for this operation. MLIR will use this to register
         /// the operation and uniquely identify it throughout the system. The name
         /// provided here must be prefixed by the parent dialect namespace followed
         /// by a `.`.
         static llvm::StringRef getOperationName() { return "b.subscript"; }


         
         /// Return the value of the subscript?
         /// TODO
         mlir::DenseElementsAttr getValue();

         
         /// Operations may provide additional verification beyond what the attached
         /// traits provide.
         mlir::LogicalResult verifyInvariants();

         
         /// Provide an interface to build this operation from a set of input values.
         /// This interface is used by the `builder` classes to allow for easily
         /// generating instances of this operation:
         ///   mlir::OpBuilder::create<SubscriptOp>(...)
         /// This method populates the given `state` that MLIR uses to create
         /// operations. This state is a collection of all of the discrete elements
         /// that an operation may contain.
         /// TODO modify so that they fit
         /// Build a subscript with the given return type and `value` attribute.
         static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                 mlir::Type result, mlir::DenseElementsAttr value);
         /// Build a constant and reuse the type from the given 'value'.
         static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                 mlir::DenseElementsAttr value);
         /// Build a constant by broadcasting the given 'value'.
         static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                 double value);

    };

}
