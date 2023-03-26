#ifndef B_OPS
#define B_OPS

include "B/BDialect.td"

include "mlir/IR/OpBase.td"

include "mlir/IR/BuiltinTypes.td"
// NEVER INCLUDE THIS: "mlir/IR/BuiltinOps.td". Breaks everything for some reason
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinDialect.td"

include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def ConstantOp : B_Op<"constant", [Pure]> {
  // Provide a summary and description for this operation. This can be used to
  // auto-generate documentation of the operations within our dialect.
  let summary = "constant";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>
                        : tensor<2x3xf64>
    ```
  }];

  // The constant operation takes an attribute as the only input.
  let arguments = (ins Builtin_UnrankedMemRef:$value);

  // The constant operation returns a single value of TensorType.
  let results = (outs Builtin_UnrankedMemRef);

  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 1;

  // Add custom build methods for the constant operation. These method populates
  // the `state` that MLIR uses to create operations, i.e. these are used when
  // using `builder.create<ConstantOp>(...)`.
  let builders = [
    // Build a constant with a given constant tensor value.
    OpBuilder<(ins "UnrankedMemRefType":$value), [{
      build($_builder, $_state, value);
    }]>,

    // Build a constant with a given constant floating-point value.
    OpBuilder<(ins "double":$value)>
  ];

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;
}

/* Wow, tablegen actually supports nested multiline comments.

def SubscriptOp : B_Op<"subscript", [Pure /* no real sideeffects -> may be removed if dead */]> {
  let summary = "operation to access a fixed width amount of memory at an offset from an address";

  let description = [{
    This operation is used to access a fixed width amount of memory at an offset from an address.
    The fixed width is represented by the type of 
    The offset is specified by the index and the size of the memory is specified by the sizespec.
    The sizespec is either 1, 2, 4 or 8 and specifies the size of the memory to access in bytes. 
    The result is the value of the memory of the specified size at the specified offset from the address.
  }];

  let arguments = (ins Builtin_UnrankedMemRef:$addr, I64:$index, I8Attr:$sizespec);

  let results = (outs I64:$result);

  // TODO add a verifier
  let hasVerifier = 0;

  let hasRegionVerifier = 0;

  // TODO add custom assembly format
  let hasCustomAssemblyFormat = 0;

  // let assemblyFormat = ...

  let builders = [
    OpBuilder<(ins "UnrankedMemRefType":$addr, "IntegerType":$index, /* default sizespec to 8 using CArg */ CArg<"IntegerAttr", "8">:$sizespec), [{
      //build($_builder, $_state, addr, index, sizespec);
    }]>
  ];
}

*/


#endif
