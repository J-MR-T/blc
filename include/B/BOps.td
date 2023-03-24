#ifndef B_OPS
#define B_OPS

include "B/BDialect.td"

include "mlir/IR/OpBase.td"

include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/BuiltinOps.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinDialect.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

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

  // TODO add custom assembly format
  let hasCustomAssemblyFormat = 0;

  // let assemblyFormat = ...

  let builders = [
    OpBuilder<(ins "Builtin_UnrankedMemRef":$addr, "I64":$index, /* default sizespec to 8 using CArg */ CArg<"I8Attr", "8">:$sizespec), [{
      build($_builder, $_state, addr, index, sizespec);
    }]>
  ];
}


#endif
