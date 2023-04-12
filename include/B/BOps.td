#ifndef B_OPS
#define B_OPS

include "B/BDialect.td"

include "mlir/IR/AttrTypeBase.td"

include "mlir/IR/OpBase.td"

include "mlir/IR/BuiltinTypes.td"
// don't include this: "mlir/IR/BuiltinOps.td". Breaks everything for some reason
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinDialect.td"

include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
// this also seems to break everything:
// ah it only broke everything because the corresponding header wasn't included in BOps.h, done that now, so this should work and define some nice builders
include "mlir/Interfaces/InferTypeOpInterface.td"

include "mlir/Interfaces/SideEffectInterfaces.td"

// === types ===

// base type
class B_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<B_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def B_PointerType : B_Type<"Pointer", "ptr"> {
  let summary = "pointer type";
  let description = "A pointer type is a type that points to a memory location";

}

// === ops ===

// TODO add verifiers (hasVerifier = 1; impl) and assembly formats (hasCustomAssemblyFormat = 1; impl)

// alloca for local variables
def AllocaOp : B_Op<"alloca">{
  let summary = "operation to allocate a fixed width amount of memory on the stack";

  let description = [{
    This operation is used to allocate a fixed width amount of memory on the stack.
    The result is the address of the allocated memory as a b.ptr.
  }];

  let arguments = (ins I8Attr:$size);

  let results = (outs B_PointerType:$result);

  let builders = [
    OpBuilder<(ins CArg<"uint8_t", "8">:$size /* default to 8 bytes, because our lang basically only needs that */ ), [{
      build($_builder, $_state, PointerType::get($_builder.getContext()), size);
    }]>
  ];
}

def PtrToIntOp : B_Op<"ptrtoint">{
  let summary = "operation to convert a pointer to an integer";

  let description = [{
    This operation is used to convert a pointer to an integer.
    The result is the integer representation of the pointer.
  }];

  let arguments = (ins B_PointerType:$ptr);

  let results = (outs I64:$result);
}

def IntToPtrOp : B_Op<"inttoptr">{
  let summary = "operation to convert an integer to a pointer";

  let description = [{
    This operation is used to convert an integer to a pointer.
    The result is the pointer representation of the integer.
  }];

  let arguments = (ins I64:$intt);

  let results = (outs B_PointerType:$result);
}

def StoreOp : B_Op<"store">{
  let summary = "operation to store a value to a memory location";

  let description = [{
    This operation is used to store a 64 bit value to a memory location.
    The value is stored at the address specified by the pointer.
  }];

  let arguments = (ins B_PointerType:$ptr, I64:$value);

  let results = (outs);
}

def LoadOp : B_Op<"load">{
  let summary = "operation to load a value from a memory location";

  let description = [{
    This operation is used to load a 64 bit value from a memory location.
    The value is loaded from the address specified by the pointer.
  }];

  let arguments = (ins B_PointerType:$ptr);

  let results = (outs I64:$result);
}

#endif
