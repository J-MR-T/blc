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

// TODO if there was pointer arithmetic, then ptrtoint(inttoptr(x)) -> x would not be a valid transformation (well, depends, but it would change provenance), but I think it's fine in this case, because there isn't really provenance anyway

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

  let arguments = (ins I8Attr:$width);

  let results = (outs B_PointerType:$result);

  let builders = [
    OpBuilder<(ins CArg<"uint8_t", "8">:$width /* default to 8 bytes, because our lang basically only needs that */ ), [{
      build($_builder, $_state, PointerType::get($_builder.getContext()), width);
    }]>
  ];
}

def PtrToIntOp : B_Op<"ptrtoint", [Pure]>{
  let summary = "operation to convert a pointer to an integer";

  let description = [{
    This operation is used to convert a pointer to an integer.
    The result is the integer representation of the pointer.
  }];

  let arguments = (ins B_PointerType:$ptr);

  let results = (outs I64:$result);

  let hasCanonicalizer = 1; // canonicalizes ptrtoint(inttoptr(x)) -> x
}

def IntToPtrOp : B_Op<"inttoptr", [Pure]>{
  let summary = "operation to convert an integer to a pointer";

  let description = [{
    This operation is used to convert an integer to a pointer.
    The result is the pointer representation of the integer.
  }];

  let arguments = (ins I64:$intt);

  let results = (outs B_PointerType:$result);

  let hasCanonicalizer = 1; // canonicalizes inttoptr(ptrtoint(x)) -> x
}

def StoreOp : B_Op<"store">{
  let summary = "operation to store a value to a memory location";

  let description = [{
    This operation is used to store a width byte value to a memory location.
    The value is stored at the address specified by the pointer.
  }];

  let arguments = (ins B_PointerType:$ptr, I64:$value, I8Attr:$width);

  let results = (outs);
}

def LoadOp : B_Op<"load", [Pure]>{
  let summary = "operation to load a value from a memory location";

  let description = [{
    This operation is used to load a width byte value from a memory location.
    The value is loaded from the address specified by the pointer.
  }];

  let arguments = (ins B_PointerType:$ptr, I8Attr:$width);

  let results = (outs I64:$result);
}

#endif
