#ifndef B_DIALECT
#define B_DIALECT

// important: This file is not used directly, only by being included in the BOps file!
// that explains why "BOpsDialect.[h/cpp].inc" needs to be used elsewhere, because this file content only *really* exists in BOps.

// include stuff for tablegen
include "mlir/IR/OpBase.td"

def B_Dialect : Dialect {
  let name = "b";

  let summary = "An example dialect for the B language";

  let description = [{
      An example dialect for the B language
  }];

  let cppNamespace = "::mlir::b";

  let useDefaultTypePrinterParser = 1;
}

class B_Op<string mnemonic, list<Trait> traits = []> :
        Op<B_Dialect, mnemonic, traits>;

#endif
