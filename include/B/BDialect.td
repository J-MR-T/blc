#ifndef B_DIALECT
#define B_DIALECT

// include stuff for tablegen
include "mlir/IR/OpBase.td"

def B_Dialect : Dialect {
    let name = "b";

    let summary = "An example dialect for the B language";

    let description = [{
        aaa
    }];

    let cppNamespace = "b";
}

class B_Op<string mnemonic, list<Trait> traits = []> :
        Op<B_Dialect, mnemonic, traits>;

#endif
