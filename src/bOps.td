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
