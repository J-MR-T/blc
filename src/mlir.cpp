#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>
#include <mlir/Dialect/Traits.h>
#include <mlir/IR/MLIRContext.h>
// module op
//#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Builders.h>

#include "frontend.h"


/*
	NRoot,          // mlir equivalent: n/a
	NFunction,      // mlir equivalent: func. ???
	NParamList,     // mlir equivalent: func. ???
	NStmtDecl,      // mlir equivalent: b.alloca/ssa constr.
	NStmtReturn,    // mlir equivalent: func.return (but not in scf range, needs range)
	NStmtBlock,     // mlir equivalent: scf if/while ranges
	NStmtWhile,     // mlir equivalent: scf.condition
	NStmtIf,        // mlir equivalent: scf.condition
	NExprVar,       // mlir equivalent: b.ptr/ssa constr.
	NExprNum,       // mlir equivalent: ???.constant (or attributes?)
	NExprCall,      // mlir equivalent: func.call
	NExprUnOp,      // mlir equivalents:
					// - negate/minus: arith.subi
					// - bitwise not/tilde: arith.xori/
					// - logical not: arith.cmpi 0
					// - addrof: b.ptrtoint(b.ptr)
	NExprBinOp,     // mlir equivalents:
					// - add: arith.addi
					// - sub: arith.subi
					// - mul: arith.muli
					// - div: arith.divi
					// - mod: arith.remsi
					// - lshift: arith.shli
					// - rshift: arith.shrsi
					// - comparisons: arith.cmpi
					// - land: arith.andi (should be the same as log. on i1s)
					// - lor: arith.ori (should be the same as log. on i1s)
					// - bitwise and: arith.xori
					// - bitwise or: arith.ori
	NExprSubscript, // mlir equivalent: b.load/b.store
*/

namespace Codegen::MLIR {
	mlir::ModuleOp mod;
    // TODO add locations
    mlir::Location loc = mlir::UnknownLoc::get(mod.getContext());
    mlir::Type i64 = mlir::IntegerType::get(mod.getContext(), 64);

	mlir::Value mlirGenFunction(ASTNode& fnNode){
        mlir::OpBuilder builder(mod.getContext());

        auto& paramListNode = fnNode.children[0];
        auto& bodyNode = fnNode.children[1];
        builder.getFunctionType(std::vector(bodyNode.children.size(), i64), i64);

        // TODO parameters
        

        return mlir::Value(); // TODO fix
	}

};
