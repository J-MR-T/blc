#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/ConstantRange.h>
#include <llvm/ADT/STLExtras.h>

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>
#include <mlir/Dialect/Traits.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/MLIRContext.h>
// module op
//#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Dialect.h>

#include "B/BDialect.h"
#include "B/BOps.h"
#include "mlir.h"
#include "util.h"


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
    bool warningsGenerated{false};

    void warn(printable auto... args){
        if(!ArgParse::args.nowarn()){
            llvm::errs() << "Warning: ";
            (llvm::errs() << ... << args) << "\n";
            warningsGenerated = true;
        }
    }

    void warn(const std::string& msg, mlir::Value value = {}){
        if(!ArgParse::args.nowarn()){
            llvm::errs() << "Warning: " << msg;
            if(value)
                llvm::errs() << " at " << value;
            llvm::errs() << "\n";
            warningsGenerated = true;
        }
    }

    mlir::MLIRContext ctx;
    mlir::ModuleOp mod;

    // TODO add locations
    mlir::Location loc = mlir::UnknownLoc::get(&ctx);
    mlir::Type i64 = mlir::IntegerType::get(&ctx, 64);
    mlir::func::FuncOp currentFn;

    struct BasicBlockInfo{
        bool sealed{false};
        llvm::DenseMap<uint64_t /* variable uid */, mlir::Value> regVarmap{};
    };

    llvm::DenseMap<uint64_t /* variable uid */, mlir::b::AllocaOp> autoVarmap{};

    llvm::DenseMap<mlir::Block*, BasicBlockInfo> blockInfo{};
    llvm::DenseMap<mlir::Block*, ASTNode*> blockArgsToResolve{};

    // REFACTOR: cache the result in a hash map
    inline mlir::func::FuncOp findFunction(llvm::StringRef name){
		mlir::func::FuncOp fn;
		mod.walk([&](mlir::func::FuncOp func){
			if(func.getName() == name) fn = func;
		});

		return fn;
    }

	mlir::Value varmapLookup(mlir::Block* block, ASTNode& node){
		if(node.ident.type == IdentifierInfo::AUTO){
			assert(autoVarmap.find(node.ident.uID) != autoVarmap.end() && "auto var not found in varmap");
			return autoVarmap[node.ident.uID];
		}else{
            assert(node.ident.type == IdentifierInfo::REGISTER && "variable can only be auto or register");
            assert(blockInfo.find(block) != blockInfo.end() && "block not found in blockInfo");
			auto& [sealed, regVarmap] = blockInfo[block];

			// try to find the variable in the current block
			auto it = regVarmap.find(node.ident.uID);
			if(it != regVarmap.end())
				return it->second;

			if(sealed){
				// if the block is sealed, we know all the predecessors, so if there is a single one, query that
				// if there are multiple ones, make a phi node/block arg and fill it now

				if(auto pred = block->getSinglePredecessor()){
					return varmapLookup(pred, node);
                }else{
                    auto blockArg = block->addArgument(i64, loc);

                    regVarmap[node.ident.uID] = blockArg;

                    for(auto pred:  block->getPredecessors()){
						auto term = pred->getTerminator();
                        DEBUGLOG("term: " << term << " bb: " << pred << " block: " << block << " blockArg: " << blockArg)
                        term->setOperand(blockArg.getArgNumber(), varmapLookup(pred, node));
                    }

                    return blockArg;
                }
			}else{
				// if it isn't, we *have* to have a phi node/block arg, and fill it later
                auto blockArg = block->addArgument(i64, loc);

                blockArgsToResolve[block] = &node;

                return regVarmap[node.ident.uID] = blockArg;
			}
		}
	}

    /// just for convenience
    /// can *only* be called with register vars, as auto vars need to be looked up in the alloca map
    inline mlir::Value setRegisterVar(mlir::Block* block, ASTNode& node, mlir::Value val) noexcept{
        assert(node.ident.type == IdentifierInfo::REGISTER && "can only update register vars with this method");

        auto& [sealed, varmap] = blockInfo[block];
        return varmap[node.ident.uID] = val;
    }

    inline mlir::Value setRegisterVar(mlir::OpBuilder& builder, ASTNode& node, mlir::Value val) noexcept{
        return setRegisterVar(builder.getInsertionBlock(), node, val);
    }

	// TODO well, take care of this annoying scf.yield/return business

	void genStmts(ASTNode& stmtNode, mlir::OpBuilder& builder) noexcept;
	mlir::Value genExpr(ASTNode& exprNode, mlir::OpBuilder& builder) noexcept;

	void genStmt(ASTNode& stmtNode, mlir::OpBuilder& builder) noexcept{
        switch(stmtNode.type){
        case ASTNode::Type::NStmtDecl: // always contains initializer
            {
            auto initializerNode = stmtNode.children[0];
            auto initializer = genExpr(initializerNode, builder);
            if(stmtNode.ident.type == IdentifierInfo::AUTO){
                auto insertPoint = builder.getInsertionPoint();
                auto insertBlock = builder.getInsertionBlock();
                builder.setInsertionPointToStart(&currentFn.getBody().getBlocks().front());

                // create alloca at start of entry block
                auto alloca = autoVarmap[stmtNode.ident.uID] = builder.create<mlir::b::AllocaOp>(loc, 8);

				builder.create<mlir::b::StoreOp>(loc, alloca, initializer);

                // reset insertion point
                builder.setInsertionPoint(insertBlock, insertPoint);
            }else{
                setRegisterVar(builder, stmtNode, initializer);
            }
            break;
            }
        case ASTNode::Type::NStmtReturn:
        case ASTNode::Type::NStmtBlock:
        case ASTNode::Type::NStmtWhile:
        case ASTNode::Type::NStmtIf:
            break;
            // TODO

        case ASTNode::Type::NExprVar:
        case ASTNode::Type::NExprNum:
        case ASTNode::Type::NExprCall:
        case ASTNode::Type::NExprUnOp:
        case ASTNode::Type::NExprBinOp:
        case ASTNode::Type::NExprSubscript:
            genExpr(stmtNode, builder);
            break;

        case ASTNode::Type::NRoot:
        case ASTNode::Type::NFunction:
        case ASTNode::Type::NParamList:
            assert(false);
            break;
        }
    }

	void genStmts(ASTNode& blockNode, mlir::OpBuilder& builder) noexcept{
		for(auto& stmtNode : blockNode.children){
			genStmt(stmtNode, builder);
            if(stmtNode.type == ASTNode::Type::NStmtReturn)
                // stop the generation for this block
                break;
		}
	}

	mlir::Value genExpr(ASTNode& exprNode, mlir::OpBuilder& builder) noexcept{
        using namespace mlir::arith;
        using Type = Token::Type;

		switch(exprNode.type){
			case ASTNode::Type::NExprVar:
				{
                auto value = varmapLookup(builder.getBlock(), exprNode);
                if(exprNode.ident.type == IdentifierInfo::REGISTER){
                    return value;
                }else{
                    assert(exprNode.ident.type == IdentifierInfo::AUTO && "variable can only be auto or register");
                    //return builder.create<mlir::b::LoadOp>(loc, value);
                }
				}
			case ASTNode::Type::NExprNum:
				return builder.create<mlir::arith::ConstantIntOp>(loc, exprNode.value, i64);
			case ASTNode::Type::NExprBinOp:
                {
				auto childOp1 = genExpr(exprNode.children[0], builder);
				auto childOp2 = genExpr(exprNode.children[1], builder);
				switch(exprNode.op){
					// sadly no nicer way to just change the Op type, can't just save a type as a variable and use a unified case :(
					// and using op names instead would mean they'd have to be parsed again (right?), that would be a waste

                    // bitwise
                    case Type::AMPERSAND:   return builder.create<AndIOp>(loc, std::move(childOp1), std::move(childOp2));
                    case Type::BITWISE_OR:  return builder.create<OrIOp> (loc, std::move(childOp1), std::move(childOp2));
                    case Type::BITWISE_XOR: return builder.create<XOrIOp>(loc, std::move(childOp1), std::move(childOp2));

                    // arithmetic
					case Type::PLUS:   return builder.create<AddIOp> (loc, std::move(childOp1), std::move(childOp2));
					case Type::MINUS:  return builder.create<SubIOp> (loc, std::move(childOp1), std::move(childOp2));
					case Type::TIMES:  return builder.create<MulIOp> (loc, std::move(childOp1), std::move(childOp2));
					case Type::DIV:    return builder.create<DivSIOp>(loc, std::move(childOp1), std::move(childOp2));
					case Type::MOD:    return builder.create<RemSIOp>(loc, std::move(childOp1), std::move(childOp2));
					case Type::SHIFTL: return builder.create<ShLIOp> (loc, std::move(childOp1), std::move(childOp2));
					case Type::SHIFTR: return builder.create<ShRSIOp>(loc, std::move(childOp1), std::move(childOp2));

				    // comparisons
					default:
                        CmpIPredicate pred;
                        switch(exprNode.op){
                            case Type::EQUAL:         pred = CmpIPredicate::eq;  break;
                            case Type::NOT_EQUAL:     pred = CmpIPredicate::ne;  break;
                            case Type::LESS:          pred = CmpIPredicate::slt; break;
                            case Type::GREATER:       pred = CmpIPredicate::sgt; break;
                            case Type::LESS_EQUAL:    pred = CmpIPredicate::sle; break;
                            case Type::GREATER_EQUAL: pred = CmpIPredicate::sge; break;
                            default:
                                assert(false);
                                break;
                        }
                        return builder.create<mlir::arith::CmpIOp>(loc, pred, std::move(childOp1), std::move(childOp2));
				}
                assert(false);
                }
			case ASTNode::Type::NExprUnOp:
                {
				auto childOp = genExpr(exprNode.children[0], builder);
                switch(exprNode.op){
                    case Type::MINUS: return builder.create<SubIOp>(loc, builder.create<ConstantIntOp>(loc,    0, i64), childOp);
                    case Type::TILDE: return builder.create<XOrIOp>(loc, builder.create<ConstantIntOp>(loc, -1ll, i64), childOp);
                    case Type::LOGICAL_NOT:
                        return builder.create<ExtUIOp>(loc, i64,
                            builder.create<CmpIOp>(loc, CmpIPredicate::eq, childOp, builder.create<ConstantIntOp>(loc, 0, i64))
                        );
                    default:
                        assert(false);
                        break;
                }
                assert(false);
                }
			case ASTNode::Type::NExprCall:
                {
                auto calledFn = exprNode.ident.name;

                // TODO I bet this doesn't work with functions which aren't declared yet
                auto callee = mod.lookupSymbol<mlir::func::FuncOp>(calledFn);
                assert(callee && "Function for call not found -> TODO: forward declarations");


                llvm::SmallVector<mlir::Value, 8> args(exprNode.children.size());
                for(unsigned int i = 0; i < exprNode.children.size(); ++i)
                    args[i] = genExpr(exprNode.children[i], builder);

                if(args.size() != callee->getNumOperands()){
                    // hw02.txt: "Everything else is handled as in ANSI C", hw04.txt: "note that parameters/arguments do not need to match"
                    // but: from the latest C11 standard draft: "the number of arguments shall agree with the number of parameters"
                    // (i just hope thats the same as in C89/ANSI C, can't find that standard anywhere online, Ritchie & Kernighan says this:
                    // "The effect of the call is undefined if the number of arguments disagrees with the number of parameters in the
                    // definition of the function", which is basically the same)
                    // so technically this is undefined behavior >:)
                    using namespace SemanticAnalysis;
                    if(auto it  = externalFunctionsToNumParams.find(exprNode.ident.name);
                            it == externalFunctionsToNumParams.end() ||
                                  externalFunctionsToNumParams[exprNode.ident.name] != EXTERNAL_FUNCTION_VARARGS){
                        // in this case, the function is either not found or not varargs, so something weird is going on
                        warn("Call to function ", exprNode.ident.name, " with ", args.size(), " arguments, but function has ", callee->getNumOperands(), " parameters");
                        return mlir::Value();
                    }
                }
                return builder.create<mlir::func::CallOp>(loc, callee, args).getResult(0);
                }
			case ASTNode::Type::NExprSubscript:
				// TODO
				break;
			default:
				break;
		}

        assert(false);
	}

	mlir::func::FuncOp genFunction(ASTNode& fnNode) noexcept{
        mlir::OpBuilder builder(mod.getContext());

        auto& paramListNode = fnNode.children[0];
        auto& bodyNode = fnNode.children[1];

		// parameters
        auto fnType = builder.getFunctionType(std::vector(bodyNode.children.size(), i64), i64);
		auto fn = builder.create<mlir::func::FuncOp>(loc, fnNode.ident.name, fnType);

		genStmts(bodyNode, builder);

        return fn;
	}

    bool generate(AST& ast) noexcept{
        // TODO this seems to be right, but generates linker errors
        //ctx.getOrLoadDialect<mlir::b::BDialect>();

		mlir::OpBuilder builder(&ctx);
        mod = mlir::ModuleOp::create(loc);


        ASTNode& root = ast.root;

        // declare implicitly declared functions
        for(auto& entry: SemanticAnalysis::externalFunctionsToNumParams){
            auto fnParamCount = entry.second;
            llvm::SmallVector<mlir::Type, 8> params;
            if(fnParamCount == EXTERNAL_FUNCTION_VARARGS){
				// TODO varargs unsupported for now because mlir doesn't seem to support them
				warn("varargs functions are not supported with the MLIR backend yet, stopping compilation");
				return false;
            }else{
				// TODO something is out of bounds here apparently
                params = llvm::SmallVector<mlir::Type, 8>(fnParamCount, i64);
            }

			builder.getFunctionType(params, i64);
        }

        // declare all functions in the file, to easily allow forward declarations
        auto& children = root.children;
        for(auto& fnNode : children){
            auto paramNum = fnNode.children[0].children.size();
            auto typelist = llvm::SmallVector<mlir::Type, 8>(paramNum, i64);
            i64.getIntOrFloatBitWidth();
            auto fnTy = builder.getFunctionType(typelist, i64);
            if(findFunction(fnNode.ident.name)){
                std::cerr << "fatal error: redefinition of function '" << fnNode.ident.name << "'\n";
                return false;
            }
			builder.create<mlir::func::FuncOp>(loc, fnNode.ident.name, fnTy);
        }

		for(auto& fn: children){
			genFunction(fn);
		}

		// TODO check for validity etc.

		return true;
	}
};
