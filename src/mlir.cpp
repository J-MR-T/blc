#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/ConstantRange.h>
#include <llvm/ADT/STLExtras.h>

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>
#include <mlir/Dialect/Traits.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
// module op
//#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Dialect.h>
#include <llvm/Support/Debug.h>
// pass stuff
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/Passes.h>
#include <mlir/Transforms/DialectConversion.h>
// conversion stuff
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>

#include "B/BDialect.h"
#include "B/BOps.h"
#include "mlir.h"
#include "util.h"

// for combining
#include "mlir/IR/PatternMatch.h"
namespace{
    // include patterns from declarative rewrite framework
#include "BCombine.inc"
}


/*
    NRoot,          // mlir equivalent: n/a
    NFunction,      // mlir equivalent: func.func
    NParamList,     // mlir equivalent: block args for entry block
    NStmtDecl,      // mlir equivalent: b.alloca/ssa constr.
    NStmtReturn,    // mlir equivalent: func.return
    NStmtBlock,     // mlir equivalent: -
    NStmtWhile,     // mlir equivalent: cf
    NStmtIf,        // mlir equivalent: cf
    NExprVar,       // mlir equivalent: b.ptr/ssa constr.
    NExprNum,       // mlir equivalent: arith.constant
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
class Generator{
public:
    bool warningsGenerated{false};
    bool successful{true};

    AST& ast;

    mlir::MLIRContext& ctx;
    mlir::OpBuilder builder;
    // TODO add actual locations
    mlir::Location loc;
    mlir::ModuleOp mod;
    
    mlir::Type i1 = mlir::IntegerType::get(&ctx, 1);
    mlir::Type i64 = mlir::IntegerType::get(&ctx, 64);
    mlir::func::FuncOp currentFn;
    mlir::Block* entryBB;

    struct BasicBlockInfo{
        bool sealed{false};
        // TODO vector instead
        llvm::DenseMap<uint64_t /* variable uid */, mlir::Value> regVarmap{};
    };

    // TODO vector instead
    llvm::DenseMap<uint64_t /* variable uid */, mlir::b::AllocaOp> autoVarmap{};

    llvm::DenseMap<mlir::Block*, BasicBlockInfo> blockInfo{};
    llvm::DenseMap<mlir::Block*, ASTNode*> blockArgsToResolve{};

    Generator(mlir::MLIRContext& ctx, AST& ast) : ast(ast), ctx(ctx), builder(&ctx), loc(builder.getUnknownLoc()), mod(mlir::ModuleOp::create(loc)){ 
        ctx.getOrLoadDialect<mlir::b::BDialect>();
        ctx.getOrLoadDialect<mlir::func::FuncDialect>();
        ctx.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
        ctx.getOrLoadDialect<mlir::arith::ArithDialect>();
        ctx.getOrLoadDialect<mlir::LLVM::LLVMDialect>();

        builder.setInsertionPointToStart(mod.getBody());

        ctx.printOpOnDiagnostic(true);
        ctx.printStackTraceOnDiagnostic(true);
    }

    mlir::Value makePoison(){
        // TODO maybe define a proper poison value in the b dialect or smth, but for right now this is totally fine
        return builder.create<mlir::arith::ConstantIntOp>(loc, -1, i64);
    }

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

    // REFACTOR: cache the result in a hash map
    inline mlir::func::FuncOp findFunction(llvm::StringRef name){
        mlir::func::FuncOp fn;
        mod.walk([&](mlir::func::FuncOp func){
            if(func.getName() == name) fn = func;
        });

        return fn;
    }

    inline void setBlockArgForPredecessor(mlir::Block* blockArgParent, mlir::Block* pred, mlir::BlockArgument blockArg, mlir::Value setTo){
        auto* term = pred->getTerminator();
        // TODO i can already see this not working, because of the difference in cf branches etc.
        // maybe there is some kind of 'block arg user' interface/class? doesn't appear so...

        // TODO yep, doesn't work
        // I guess just make a case distinction for every possible branching terminator...

        //IFDEBUG(
        //    for(unsigned i = 0; i < pred->getNumSuccessors(); ++i){
        //        auto* successor = pred->getSuccessor(i);
        //        if(successor == successor)
        //            assert(i == blockArg.getArgNumber() && "block arg number doesn't match successor index");
        //    }
        //)

        //assert((term->getNumOperands() <= blockArg.getArgNumber() || term->getOperand(blockArg.getArgNumber()) == mlir::Value()) && "block arg already set");

        // TODO no idea if this will harm performance, I hope it doesn't and gets optimized away

        // basically like an if expression
        auto getSuccessorOperands = [&]() -> mlir::SuccessorOperands {
            if(auto branch = mlir::dyn_cast<mlir::cf::BranchOp>(term)){
                return branch.getSuccessorOperands(0);
            }else{
                auto condBranch = mlir::dyn_cast<mlir::cf::CondBranchOp>(term);
                assert(condBranch && "unhandled terminator");

                for(auto [index, predSucc]: llvm::enumerate(condBranch.getSuccessors())){
                    if(predSucc == blockArgParent){
                        return condBranch.getSuccessorOperands(index);
                    }
                }
                assert(false && "successor not found");
            }
        };
        auto successorOperands = getSuccessorOperands();
        successorOperands.append(setTo);
        assert(successorOperands.size() == blockArgParent->getNumArguments() && successorOperands.size() - 1 == blockArg.getArgNumber() && "successor operands size doesn't match block arg count");
    }

    // fills phi nodes with correct values, assumes block is sealed
    inline void fillBlockArgs(mlir::Block* block) noexcept{
        for(auto blockArg: block->getArguments()){
            for(auto* pred: block->getPredecessors()){
                assert(blockArgsToResolve.find(block) != blockArgsToResolve.end() && "block arg not found in blockArgsToResolve");

                setBlockArgForPredecessor(block, pred, blockArg, varmapLookup(pred, *blockArgsToResolve[block]));
            }
        }
    }


    template<unsigned N>
    std::array<mlir::Block*, N> createBlocksAfterCurrent(){
        auto currentBB = builder.getInsertionBlock();

        auto insertBeforeIt = currentBB->getIterator();
        ++insertBeforeIt;
        auto ret = std::array<mlir::Block*, N>();
        for(auto& block: ret)
            block = builder.createBlock(currentBB->getParent(), insertBeforeIt);

        return ret;
    }

    mlir::Operation* getTerminatorOrNull(mlir::Block* block){
        if(block->empty())
            return nullptr;

        auto* possibleTerminator = &block->back();
        if(possibleTerminator->hasTrait<mlir::OpTrait::IsTerminator>())
            return possibleTerminator;

        return nullptr;
    }

    // Seals the block and fills phis
    inline void sealBlock(mlir::Block* block){
        auto& [sealed, _] = blockInfo[block];
        sealed = true;
        fillBlockArgs(block);
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

                    for(auto* pred:  block->getPredecessors()){
                        setBlockArgForPredecessor(block, pred, blockArg, varmapLookup(pred, node));
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

    inline mlir::Value setRegisterVar(ASTNode& node, mlir::Value val) noexcept{
        return setRegisterVar(builder.getInsertionBlock(), node, val);
    }

    void genStmt(ASTNode& stmtNode) noexcept{
        switch(stmtNode.type){
        case ASTNode::Type::NStmtDecl: // always contains initializer
            {
            auto initializerNode = stmtNode.children[0];
            auto initializer = genExpr(initializerNode);
            if(stmtNode.ident.type == IdentifierInfo::AUTO){
                auto insertPoint = builder.getInsertionPoint();
                auto insertBB = builder.getInsertionBlock();
                builder.setInsertionPointToStart(entryBB);

                // create alloca at start of entry block
                auto alloca = autoVarmap[stmtNode.ident.uID] = builder.create<mlir::b::AllocaOp>(loc, 8);

                // reset insertion point
                builder.setInsertionPoint(insertBB, insertPoint);

                // store initializer
                builder.create<mlir::b::StoreOp>(loc, alloca, initializer, 8);
            }else{
                setRegisterVar(stmtNode, initializer);
            }
            break;
            }
        case ASTNode::Type::NStmtReturn:
            {
            auto val = stmtNode.children.size() == 1 ? genExpr(stmtNode.children[0]): makePoison(); // TODO warn about the second case
            builder.create<mlir::func::ReturnOp>(loc, val);
            }
            break;
        case ASTNode::Type::NStmtBlock:
            genStmts(stmtNode);
            break;
        case ASTNode::Type::NStmtWhile:
            {
            // using cf
            auto beforeLoopBB = builder.getInsertionBlock();

            auto [headerBB, bodyBB, contBB] = createBlocksAfterCurrent<3>();

            blockInfo[headerBB].sealed = false;
            blockInfo[bodyBB].sealed = true;
            blockInfo[contBB].sealed = true; // technically we haven't generated the header yet, but as soon as we will, it will be sealed (only possible predecessor is headerBB)

            // branch from before to header
            builder.setInsertionPointToEnd(beforeLoopBB);
            builder.create<mlir::cf::BranchOp>(loc, headerBB);

            // generate header
            builder.setInsertionPointToEnd(headerBB);
            auto condI64 = genExpr(stmtNode.children[0]);
            auto condI1 = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ne, std::move(condI64), builder.create<mlir::arith::ConstantIntOp>(loc, 0, i64));
            builder.create<mlir::cf::CondBranchOp>(loc, condI1, bodyBB, contBB);

            // generate body
            builder.setInsertionPointToEnd(bodyBB);
            genStmt(stmtNode.children[1]);
            bodyBB = builder.getInsertionBlock(); // because the body can generate new blocks

            // if the body is without a terminator (return/etc.), branch to header
            if(!getTerminatorOrNull(bodyBB))
                builder.create<mlir::cf::BranchOp>(loc, headerBB);

            sealBlock(headerBB); // seal before block, after has been generated now

            // reset insertion point to contBB
            builder.setInsertionPointToEnd(contBB);
            }
            break;
        case ASTNode::Type::NStmtIf:
            {
            // using cf
            auto beforeIfBB = builder.getInsertionBlock();

            auto [thenBB, elseBB, contBB] = createBlocksAfterCurrent<3>();

            blockInfo[thenBB].sealed = true;
            blockInfo[elseBB].sealed = true;
            blockInfo[contBB].sealed = true;

            builder.setInsertionPointToEnd(beforeIfBB);

            // generate condition
            auto condI64 = genExpr(stmtNode.children[0]);
            auto condI1 = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ne, std::move(condI64), builder.create<mlir::arith::ConstantIntOp>(loc, 0, i64));
            builder.create<mlir::cf::CondBranchOp>(loc, condI1, thenBB, elseBB);

            // generate then
            builder.setInsertionPointToEnd(thenBB);
            genStmt(stmtNode.children[1]);
            thenBB = builder.getInsertionBlock(); // because the then can generate new blocks

            // if the then is without a terminator (return/etc.), branch to cont
            if(!getTerminatorOrNull(thenBB))
                builder.create<mlir::cf::BranchOp>(loc, contBB);

            // generate else
            builder.setInsertionPointToEnd(elseBB);

            // if there is an else, generate it, otherwise keep the block, but just branch to cont immediately, for simplicity
            bool hasElse = stmtNode.children.size() == 3;
            if(hasElse){
                genStmt(stmtNode.children[2]);
                elseBB = builder.getInsertionBlock(); // because the else can generate new blocks
            }

            // if the else is without a terminator (return/etc.), branch to cont
            if(!getTerminatorOrNull(elseBB))
                builder.create<mlir::cf::BranchOp>(loc, contBB);

            // continue at cont
            builder.setInsertionPointToEnd(contBB);
            }
            break;

        case ASTNode::Type::NExprVar:
        case ASTNode::Type::NExprNum:
        case ASTNode::Type::NExprCall:
        case ASTNode::Type::NExprUnOp:
        case ASTNode::Type::NExprBinOp:
        case ASTNode::Type::NExprSubscript:
            genExpr(stmtNode);
            break;

        default:
            assert(false && "invalid node type for stmt");
            break;
        }
    }

    void genStmts(ASTNode& blockNode) noexcept{
        assert(blockNode.type == ASTNode::Type::NStmtBlock && "genStmts can only be called with a block node");

        for(auto& stmtNode : blockNode.children){
            genStmt(stmtNode);
            if(stmtNode.type == ASTNode::Type::NStmtReturn)
                // stop the generation for this block
                break;
        }
    }

    mlir::arith::AddIOp subscriptAddress(ASTNode& subscriptNode) noexcept{
        assert(subscriptNode.type == ASTNode::Type::NExprSubscript && "subscriptAddress can only be called with a subscript node");

        auto addr = genExpr(subscriptNode.children[0]);
        auto index = genExpr(subscriptNode.children[1]);
        auto sizespec = subscriptNode.value;

        auto ptrAsInt = builder.create<mlir::arith::AddIOp>(loc, addr, builder.create<mlir::arith::MulIOp>(loc, index, builder.create<mlir::arith::ConstantIntOp>(loc, sizespec, i64)));
        return ptrAsInt;
    }

    // TODO maybe use mlir::TypedValue<mlir::IntegerType>as return type instead
    mlir::Value genExpr(ASTNode& exprNode) noexcept{
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

                    auto alloca = mlir::dyn_cast<mlir::b::AllocaOp>(value.getDefiningOp());
                    assert(alloca && "auto variable must be an alloca");
                    return builder.create<mlir::b::LoadOp>(loc, alloca, 8);
                }
                }
            case ASTNode::Type::NExprNum:
                return builder.create<mlir::arith::ConstantIntOp>(loc, exprNode.value, i64);
            case ASTNode::Type::NExprBinOp:
                {
                auto& lhsNode = exprNode.children[0];
                auto& rhsNode = exprNode.children[1];

                // shortcircuiting
                if(bool isAnd = exprNode.op == Type::LOGICAL_AND; isAnd || exprNode.op == Type::LOGICAL_OR){
                    auto lhs = genExpr(lhsNode);

                    auto lhsBB  = builder.getInsertionBlock();
                    auto [rhsBB, contBB] = createBlocksAfterCurrent<2>();
                    auto contArg = contBB->addArgument(i1, loc); // i1 arg, gets extended to i64 later

                    // know all predecessors of rhs, cont immediately -> seal
                    blockInfo[rhsBB].sealed = true;
                    blockInfo[contBB].sealed = true;

                    builder.setInsertionPointToEnd(lhsBB);
                    auto lhsCmp = builder.create<CmpIOp>(loc, CmpIPredicate::ne, std::move(lhs), builder.create<ConstantIntOp>(loc, 0, i64));

                    if(isAnd){
                        builder.create<mlir::cf::CondBranchOp>(loc, lhsCmp, rhsBB, mlir::ArrayRef<mlir::Value>(), contBB, mlir::ArrayRef<mlir::Value>(builder.create<ConstantIntOp>(loc, 0, i1)));
                    }else{
                        builder.create<mlir::cf::CondBranchOp>(loc, lhsCmp, contBB, mlir::ArrayRef<mlir::Value>(builder.create<ConstantIntOp>(loc, 1, i1)), rhsBB, mlir::ArrayRef<mlir::Value>());
                    }

                    // rhs
                    builder.setInsertionPointToEnd(rhsBB);
                    auto rhs = genExpr(rhsNode);
                    rhsBB = builder.getInsertionBlock(); // because the rhs can generate new blocks
                    auto rhsCmp = builder.create<CmpIOp>(loc, CmpIPredicate::ne, std::move(rhs), builder.create<ConstantIntOp>(loc, 0, i64));
                    builder.create<mlir::cf::BranchOp>(loc, contBB, /* block args */ mlir::ArrayRef<mlir::Value>({rhsCmp}));

                    // rhs can't generate phis/block args because it is sealed and has a single predecessor

                    builder.setInsertionPointToEnd(contBB);
                    // don't need to fill cont phis, because it is sealed from the start -> all phis sealed
                    return builder.create<ExtUIOp>(loc, i64, contArg);
                }

                auto rhs = genExpr(rhsNode);
                if(exprNode.op == Type::ASSIGN){
                    if(lhsNode.type == ASTNode::Type::NExprSubscript){
                        // storing subscript
                        auto ptrAsInt = subscriptAddress(lhsNode);
                        auto ptr = builder.create<mlir::b::IntToPtrOp>(loc, ptrAsInt);
                        builder.create<mlir::b::StoreOp>(loc, ptr, rhs, lhsNode.value);
                    }else{
                        // normal assign
                        assert(lhsNode.type == ASTNode::Type::NExprVar && "lhs of assign must be a variable, should have been checked in SemanticAnalysis");

                        if(lhsNode.ident.type == IdentifierInfo::REGISTER){
                            setRegisterVar(lhsNode, rhs);
                        }else{
                            assert(lhsNode.ident.type == IdentifierInfo::AUTO && "variable can only be auto or register");

                            auto alloca = autoVarmap[lhsNode.ident.uID];
                            builder.create<mlir::b::StoreOp>(loc, alloca, rhs, 8);
                        }
                    }
                    return rhs;
                }
                auto lhs = genExpr(lhsNode);

                switch(exprNode.op){
                    // sadly no nicer way to just change the Op type, can't just save a type as a variable and use a unified case :(
                    // and using op names instead would mean they'd have to be parsed again (right?), that would be a waste

#define BIN_OP(type) return builder.createOrFold<type>(loc, std::move(lhs), std::move(rhs))

                    // bitwise
                    case Type::AMPERSAND:   BIN_OP(AndIOp);
                    case Type::BITWISE_OR:  BIN_OP(OrIOp);
                    case Type::BITWISE_XOR: BIN_OP(XOrIOp);

                    // arithmetic
                    case Type::PLUS:   BIN_OP(AddIOp);
                    case Type::MINUS:  BIN_OP(SubIOp);
                    case Type::TIMES:  BIN_OP(MulIOp);
                    case Type::DIV:    BIN_OP(DivSIOp);
                    case Type::MOD:    BIN_OP(RemSIOp);
                    case Type::SHIFTL: BIN_OP(ShLIOp);
                    case Type::SHIFTR: BIN_OP(ShRSIOp);

#undef BIN_OP

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
                                assert(false && "invalid expression");
                                break;
                        }
                        return builder.create<mlir::arith::ExtUIOp>(loc, i64, builder.create<mlir::arith::CmpIOp>(loc, pred, std::move(lhs), std::move(rhs)));
                }
                }
            case ASTNode::Type::NExprUnOp:
                {
#define childOp genExpr(exprNode.children[0])
                switch(exprNode.op){
                    case Type::MINUS: return builder.createOrFold<SubIOp>(loc, builder.create<ConstantIntOp>(loc,    0, i64), childOp);
                    case Type::TILDE: return builder.createOrFold<XOrIOp>(loc, builder.create<ConstantIntOp>(loc, -1ll, i64), childOp);
                    case Type::LOGICAL_NOT:
                        return builder.create<ExtUIOp>(loc, i64,
                            builder.create<CmpIOp>(loc, CmpIPredicate::eq, childOp, builder.create<ConstantIntOp>(loc, 0, i64))
                        );
#undef childOp
                    case Type::AMPERSAND:
                        if(exprNode.children[0].type == ASTNode::Type::NExprVar){
                            return builder.create<mlir::b::PtrToIntOp>(loc, autoVarmap[exprNode.children[0].ident.uID]);
                        }else{
                            assert(exprNode.children[0].type == ASTNode::Type::NExprSubscript && "only variables and subscripts can have their address taken");

                            // storing subscript but just the address
                            return subscriptAddress(exprNode.children[0]);
                        }
                    default:
                        assert(false);
                        break;
                }
                assert(false);
                }
            case ASTNode::Type::NExprCall:
                {
                auto calledFn = exprNode.ident.name;

                auto callee = mod.lookupSymbol<mlir::func::FuncOp>(calledFn);
                assert(callee && "Function for call not found -> forward declarations went wrong");


                llvm::SmallVector<mlir::Value, 8> args(exprNode.children.size());
                for(unsigned int i = 0; i < exprNode.children.size(); ++i)
                    args[i] = genExpr(exprNode.children[i]);

                if(args.size() != callee.getNumArguments()){
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
                        warn("Call to function ", exprNode.ident.name, " with ", args.size(), " arguments, but function has ", callee.getNumArguments(), " parameters");
                        return makePoison();
                    }
                }
                return builder.create<mlir::func::CallOp>(loc, callee, args).getResult(0);
                }
            case ASTNode::Type::NExprSubscript:
                // this is a loading subscript, storing is in the assignment above
                {
                auto ptrAsInt = subscriptAddress(exprNode);
                auto ptr = builder.create<mlir::b::IntToPtrOp>(loc, ptrAsInt);
                return builder.create<mlir::b::LoadOp>(loc, ptr, exprNode.value);
                }
            default:
                break;
        }
        assert(false && "Invalid expression node");
    }

    mlir::func::FuncOp genFunction(ASTNode& fnNode) noexcept{
        auto& paramListNode = fnNode.children[0];
        auto& bodyNode = fnNode.children[1];

        auto fn = mod.lookupSymbol<mlir::func::FuncOp>(fnNode.ident.name);

        // no block by default
        entryBB = &fn.getBody().emplaceBlock();
        assert(fn.getBody().hasOneBlock() && "Function body doesn't have exactly one block");

        blockInfo[entryBB].sealed = true;

        builder.setInsertionPointToStart(entryBB);

        // generate parameters
        for(unsigned int i = 0; i < paramListNode.children.size(); ++i){
            auto& paramNode = paramListNode.children[i];
            auto arg = entryBB->addArgument(i64, loc);
            setRegisterVar(paramNode, arg);
        }

        genStmt(bodyNode);

        auto* endBB = builder.getInsertionBlock();
        if(!getTerminatorOrNull(endBB)){
            if(!endBB->hasNoPredecessors())
                warn("Function \"", fn.getName(), "\" might not return a value");

            builder.create<mlir::func::ReturnOp>(loc, (mlir::Value)builder.create<mlir::arith::ConstantIntOp>(loc, 0, i64));
        }

        return fn;
    }

    void generate() noexcept{
        ASTNode& root = ast.root;

        // declare implicitly declared functions
        for(auto& entry: SemanticAnalysis::externalFunctionsToNumParams){
            auto fnParamCount = entry.second;
            llvm::SmallVector<mlir::Type, 8> params;
            if(fnParamCount == EXTERNAL_FUNCTION_VARARGS){
                // TODO varargs unsupported for now because mlir doesn't seem to support them (well the llvm dialect does...)
                warn("varargs functions are not supported with the MLIR backend yet, stopping compilation");
                successful = false;
                return;
            }else{
                params = llvm::SmallVector<mlir::Type, 8>(fnParamCount, i64);
            }

            auto decl = builder.create<mlir::func::FuncOp>(loc, entry.first(), builder.getFunctionType(params, i64));
            decl.setPrivate(); // TODO technically this is a bit stupid, because these fns are exactly the ones that arent private, but "func.func op symbol declaration cannot have public visibility"
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
                successful = false;
                return;
            }
            auto decl = builder.create<mlir::func::FuncOp>(loc, fnNode.ident.name, fnTy);
            decl.setPrivate();
        }

        for(auto& fn: children){
            genFunction(fn);
        }

        // check for validty
        if (failed(mlir::verify(mod))) {
            mod.emitError("Module verification failed");
            successful = false;
        }
    }

};


std::tuple<bool /* success? */, bool /* warnings generated? */, mlir::OwningOpRef<mlir::ModuleOp>> generate(mlir::MLIRContext& ctx, AST& ast) noexcept{
    Generator gen(ctx, ast);
    gen.generate();
    return {gen.successful, gen.warningsGenerated, gen.mod};
}

// returns whether or not pass succeeded
bool runCanonicalizer(mlir::ModuleOp mod) noexcept{
    mlir::PassManager pm(mod->getName());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
    return succeeded(pm.run(mod));
}

void canonicalizerTest() noexcept{
    mlir::MLIRContext ctx;
    ctx.getOrLoadDialect<mlir::b::BDialect>();
    ctx.getOrLoadDialect<mlir::func::FuncDialect>();
    ctx.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
    ctx.getOrLoadDialect<mlir::arith::ArithDialect>();

    mlir::OpBuilder builder(&ctx);
    auto loc = builder.getUnknownLoc();

    auto mod = mlir::ModuleOp::create(loc);
    builder.setInsertionPointToStart(mod.getBody());
    auto fn = builder.create<mlir::func::FuncOp>(loc, "test", builder.getFunctionType({}, {}));
    auto* entryBB = fn.addEntryBlock();
    builder.setInsertionPointToStart(entryBB);

    auto a = builder.create<mlir::b::IntToPtrOp>(loc, builder.create<mlir::arith::ConstantIntOp>(loc, 0, mlir::IntegerType::get(&ctx, 64)));
    auto b = builder.create<mlir::b::PtrToIntOp>(loc, a);
    builder.create<mlir::func::ReturnOp>(loc, (mlir::Value) b);

    mod.dump();

    runCanonicalizer(mod);

    mod.dump();
}

// === lowering ===

// TODO whats the difference between using the adaptor to get an operand and using op.getOperand()?

struct AllocaOpLowering : public mlir::ConvertOpToLLVMPattern<mlir::b::AllocaOp> {
    // reuse the existing constructor from ConvertOpToLLVMPattern
    using ConvertOpToLLVMPattern<mlir::b::AllocaOp>::ConvertOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(mlir::b::AllocaOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override{
        auto elementType = rewriter.getIntegerType(adaptor.getWidth());
        auto elementPtrType = getTypeConverter()->getPointerType(elementType);
        auto widthAsValue = rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI64Type(), adaptor.getWidth());
        rewriter.replaceOpWithNewOp<mlir::LLVM::AllocaOp>(op, elementPtrType, elementType, widthAsValue, 0);
        return mlir::success();
    }
};

struct IntToPtrOpLowering : public mlir::ConvertOpToLLVMPattern<mlir::b::IntToPtrOp> {
    // reuse the existing constructor from ConvertOpToLLVMPattern
    using ConvertOpToLLVMPattern<mlir::b::IntToPtrOp>::ConvertOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(mlir::b::IntToPtrOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override{
        rewriter.replaceOpWithNewOp<mlir::LLVM::IntToPtrOp>(op, getTypeConverter()->getPointerType(rewriter.getIntegerType(64)), adaptor.getIntt());
        return mlir::success();
    }
};

struct PtrToIntOpLowering : public mlir::ConvertOpToLLVMPattern<mlir::b::PtrToIntOp> {
    // reuse the existing constructor from ConvertOpToLLVMPattern
    using ConvertOpToLLVMPattern<mlir::b::PtrToIntOp>::ConvertOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(mlir::b::PtrToIntOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override{
        rewriter.replaceOpWithNewOp<mlir::LLVM::PtrToIntOp>(op, rewriter.getIntegerType(64), adaptor.getPtr());
        return mlir::success();
    }
};

struct StoreOpLowering : public mlir::ConvertOpToLLVMPattern<mlir::b::StoreOp> {
    // reuse the existing constructor from ConvertOpToLLVMPattern
    using ConvertOpToLLVMPattern<mlir::b::StoreOp>::ConvertOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(mlir::b::StoreOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override{
        // type is inferred through type of value
        auto index = rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI64Type(), 0);
        auto intType = rewriter.getIntegerType(adaptor.getWidth()*8);
        rewriter.replaceOpWithNewOp<mlir::LLVM::StoreOp>(op, adaptor.getValue(), rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), getTypeConverter()->getPointerType(intType), intType, adaptor.getPtr(), mlir::ValueRange({index})));
        return mlir::success();
    }
};

struct LoadOpLowering : public mlir::ConvertOpToLLVMPattern<mlir::b::LoadOp> {
    // reuse the existing constructor from ConvertOpToLLVMPattern
    using ConvertOpToLLVMPattern<mlir::b::LoadOp>::ConvertOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(mlir::b::LoadOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override{
        auto load = rewriter.create<mlir::LLVM::LoadOp>(op.getLoc(), rewriter.getIntegerType(8*adaptor.getWidth()) /* explicit, because pointers are opaque */, adaptor.getPtr());
        if(adaptor.getWidth()!=8)
            rewriter.replaceOpWithNewOp<mlir::LLVM::ZExtOp>(op, rewriter.getIntegerType(64), load);
        else
            rewriter.replaceOp(op, load.getResult());
        return mlir::success();
    }
};


mlir::LogicalResult lowerToLLVM(mlir::ModuleOp mod) noexcept{
    IFDEBUG(llvm::setCurrentDebugType("dialect-conversion")); // like debug-only=dialect-conversion

    mlir::MLIRContext& ctx = *mod.getContext();
    mlir::ConversionTarget target(ctx);
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();
    target.addLegalOp<mlir::ModuleOp>();

    // needed for other conversions for the pre-existing dialect, as well as for the b::PointerType
    mlir::LLVMTypeConverter typeConverter(&ctx);
    assert(typeConverter.useOpaquePointers() && "opaque pointers are required for the lowering to llvm");
    typeConverter.addConversion([&ctx](mlir::b::PointerType) { return mlir::LLVM::LLVMPointerType::get(&ctx,0); });

    mlir::RewritePatternSet patterns(&ctx);
    //mlir::populateSCFToControlFlowConversionPatterns(patterns);
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);

    patterns.add<AllocaOpLowering, IntToPtrOpLowering, PtrToIntOpLowering, StoreOpLowering, LoadOpLowering>(typeConverter);

    return mlir::applyFullConversion(mod, target, std::move(patterns));
}

/// MLIR automatically inserts malloc/free declarations in the llvmir dialect module -> LLVM IR module conversion. This is annoying, because it doesn't regard custom declarations for malloc/free
void workaroundAutomaticFreeMallocdecls(mlir::ModuleOp mod) noexcept{
    auto mallocDecl = mod.lookupSymbol<mlir::LLVM::LLVMFuncOp>("malloc");
    auto freeDecl = mod.lookupSymbol<mlir::LLVM::LLVMFuncOp>("free");

    mlir::OpBuilder builder(mod.getContext());
    if(mallocDecl){
        auto mallocUsesOpt = mallocDecl.getSymbolUses(mod);
        assert(mallocUsesOpt.has_value() && "declarations should only be inserted upon use");

        for(auto mallocCallUse:*mallocUsesOpt){
            auto mallocCall = mallocCallUse.getUser();

            builder.setInsertionPoint(mallocCall);
            auto mallocPtrToInt = builder.create<mlir::LLVM::PtrToIntOp>(mallocCall->getLoc(), builder.getIntegerType(64), mallocCall->getResult(0));
            mallocCall->replaceAllUsesWith(mallocPtrToInt);
        }
    }

    if(freeDecl){
        auto freeUsesOpt = freeDecl.getSymbolUses(mod);
        assert(freeUsesOpt.has_value() && "declarations should only be inserted upon use");

        for(auto freeCallUse: *freeUsesOpt){
            auto freeCall = freeCallUse.getUser();

            builder.setInsertionPoint(freeCall);
            assert(freeCall->getResult(0).use_empty() && "free result should not be used");
            auto intToPtr = builder.create<mlir::LLVM::IntToPtrOp>(freeCall->getLoc(), mlir::LLVMTypeConverter(mod.getContext()).getPointerType(builder.getIntegerType(64)), freeCall->getOperand(0));
            freeCall->setOperand(0, intToPtr);
        }
    }
}

}; // end namespace Codegen::MLIR

// patterns stuff

void mlir::b::IntToPtrOp::getCanonicalizationPatterns(mlir::RewritePatternSet& results, MLIRContext* context){
    results.add<PointerIntRoundTripPattern>(context);
}

void mlir::b::PtrToIntOp::getCanonicalizationPatterns(mlir::RewritePatternSet& results, MLIRContext* context){
    results.add<IntPointerRoundTripPattern>(context);
}

