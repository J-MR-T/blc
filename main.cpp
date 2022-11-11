#include <iostream>
#include <iostream>
#include <string>
#include <array>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <memory>
#include <ranges>
#include <functional>
#include <regex>
#include <tuple>
#include <fstream>
#include <chrono>
#include <cctype>
#include <iomanip>
#include <sstream>
#include <charconv>
#pragma GCC diagnostic push 
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <llvm/IR/IRBuilder.h>
#pragma GCC diagnostic pop

using std::string;
using std::unique_ptr;

#ifndef NDEBUG
#define DEBUGLOG(x) std::cerr << x << std::endl; fflush(stderr);
#else
#define DEBUGLOG(x)
#endif

struct Token{
public:
    enum class Type{
        EMPTY, // "nothing token"

        NUM,
        IDENTIFIER,
        KW_AUTO,
        KW_REGISTER,
        KW_IF,
        KW_ELSE,
        KW_WHILE,
        KW_RETURN,
        L_PAREN,   // (
        R_PAREN,   // )
        L_BRACKET, // [
        R_BRACKET, // ]
        L_BRACE,   // {
        R_BRACE,   // }
        SEMICOLON,
        COMMA,

        // operators
        // unary
        LOGICAL_NOT, // !, beware: !=
        TILDE,
        AMPERSAND, // beware: &&
                   // & can be 'address operator' or 'bitwise and' depending on the context
        // MINUS can also be a unary op!

        // binary
        BITWISE_OR, // beware: ||
        BITWISE_XOR,
        BITWISE_NEG,
        AT,
        PLUS,
        MINUS, // can be unary or binary, depending on context
        ASTERISK,
        DIV,
        MOD,
        SHIFTL,
        SHIFTR,
        LESS,
        GREATER,
        LESS_EQUAL,
        GREATER_EQUAL,
        EQUAL,
        NOT_EQUAL,
        LOGICAL_AND,
        LOGICAL_OR,
        ASSIGN, // beware: ==
        EOP, // end of program
    };
    

    Type type;
    std::string value{""}; // only used for identifiers and numbers

    // turns the tokens back into their original strings for pretty printing
    static string toString(Token::Type type){
        switch(type){
            case Type::EMPTY:
                return "EMPTY";
            case Type::NUM:
                return "NUM";
            case Type::IDENTIFIER:
                return "IDENTIFIER";
            case Type::KW_AUTO:
                return "auto";
            case Type::KW_REGISTER:
                return "register";
            case Type::KW_IF:
                return "if";
            case Type::KW_ELSE:
                return "else";
            case Type::KW_WHILE:
                return "while";
            case Type::KW_RETURN:
                return "return";
            case Type::SEMICOLON:
                return ";";
            case Type::COMMA:
                return ",";
            case Type::L_PAREN:
                return "(";
            case Type::R_PAREN:
                return ")";
            case Type::L_BRACKET:
                return "[";
            case Type::R_BRACKET:
                return "]";
            case Type::L_BRACE:
                return "{";
            case Type::R_BRACE:
                return "}";
            case Type::LOGICAL_NOT:
                return "!";
            case Type::TILDE:
                return "~";
            case Type::AMPERSAND:
                return "&";
            case Type::BITWISE_OR:
                return "|";
            case Type::BITWISE_XOR:
                return "^";
            case Type::BITWISE_NEG:
                return "~";
            case Type::AT:
                return "@";
            case Type::PLUS:
                return "+";
            case Type::MINUS:
                return "-";
            case Type::ASTERISK:
                return "*";
            case Type::DIV:
                return "/";
            case Type::MOD:
                return "%";
            case Type::SHIFTL:
                return "<<";
            case Type::SHIFTR:
                return ">>";
            case Type::LESS:
                return "<";
            case Type::GREATER:
                return ">";
            case Type::LESS_EQUAL:
                return "<=";
            case Type::GREATER_EQUAL:
                return ">=";
            case Type::EQUAL:
                return "==";
            case Type::NOT_EQUAL:
                return "!=";
            case Type::ASSIGN:
                return "=";
            case Type::LOGICAL_AND:
                return "&&";
            case Type::LOGICAL_OR:
                return "||";
            case Type::EOP:
                return "EOP";
            }
        return "UNKNOWN";
    }

    string toString(){
        if(type == Type::NUM || type == Type::IDENTIFIER){
            return string{value};
        }else{
            return Token::toString(type);
        }
    }

    bool operator == (const Token& other) const{
        return type == other.type && value == other.value;
    }

    // this is unbelievably ugly, but i need it in the following expression constructor
    string empty(){
        return "";
    }

};

class ParsingException : public std::runtime_error {
    public:
        ParsingException(string msg) : std::runtime_error(msg){}
};

Token emptyToken{
    Token::Type::EMPTY,
    ""
};

class Tokenizer{
public:

class UnexpectedTokenException :  public ParsingException {
    public:
        UnexpectedTokenException(Tokenizer& tokenizer, Token::Type expected = Token::Type::EMPTY) : ParsingException(
                  tokenizer.peekToken().empty() // very ugly, but it is so the token is peeked and can be accessed later, together with the accurate line number
                + "Line " + std::to_string(tokenizer.getLineNum()) + ": "
                + "Unexpected token: " + (tokenizer.peeked).toString() + (((tokenizer.peeked).type==Token::Type::NUM || (tokenizer.peeked).type == Token::Type::IDENTIFIER)?" (type: "+Token::toString((tokenizer.peeked).type)+")" : "") 
                + (expected != Token::Type::EMPTY ? (", expected: ") + Token::toString(expected):"")
                ){ }
};

    static string initProg(std::ifstream& inputFile){
        std::stringstream ss1;
        ss1 << inputFile.rdbuf();
        return ss1.str();
    }

    Tokenizer(string prog) : matched(emptyToken), prog(prog), peeked(emptyToken) {}

    Tokenizer(std::ifstream& inputFile) : matched(emptyToken), prog(initProg(inputFile)), peeked(emptyToken) {
        std::stringstream ss2;
        ss2 << std::endl;
        newline = ss2.str();

    }

    

    static const std::unordered_map<string, Token::Type> keywords; // initialized below
    const std::regex numberRegex{"[0-9]+"};
    const std::regex identifierRegex{"[a-zA-Z_][a-zA-Z0-9_]*"};

    string newline;

    Token matched;

    Token peekToken(){
#define RET(type) peeked = Token{type}; return peeked;
#define RET_NAME(type,name) peeked = Token{type, name}; return peeked;
        if(peeked == emptyToken){
            if(progI>=prog.size()){
                //return EOP token
                RET(Token::Type::EOP);
            }

            // skip whitespace & comments 
            while(true){
                progI=prog.find_first_not_of(" \f\n\r\t\v", progI); // same chars as isspace uses
                if(progI>=prog.size()){
                    RET(Token::Type::EOP);
                }


                if(prog[progI] == '/'){
                    if(progI+1 < prog.size() && prog[progI+1] == '/'){
                        // single line comment
                        progI+=2;

                        progI=prog.find(newline, progI);
                        progI+=newline.size();
                    }else if(progI+1 >= prog.size()){
                        RET(Token::Type::EOP);
                    }else{
                        break;
                    }
                }else{
                    break;
                }
            }

            if(prog.size()-progI <= 0){
                //return EOP token
                RET(Token::Type::EOP);
            }


            //NUM
            if(isdigit(prog[progI])){
                std::smatch match;
                if(std::regex_search(prog.cbegin()+progI, prog.cend(), match, numberRegex)){
                    string numStr = match[0];
                    progI += numStr.size();
                    RET_NAME(Token::Type::NUM, numStr);
                }
            }

            //IDENTIFIER
            if(isalpha(prog[progI]) || prog[progI] == '_'){
                std::smatch match;
                if(std::regex_search(prog.cbegin()+progI, prog.cend(), match, identifierRegex)){
                    string idStr = match[0];
                    progI += idStr.size();
                    //check if it's a keyword
                    if(keywords.contains(idStr)){
                        RET(keywords.at(idStr));
                    }else{
                        RET_NAME(Token::Type::IDENTIFIER, idStr);
                    }
                }
            }

            //single characters
            Token::Type type = Token::Type::EMPTY;
            //parentheses, brackets, braces, unabiguous operators, ...
            switch(prog[progI]){
                case '(':
                    type = Token::Type::L_PAREN;
                    break;
                case ')':
                    type = Token::Type::R_PAREN;
                    break;
                case '[':
                    type = Token::Type::L_BRACKET;
                    break;
                case ']':
                    type = Token::Type::R_BRACKET;
                    break;
                case '{':
                    type = Token::Type::L_BRACE;
                    break;
                case '}':
                    type = Token::Type::R_BRACE;
                    break;
                case '~':
                    type = Token::Type::TILDE;
                    break;
                case '^':
                    type = Token::Type::BITWISE_XOR;
                    break;
                case '@':
                    type = Token::Type::AT;
                    break;
                case '+':
                    type = Token::Type::PLUS;
                    break;
                case '-':
                    type = Token::Type::MINUS;
                    break;
                case '*':
                    type = Token::Type::ASTERISK;
                    break;
                case '/':
                    type = Token::Type::DIV;
                    break;
                case '%':
                    type = Token::Type::MOD;
                    break;
                case ';':
                    type = Token::Type::SEMICOLON;
                    break;
                case ',':
                    type = Token::Type::COMMA;
                    break;
            }

            if(type!=Token::Type::EMPTY){
                progI++;
                RET(type);
            }

            //two characters
            if(prog.size()-progI >= 2){
                //shift operators
                if(prog[progI+0] == '<' && prog[progI+1] == '<'){
                    type = Token::Type::SHIFTL;
                }
                if(prog[progI+0] == '>' && prog[progI+1] == '>'){
                    type = Token::Type::SHIFTR;
                }

                //comparison operators
                if(prog[progI+0] == '<' && prog[progI+1] == '='){
                    type = Token::Type::LESS_EQUAL;
                }
                if(prog[progI+0] == '>' && prog[progI+1] == '='){
                    type = Token::Type::GREATER_EQUAL;
                }
                if(prog[progI+0] == '=' && prog[progI+1] == '='){
                    type = Token::Type::EQUAL;
                }
                if(prog[progI+0] == '!' && prog[progI+1] == '='){
                    type = Token::Type::NOT_EQUAL;
                }

                //boolean operators
                if(prog[progI+0] == '&' && prog[progI+1] == '&'){
                    type = Token::Type::LOGICAL_AND;
                }
                if(prog[progI+0] == '|' && prog[progI+1] == '|'){
                    type = Token::Type::LOGICAL_OR;
                }

                if(type!=Token::Type::EMPTY){
                    progI += 2;
                    RET(type);
                }
            }

            //ambiguous one character operators, ambiguity has been cleared by previous ifs
            switch(prog[progI+0]){
                case '<':
                    type = Token::Type::LESS;
                    break;
                case '>':
                    type = Token::Type::GREATER;
                    break;
                case '=':
                    type = Token::Type::ASSIGN;
                    break;
                case '&':
                    type = Token::Type::AMPERSAND;
                    break;
                case '|':
                    type = Token::Type::BITWISE_OR;
                    break;
                case '!':
                    type = Token::Type::LOGICAL_NOT;
                    break;
            }

            if(type!=Token::Type::EMPTY){
                progI++;
                RET(type);
            }

            //invalid character
            string errormsg = "Invalid character: ";
            errormsg += prog[progI+0];
            throw std::runtime_error(errormsg);

        }else{
            return peeked;
        }
    }

    std::uint64_t getLineNum(){
        //this should work on windows too, because '\r\n' also contains '\n', but honestly if windows users have wrong line numbers in their errors, so be it :P
        return std::count(prog.begin(), prog.begin()+progI, '\n')+1;
    }

    Token nextToken(){
        Token tok = peekToken();
        peeked = emptyToken;
        return tok;
    }

    bool matchToken(Token::Type type, bool advance = true){
        Token tok = peekToken();
        if(tok.type == type){
            matched = tok;
            if(advance) nextToken();
            return true;
        }else{
            matched = emptyToken;
            return false;
        }
    }

    void assertToken(Token::Type type, bool advance = true){
        if(!matchToken(type, advance)){
            throw UnexpectedTokenException(*this, type);
        }
    }

    //only used for performance tests with multiple iterations
    void reset(){
        matched = emptyToken;
        peeked = emptyToken;
        progI = 0;
    }

private:
    const string prog;
    std::size_t progI{0};
    Token peeked;

};

const std::unordered_map<string, Token::Type> Tokenizer::keywords = {
    {"auto",                     Token::Type::KW_AUTO},
    {"register",                 Token::Type::KW_REGISTER},
    {"if",                       Token::Type::KW_IF},
    {"else",                     Token::Type::KW_ELSE},
    {"while",                    Token::Type::KW_WHILE},
    {"return",                   Token::Type::KW_RETURN},
};


static int nodeCounter = 0;

class ASTNode{
public:
    enum class Type{
        NRoot, // possible children: function*
        NFunction, // possible children: paramlist, block, name: yes
        NParamList, // possible children: var*
        NStmtDecl, // possible children: var, expr (*required* initializer), op: KW_AUTO/KW_REGISTER
        NStmtReturn, // possible children: expr?
        NStmtBlock, // possible children: statement*
        NStmtWhile, // possible children: expr, statement
        NStmtIf, // possible children: expr, stmt, stmt (optional else)
        NExpr, // possible children: expr, for parenthesized exprs
        NExprVar, // possible children: none, name: yes
        NExprNum, // possible children: none, value: yes
        NExprCall, // possible children: expr*, name: yes
        NExprUnOp, // possible children: expr, op: yes
        NExprBinOp, // possible children: expr, expr, op: yes
        NSubscript, // possible children: expr(addr), expr(index), num (sizespec, 1/2/4/8)
    };

    Type type;

    string uniqueDotIdentifier;

    static const std::unordered_map<Type, string> nodeTypeToDotIdentifier; // initialization below
    static const std::unordered_map<Type, std::vector<string>> nodeTypeToDotStyling; // initialization below
    static std::unordered_map<Type, llvm::GlobalObject::LinkageTypes> functionLinkageTypes; // initialization below

    //optional node attrs
    string name;
    int64_t value;
    Token::Type op; // for UnOp, BinOp, and decl/var

    std::vector<unique_ptr<ASTNode>> children{};

    ASTNode(Type type, string name = "", Token::Type op = Token::Type::EMPTY) : type(type), name(name), op(op){
        uniqueDotIdentifier = nodeTypeToDotIdentifier.at(type) + std::to_string(nodeCounter++);
    }

    string toString() {
        if(name.empty()){
            if(type == Type::NExprNum){
                return std::to_string(value);
            } else if (type == Type::NExprBinOp || type == Type::NExprUnOp){
                return Token::toString(op);
            }else {
                return nodeTypeToDotIdentifier.at(type);
            }
        }else{
            return nodeTypeToDotIdentifier.at(type) + ": " + name;
        }
    }

    void print(std::ostream& out, int indentDepth = 0){
        string indent = "    ";
        indent.append(4*indentDepth, ' ');

        for(auto& child : children){
            out << indent << toString() << " -> " << child->toString() << ";" << std::endl;
            child->print(out, indentDepth+1);
        }
    }

    void printDOT(std::ostream& out, int indentDepth = 0, bool descend = true){
        string indent = "    ";
        indent.append(4*indentDepth, ' ');

        out << indent << uniqueDotIdentifier << " [label=\"" << toString() << "\"" ;
        if(nodeTypeToDotStyling.contains(type)){
            for(auto& styleInstr : nodeTypeToDotStyling.at(type)){
                out << ", " << styleInstr;
            }
        }
        out << "];" << std::endl;

        if(descend){
            for(auto& child : children){
                out << indent << uniqueDotIdentifier << " -> " << child->uniqueDotIdentifier << ";" << std::endl;
                child->printDOT(out, indentDepth+1);
            }
        }
    }

    void iterateChildren(std::function<void(ASTNode&)> f){
        for(auto& child : children){
            f(*child);
            child->iterateChildren(f);
        }
    }

};

std::ostream& operator<< (std::ostream& out, ASTNode& node){
    node.print(out);
    return out;
}

const std::unordered_map<ASTNode::Type, string> ASTNode::nodeTypeToDotIdentifier{
    {ASTNode::Type::NRoot,        "Root"},
    {ASTNode::Type::NFunction,    "Function"},
    {ASTNode::Type::NParamList,   "ParamList"},
    {ASTNode::Type::NStmtDecl,    "Declaration"},
    {ASTNode::Type::NStmtReturn,  "Return"},
    {ASTNode::Type::NStmtBlock,   "Block"},
    {ASTNode::Type::NStmtWhile,   "While"},
    {ASTNode::Type::NStmtIf,      "If"},
    {ASTNode::Type::NExpr,        "Expr"},
    {ASTNode::Type::NExprVar,     "Var"},
    {ASTNode::Type::NExprNum,     "Number"},
    {ASTNode::Type::NExprCall,    "FunctionCall"},
    {ASTNode::Type::NExprUnOp,    "UnOp"},
    {ASTNode::Type::NExprBinOp,   "BinOp"},
    {ASTNode::Type::NSubscript,   "Subscript"},
};

const std::unordered_map<ASTNode::Type, std::vector<string>> ASTNode::nodeTypeToDotStyling{
    {ASTNode::Type::NRoot,       {"shape=house", "style=filled", "fillcolor=lightgrey"}},
    {ASTNode::Type::NFunction,   {"shape=box", "style=filled", "fillcolor=lightblue"}},
    {ASTNode::Type::NParamList,  {"shape=invtriangle"}},
    {ASTNode::Type::NStmtBlock,  {"shape=invtriangle", "style=filled", "fillcolor=grey"}},
    {ASTNode::Type::NExprUnOp,   {"style=filled", "color=chocolate3"}},
    {ASTNode::Type::NExprBinOp,  {"style=filled", "color=chocolate1"}},
    {ASTNode::Type::NExprVar,    {"style=filled", "color=lightblue1"}},
    {ASTNode::Type::NStmtDecl,   {"shape=rectangle"}},
    {ASTNode::Type::NStmtIf,     {"shape=rectangle"}},
    {ASTNode::Type::NStmtReturn, {"shape=rectangle"}},
    {ASTNode::Type::NStmtWhile,  {"shape=rectangle"}},
};

std::unordered_map<ASTNode::Type, llvm::GlobalObject::LinkageTypes> ASTNode::functionLinkageTypes{};

class AST{
public:
    ASTNode root{ASTNode::Type::NRoot};

    void printDOT(std::ostream& out){
        out << "digraph AST {" << std::endl;
        root.printDOT(out, 0, false);
        for(auto& child : root.children){
            out << root.uniqueDotIdentifier << " -> " << child->uniqueDotIdentifier << ";" << std::endl;
            out << "subgraph cluster_" << child->uniqueDotIdentifier << " {" << std::endl;
            //function cluster styling

            out << "style=filled;" << std::endl;
            out << "color=lightgrey;" << std::endl;
            out << "node [style=filled,color=white];" << std::endl;
            out << "label = \"" << child->toString() << "\";" << std::endl;
            child->printDOT(out, 1);
            out << "}" << std::endl;
        }

        out << "}" << std::endl;
    }

    void print(std::ostream& out){
        root.print(out);
    }

    size_t getRoughMemoryFootprint(){
        size_t totalSize = 0;
        root.iterateChildren([&totalSize](ASTNode& node){
            totalSize += sizeof(node);
            //seperately add the size of the children vector
            totalSize += node.children.capacity() * sizeof(unique_ptr<ASTNode>);
        });
        return totalSize;
    }

};

// you can take a C programmer out of C
// but you can't take the C out of a C programmer
#define TUP(x,y) std::make_tuple((x),(y))

// int: precedence, bool: rassoc
static const std::unordered_map<Token::Type, std::tuple<int,bool>> operators = {
    {Token::Type::L_BRACKET,           TUP(14, false)},
    // unary: 13 (handled seperately)
    {Token::Type::ASTERISK,            TUP(12, false)},
    {Token::Type::DIV,                 TUP(12, false)},
    {Token::Type::MOD,                 TUP(12, false)},
    {Token::Type::PLUS,                TUP(11, false)},
    {Token::Type::MINUS,               TUP(11, false)},
    {Token::Type::SHIFTL,              TUP(10, false)},
    {Token::Type::SHIFTR,              TUP(10, false)},
    {Token::Type::LESS,                TUP(9,  false)},
    {Token::Type::GREATER,             TUP(9,  false)},
    {Token::Type::LESS_EQUAL,          TUP(9,  false)},
    {Token::Type::GREATER_EQUAL,       TUP(9,  false)},
    {Token::Type::EQUAL,               TUP(8,  false)},
    {Token::Type::NOT_EQUAL,           TUP(8,  false)},
    {Token::Type::AMPERSAND,           TUP(7,  false)}, //bitwise and in this case
    {Token::Type::BITWISE_XOR,         TUP(6,  false)},
    {Token::Type::BITWISE_OR,          TUP(5,  false)},
    {Token::Type::LOGICAL_AND,         TUP(4,  false)},
    {Token::Type::LOGICAL_OR,          TUP(3,  false)},
    {Token::Type::ASSIGN,              TUP(1,  true)},
};

using UnexpectedTokenException = Tokenizer::UnexpectedTokenException;

class Parser{

public:
    bool failed{false};

    Parser(string prog) : tok{prog} {}

    Parser(std::ifstream& inputFile) : tok{inputFile} {}
    // only called in case there is a return at the start of a statement -> throws exception if it fails
    unique_ptr<ASTNode> parseStmtDecl(){
        if(tok.matchToken(Token::Type::KW_REGISTER) || tok.matchToken(Token::Type::KW_AUTO)){
            auto registerOrAuto = tok.matched.type;

            tok.assertToken(Token::Type::IDENTIFIER);
            auto varName = tok.matched.value;

            tok.assertToken(Token::Type::ASSIGN);

            auto initializer = parseExpr();

            //if this returns normally, it means we have a valid initializer
            tok.assertToken(Token::Type::SEMICOLON);
            auto decl = std::make_unique<ASTNode>(ASTNode::Type::NStmtDecl, "", registerOrAuto);
            decl->children.push_back(std::make_unique<ASTNode>(ASTNode::Type::NExprVar, varName.data()));
            decl->children.push_back(std::move(initializer));
            return decl;
        }
        throw UnexpectedTokenException(tok);
    }

    unique_ptr<ASTNode> parseStmtReturn(){
        tok.assertToken(Token::Type::KW_RETURN);
        auto returnStmt= std::make_unique<ASTNode>(ASTNode::Type::NStmtReturn);
        if(tok.matchToken(Token::Type::SEMICOLON)){
            return returnStmt;
        }else{
            returnStmt->children.emplace_back(parseExpr()); //parseExpr throws exception in the case of parsing error
            tok.assertToken(Token::Type::SEMICOLON);
            return returnStmt;
        }
        throw UnexpectedTokenException(tok);
    }

    unique_ptr<ASTNode> parseBlock(){
        tok.assertToken(Token::Type::L_BRACE);
        auto block = std::make_unique<ASTNode>(ASTNode::Type::NStmtBlock);
        while(!tok.matchToken(Token::Type::R_BRACE)){
            try{
                block->children.emplace_back(parseStmt());
            }catch(ParsingException& e){
                failed = true;
                std::cerr << e.what() << std::endl;
                //skip to next statement if possible
                while(!tok.matchToken(Token::Type::SEMICOLON) && !tok.matchToken(Token::Type::EOP)){
                    tok.nextToken();
                }
                if(tok.matched.type == Token::Type::EOP){
                    throw e;
                }
            }
        }
        return block;
    }

    unique_ptr<ASTNode> parseStmtIfWhile(bool isWhile){
        if((isWhile && tok.matchToken(Token::Type::KW_WHILE)) || (!isWhile && tok.matchToken(Token::Type::KW_IF))){
            tok.assertToken(Token::Type::L_PAREN);

            auto condition = parseExpr();

            tok.assertToken(Token::Type::R_PAREN);

            auto body = parseStmt();
            auto ifWhileStmt = std::make_unique<ASTNode>(isWhile ? ASTNode::Type::NStmtWhile : ASTNode::Type::NStmtIf);
            ifWhileStmt->children.push_back(std::move(condition));
            ifWhileStmt->children.push_back(std::move(body));

            if(!isWhile && tok.matchToken(Token::Type::KW_ELSE)){
                ifWhileStmt->children.push_back(parseStmt());
            }
            return ifWhileStmt;
        }
        throw UnexpectedTokenException(tok);
    }

    unique_ptr<ASTNode> parseStmt(){
        if(tok.matchToken(Token::Type::KW_RETURN, false)){
            return parseStmtReturn();
        }else if(tok.matchToken(Token::Type::KW_IF, false) || tok.matchToken(Token::Type::KW_WHILE, false)){
            return parseStmtIfWhile(tok.matched.type == Token::Type::KW_WHILE);
        }else if(tok.matchToken(Token::Type::KW_REGISTER, false) || tok.matchToken(Token::Type::KW_AUTO, false)){
            return parseStmtDecl();
        }else if(tok.matchToken(Token::Type::L_BRACE, false)){
            return parseBlock();
        }else{
            auto expr = parseExpr();
            DEBUGLOG("parsed expr:");
#ifndef NDEBUG
            expr->printDOT(std::cerr);
#endif
            tok.assertToken(Token::Type::SEMICOLON);
            DEBUGLOG("parsed semicolon");

            return expr;
        }
    }


    //avoid left recursion
    unique_ptr<ASTNode> parsePrimaryExpression(){
        //- numbers
        //- unary ops
        //- variables
        //- calls
        //- parenthesized expressions

        if(tok.matchToken(Token::Type::NUM)){
            auto num = std::make_unique<ASTNode>(ASTNode::Type::NExprNum);
            num->value = std::stoi(tok.matched.value);;
            return num;
        }else if(tok.matchToken(Token::Type::TILDE)||tok.matchToken(Token::Type::MINUS)||tok.matchToken(Token::Type::LOGICAL_NOT)||tok.matchToken(Token::Type::AMPERSAND)){ 
            auto unOp = std::make_unique<ASTNode>(ASTNode::Type::NExprUnOp, "", tok.matched.type);
            unOp->children.emplace_back(parseExpr(13)); //unary ops have 13 prec, rassoc
            return unOp;
        }else if(tok.matchToken(Token::Type::IDENTIFIER)){
            auto ident = tok.matched.value;
            if(tok.matchToken(Token::Type::L_PAREN)){
                auto call = std::make_unique<ASTNode>(ASTNode::Type::NExprCall);
                call->name = ident;
                while(true){
                    call->children.emplace_back(parseExpr());
                    if(tok.matchToken(Token::Type::COMMA)){
                        tok.assertToken(Token::Type::IDENTIFIER, false);
                    }else if(tok.matchToken(Token::Type::R_PAREN)){
                        break;
                    }else{
                        throw UnexpectedTokenException(tok);
                    }
                }
                return call;
            }else{
                return std::make_unique<ASTNode>(ASTNode::Type::NExprVar, ident.data());
            }
        }else if(tok.matchToken(Token::Type::L_PAREN)){
            auto expr = parseExpr(0);
            tok.assertToken(Token::Type::R_PAREN);
            return expr;
        }

        throw UnexpectedTokenException(tok);
    }

    //adapted from the lecture slides
    unique_ptr<ASTNode> parseExpr(int minPrec = 0){
        unique_ptr<ASTNode> lhs = parsePrimaryExpression();
        while(true){
            Token token = tok.peekToken();
            int prec;
            bool rassoc;

            if(!operators.contains(token.type)){
                // unknown operator, let the upper level parse it/expr might be finished
                return lhs;
            }
            std::tie(prec,rassoc) = operators.at(token.type);

            if(prec < minPrec){
                return lhs;
            }
            // special handling for [
            if(tok.matchToken(Token::Type::L_BRACKET)){
                auto expr = parseExpr();
                unique_ptr<ASTNode> num;
                
                if(tok.matchToken(Token::Type::AT)){
                    // has to be followed by number
                    tok.assertToken(Token::Type::NUM, false);

                    // parse sizespec as number, validate it's 1/2/4/8
                    num = parsePrimaryExpression();
                    if(
                            !(
                                num->value==1 ||
                                num->value==2 ||
                                num->value==4 ||
                                num->value==8
                             )
                      ){
                        throw ParsingException("Line " +std::to_string(tok.getLineNum())+": Expression containing " + expr->toString() + " was followed by @, but @ wasn't followed by 1/2/4/8");
                    }
                }else{
                    num = std::make_unique<ASTNode>(ASTNode::Type::NExprNum);
                    num->value = 8; //8 by default
                }

                tok.assertToken(Token::Type::R_BRACKET);

                auto subscript = std::make_unique<ASTNode>(ASTNode::Type::NSubscript);
                subscript->children.emplace_back(std::move(lhs));
                subscript->children.emplace_back(std::move(expr));
                subscript->children.emplace_back(std::move(num));

                lhs = std::move(subscript);
                continue; // continue at the next level up
            }
            tok.nextToken(); // advance the tokenizer, now that we have actually consumed the token

            int newPrec = rassoc ? prec : prec+1;
            unique_ptr<ASTNode> rhs = parseExpr(newPrec);
            auto newLhs = std::make_unique<ASTNode>(ASTNode::Type::NExprBinOp, "", token.type);
            newLhs->children.push_back(std::move(lhs));
            newLhs->children.push_back(std::move(rhs));
            lhs = std::move(newLhs);
        }
    }

    unique_ptr<ASTNode> parseFunction(){
        if(tok.matchToken(Token::Type::IDENTIFIER)){
            auto name = tok.matched.value;
            if(tok.matchToken(Token::Type::L_PAREN)){
                auto paramlist = std::make_unique<ASTNode>(ASTNode::Type::NParamList);
                try{
                    while(true){
                        if(tok.matchToken(Token::Type::R_PAREN)){
                            break;
                        }else if(tok.matchToken(Token::Type::COMMA) && tok.matchToken(Token::Type::IDENTIFIER, false)){
                            // a comma needs to be followed by an identifier, so this needs to be checked here, but let the next loop iteration actually handle the identifier
                            continue;
                        }else if(tok.matchToken(Token::Type::IDENTIFIER)){
                            auto paramname = tok.matched.value;
                            //identifiers need to be seperated by commas, not follow each other directly
                            if(tok.matchToken(Token::Type::IDENTIFIER, false)){
                                throw UnexpectedTokenException(tok, Token::Type::R_PAREN);
                            }
                            paramlist->children.emplace_back(std::make_unique<ASTNode>(ASTNode::Type::NExprVar, paramname.data()));
                        }else{
                            throw UnexpectedTokenException(tok, Token::Type::R_PAREN);
                        }
                    }
                }catch(UnexpectedTokenException& e){
                    std::cerr << e.what() << std::endl;
                    //skip to next known block
                    while(!tok.matchToken(Token::Type::L_BRACE, false) && !tok.matchToken(Token::Type::EOP, false)){
                        tok.nextToken();
                    }
                    if(tok.matched.type == Token::Type::EOP){
                        throw e;
                    }
                }
                // at this point an R_PAREN or syntax error is guaranteed
                auto body = parseBlock();
                auto func = std::make_unique<ASTNode>(ASTNode::Type::NFunction, name.data());
                func->children.push_back(std::move(paramlist));
                func->children.push_back(std::move(body));
                DEBUGLOG("parsed function \"" << name << "\"");
                return func;
            }
        }
        throw UnexpectedTokenException(tok);
    }

    unique_ptr<AST> parse(){
        auto ast = std::make_unique<AST>();
        auto& root = ast->root;

        // parse a bunch of functions
        while(!tok.matchToken(Token::Type::EOP, false)){
            root.children.emplace_back(parseFunction());
        }
        return ast;
    }

    void resetTokenizer(){
        tok.reset();
    }

    
private:
    Tokenizer tok;
};

namespace SemanticAnalysis{
    bool failed{false};

#define SEMANTIC_ERROR(msg) \
    std::cerr << "Semantic Analysis error: " << msg << std::endl; \
    failed = true

    // decls only contains variable (name, isRegister), because ASTNodes have no copy constructor and using a unique_ptr<>& doesn't work for some unknown reason
    // feels like artemis man
    // quadratic in the number of variables/scopes (i.e. if both are arbitrarily large)
    void analyzeNode(ASTNode& node, std::vector<std::tuple<string,bool>> decls = {}) noexcept {
        //checks that declaratiosn happen before use
        if(node.type == ASTNode::Type::NFunction){
            // add params to decls
            for(auto& param : node.children[0]->children){
                decls.emplace_back(param->name, true); // parameters are considered register variables
            }
            analyzeNode(*node.children[1],decls);
        }else if(node.type == ASTNode::Type::NStmtBlock){
            // add local vars to decls
            std::vector<string> sameScopeDecls{};
            for(auto& stmt : node.children){
                if(stmt->type == ASTNode::Type::NStmtDecl){
                    //forbid same scope shadowing
                    if(std::find(sameScopeDecls.begin(), sameScopeDecls.end(), stmt->children[0]->name) != sameScopeDecls.end()){
                        SEMANTIC_ERROR("Variable \"" << stmt->children[0]->name << "\" was declared twice in the same scope");
                    }
                    // this makes it even more horribly inefficient (and quadratic in even more cases), but it fixes the overriding of variables, which was problematic before
                    // if id implement it again, i would do it with an unordered_map<string, ||some kind of list||> instead of a vector of tuples
                    std::erase_if(decls,[&stmt](auto& decl){
                        return std::get<0>(decl) == stmt->children[0]->name;
                    });
                    sameScopeDecls.emplace_back(stmt->children[0]->name);
                    decls.emplace_back(stmt->children[0]->name, stmt->op == Token::Type::KW_REGISTER);
                }

                try{
                    analyzeNode(*stmt,decls);
                }catch(std::runtime_error& e){
                    SEMANTIC_ERROR(e.what());
                }
            }
        }else if(node.type == ASTNode::Type::NExprVar){
            // check if var is declared
            bool found = false;
            for(auto& decl : decls){
                if(std::get<0>(decl) == node.name){
                    found = true;
                    break;
                }
            }
            if(!found){
                SEMANTIC_ERROR("Variable " + node.name + " used before declaration");
            }
        }else if((node.type == ASTNode::Type::NExprBinOp && node.op == Token::Type::ASSIGN) || (node.type == ASTNode::Type::NExprUnOp && node.op == Token::Type::AMPERSAND)){
            if(node.type == ASTNode::Type::NExprUnOp && node.children[0]->type == ASTNode::Type::NExprVar){
                // register variables and parameters are not permitted as operands to the unary addrof & operator
                // subscript is fine and thus left out here
                for(auto& decl : decls){
                    string declName;
                    bool isRegister;
                    std::tie(declName,isRegister) = decl;
                    if(isRegister && declName == node.children[0]->name){
                        SEMANTIC_ERROR("Cannot take the address of a register variable (or parameter)");
                    }
                }
            }
            // lhs/only child must be subscript or identifier
            if(node.children[0]->type != ASTNode::Type::NSubscript && node.children[0]->type != ASTNode::Type::NExprVar){
                SEMANTIC_ERROR("LHS of assignment/addrof must be a variable or subscript array access, got node which prints as: " << std::endl << node);
            }
            for(auto& child : node.children){
                analyzeNode(*child, decls);
            }
        }else{
            for(auto& child : node.children){
                analyzeNode(*child, decls);
            }
        }
    }
    
    void analyze(AST& ast){
        std::unordered_map<string, bool> decls{}; // maps name to isRegister(/parameter), uses contains to check in O(1), insert on average in O(1)
        analyzeNode(ast.root);
    }

}

namespace ArgParse{

    struct Arg{
        std::string shortOpt{""};
        std::string longOpt{""};
        uint32_t pos{0}; //if 0, no positional arg
        std::string description{""};
        bool required{false};
        bool flag{false};

        //define < operator, necessary for map
        bool operator<(const Arg& other) const{
            return shortOpt < other.shortOpt;
        }
    };

    // possible arguments
    const std::array<Arg,11> possible{{
        {"h", "help", 0, "Show this help message and exit",  false, true},
        {"i", "input", 1, "Input file",  true, false},
        {"d", "dot", 0, "Output AST in GraphViz DOT format (to stdout by default, or file using -o) (overrides -p)", false, true},
        {"o", "output", 2, "Output file for AST (requires -p)", false, false},
        {"p", "print", 0, "Print AST (-d for DOT format highly recommended instead)", false, true},
        {"E", "preprocess", 0, "Run the C preprocessor on the input file before parsing it", false, true},
        {"u", "url", 0, "Instead of printing the AST in DOT format to the console, print a URL to visualize it in the browser (requires -d or -p)", false, true},
        {"n", "nosemantic", 0, "Don't run semantic analysis on the AST", false, true},
        {"b", "benchmark", 0, "Time execution time for parsing and semantic analysis and print memory footprint", false, true},
        {"",  "iterations", 0, "Number of iterations to run the benchmark for (default 1, requires -b)", false, false},
        {"l",  "llvm", 0, "Output LLVM IR (mutually exclusive with p/d/u), by default to stdout, except if an output file is specified using -o", false, true},
    }};

    void printHelp(){
        std::cerr << "A Parser for a B like language" << std::endl;
        std::cerr << "Usage: " << std::endl;
        for(auto& arg:possible){
            std::cerr << "  ";
            if(arg.shortOpt != ""){
                std::cerr << "-" << arg.shortOpt;
            }
            if(arg.longOpt != ""){
                if(arg.shortOpt != ""){
                    std::cerr << ", ";
                }
                std::cerr << "--" << arg.longOpt;
            }
            if(arg.pos != 0){
                std::cerr << " (or positional, at position " << arg.pos << ")";
            }else if(arg.flag){
                std::cerr << " (flag)";
            }
            std::cerr << std::endl;
            std::cerr << "    " << arg.description << std::endl;
        }

        std::cerr << std::endl;
        std::cerr << "Example: " << std::endl;
        std::cerr << "  " << "main -i input.b -p -d -o output.dot" << std::endl;
        std::cerr << "  " << "main input.b -pd output.dot" << std::endl;
        std::cerr << "  " << "main input.b -pdu" << std::endl;
    }

    //unordered_map doesnt work because of hash reasons (i think), so just define <, use ordered
    std::map<Arg, std::string> parse(int argc, char *argv[]){
        std::stringstream ss;
        ss << " ";
        for (int i = 1; i < argc; ++i) {
            ss << argv[i] << " ";
        }

        string argString = ss.str();

        std::map<Arg, std::string> args;

        //handle positinoal args first, they have lower precedence
        //find them all, put them into a vector, then match them to the possible args
        uint32_t actualPositon = 1;
        std::vector<string> positionalArgs{};
        for(int i = 1; i < argc; ++i){
            for(const auto& arg : possible){
                if(!arg.flag && (("-"+arg.shortOpt) == string{argv[i-1]} || ("--"+arg.longOpt) == string{argv[i-1]})){
                    //the current arg is the value to another argument, so we dont count it
                    goto cont;
                }
            }


            if(argv[i][0] != '-'){
                // now we know its a positional arg
                positionalArgs.emplace_back(argv[i]);
                actualPositon++;
            }
cont:
            continue;
        }

        for(const auto& arg : possible){
            if(arg.pos != 0){
                //this is a positional arg
                if(positionalArgs.size() > arg.pos-1){
                    args[arg] = positionalArgs[arg.pos-1];
                }
            }
        }

        bool missingRequired = false;

        //long/short/flags
        for(const auto& arg : possible){
            if(!arg.flag){
                std::regex matchShort{" -"+arg.shortOpt+"\\s*([^\\s]+)"};
                std::regex matchLong{" --"+arg.longOpt+"(\\s*|=)([^\\s=]+)"};
                std::smatch match;
                if(arg.shortOpt!="" && std::regex_search(argString, match, matchShort)){
                    args[arg] = match[1];
                }else if(arg.longOpt!="" && std::regex_search(argString, match, matchLong)){
                    args[arg] = match[2];
                }else if(arg.required && !args.contains(arg)){
                    std::cerr << "Missing required argument: -" << arg.shortOpt << "/--" << arg.longOpt << std::endl;
                    missingRequired = true;
                }
            }else{
                std::regex matchFlagShort{" -[a-zA-z]*"+arg.shortOpt};
                std::regex matchFlagLong{" --"+arg.longOpt};
                if(std::regex_search(argString, matchFlagShort) || std::regex_search(argString, matchFlagLong)){
                    args[arg] = ""; //empty string for flags, will just be checked using .contains
                }
            };
        }

        if(missingRequired){
            printHelp();
            exit(EXIT_FAILURE);
        }
        return args;
    }

}

// taken from https://stackoverflow.com/a/17708801
string url_encode(const string &value) {
    std::ostringstream escaped;
    escaped.fill('0');
    escaped << std::hex;

    for (string::const_iterator i = value.begin(), n = value.end(); i != n; ++i) {
        string::value_type c = (*i);

        // Keep alphanumeric and other accepted characters intact
        if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
            escaped << c;
            continue;
        }

        // Any other characters are percent-encoded
        escaped << std::uppercase;
        escaped << '%' << std::setw(2) << int((unsigned char) c);
        escaped << std::nouppercase;
    }

    return escaped.str();
}

// ------------------------------------------------------------------------------------------------------
// HW 4 START
// Exceptions:
// - main, obviously
// - semantic analysis has been changed in order to determine the linkage of functions
// ------------------------------------------------------------------------------------------------------

namespace Codegen{
    void generate(AST& ast){
        //example code
        llvm::LLVMContext ctx;
        auto moduleUP = std::make_unique<llvm::Module>("mod", ctx);
        llvm::Type* i64 = llvm::Type::getInt64Ty(ctx);

        //llvm::FunctionType* fnTy = llvm::FunctionType::get(i64, {i64}, false);
        //llvm::Function* fn = llvm::Function::Create(fnTy,
        //llvm::GlobalValue::ExternalLinkage, "addOne", moduleUP.get());
        //llvm::BasicBlock* entryBB = llvm::BasicBlock::Create(ctx, "entry", fn);
        //llvm::IRBuilder<> irb(entryBB);
        //llvm::Value* add = irb.CreateAdd(fn->getArg(0), irb.getInt64(1));
        //irb.CreateRet(add);


        auto& children = ast.root.children;
        for(auto& child:children){
            switch(child->type){
                case ASTNode::Type::NFunction:
                    {
                        auto& paramlist = child->children[0];
                        auto& typelist = paramlist.map([](auto& _){return i64;}); //i hope this isn't horribly inefficient (it is, isn't it?) TODO maybe cache size 1-10 for these lists
                        llvm::FunctionType* fnTy = llvm::FunctionType::get(i64, typelist);
                        //llvm::Function* fn = llvm::Function::Create(fnTy,
                    }
                    break;
                case ASTNode::Type::NParamList:
                case ASTNode::Type::NStmtDecl:
                case ASTNode::Type::NStmtReturn:
                case ASTNode::Type::NStmtBlock:
                case ASTNode::Type::NStmtWhile:
                case ASTNode::Type::NStmtIf:
                case ASTNode::Type::NExpr:
                case ASTNode::Type::NExprVar:
                case ASTNode::Type::NExprNum:
                case ASTNode::Type::NExprCall:
                case ASTNode::Type::NExprUnOp:
                case ASTNode::Type::NExprBinOp:
                case ASTNode::Type::NSubscript:
                    break;
                    //impossible (hopfully :monkaGiga:) :
                case ASTNode::Type::NRoot:
                    break;
            }
        }

        // TODO change this to the correct output stream at the end
        //moduleUP->print(llvm::outs(), nullptr);
    }
}


int main(int argc, char *argv[])
{
    try{
        auto args = ArgParse::parse(argc, argv);

        if(args.contains(ArgParse::possible[0])){
            ArgParse::printHelp();
            return 0;
        }

        std::string inputFilename = args.at(ArgParse::possible[1]);
        int iterations = 1;
        if(args.contains(ArgParse::possible[9])){
            iterations = std::stoi(args.at(ArgParse::possible[9]));
        }

        if(access(inputFilename.c_str(),R_OK) != 0){
            perror("Could not open input file");
            return 1;
        }

        std::ifstream inputFile{inputFilename};
        if(args.contains(ArgParse::possible[5])){
            DEBUGLOG("running preprocessor");
            //this is a bit ugly, but it works
            std::stringstream ss;

            auto epochsecs = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count(); //cpp moment
            ss << "cpp -E -P " << args.at(ArgParse::possible[1]) << " > /tmp/" << epochsecs << ".bpreprocessed";
            system(ss.str().c_str());

            inputFile = std::ifstream{"/tmp/" + std::to_string(epochsecs) + ".bpreprocessed"};
        }

        Parser parser{inputFile};

        auto parseStart = std::chrono::high_resolution_clock::now();
        unique_ptr<AST> ast;
        for(int i = 0; i<iterations; i++){
            ast = parser.parse();
            parser.resetTokenizer();
        }
        auto parseEnd = std::chrono::high_resolution_clock::now();

        if(args.contains(ArgParse::possible[5])) system("rm /tmp/*.bpreprocessed");

        auto semanalyzeStart = std::chrono::high_resolution_clock::now();
        if(!args.contains(ArgParse::possible[7])){
            for(int i = 0; i<iterations; i++){
                SemanticAnalysis::failed = false;
                SemanticAnalysis::analyze(*ast);
            }
        }
        auto semanalyzeEnd = std::chrono::high_resolution_clock::now();

        if(parser.failed || SemanticAnalysis::failed){
            return EXIT_FAILURE;
        }

        if(args.contains(ArgParse::possible[4]) || args.contains(ArgParse::possible[2])){
            if(args.contains(ArgParse::possible[6])){
                std::stringstream ss;
                ast->printDOT(ss);
                auto compactSpacesRegex = std::regex("\\s+");
                auto str = std::regex_replace(ss.str(), compactSpacesRegex, " ");
                std::cout << "https://dreampuf.github.io/GraphvizOnline/#" << url_encode(str) << std::endl;
            }else if(args.contains(ArgParse::possible[2])){
                if(args.contains(ArgParse::possible[3])){
                    std::ofstream outputFile{args.at(ArgParse::possible[3])};
                    ast->printDOT(outputFile);
                    outputFile.close();
                }else{
                    ast->printDOT(std::cout);
                }
            }else{
                ast->print(std::cout);
            }
        }else if(args.contains(ArgParse::possible[10])){
            Codegen::generate(*ast);
            //TODO think about output method
        }
        //print execution times
        if(args.contains(ArgParse::possible[8])){
            std::cout << "Average parse time (over "              << iterations << " iterations): " << (1e-9*std::chrono::duration_cast<std::chrono::nanoseconds>(parseEnd - parseStart).count())/((double)iterations)           << "s"  << std::endl;
            std::cout << "Average semantic analysis time: (over " << iterations << " iterations): " << (1e-9*std::chrono::duration_cast<std::chrono::nanoseconds>(semanalyzeEnd - semanalyzeStart).count())/((double)iterations) << "s"  << std::endl;
            std::cout << "Memory usage: "                         << 1e-6*(ast->getRoughMemoryFootprint())                                                                                                                       << "MB" << std::endl;
        }
    }catch(std::exception& e){
        std::cerr << "Error (instance of '"<< typeid(e).name() << "'): " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
