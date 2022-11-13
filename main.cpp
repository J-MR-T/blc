#include <iostream>
#include <iostream>
#include <string>
#include <array>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <unordered_set>
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
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Verifier.h>
#pragma GCC diagnostic pop

using std::string;
using std::unique_ptr;

#ifndef NDEBUG
#define DEBUGLOG(x) std::cerr << x << std::endl; fflush(stderr);
#else
#define DEBUGLOG(x)
#endif

#define STRINGIZE(x) #x
#define STRINGIZE_MACRO(x) STRINGIZE(x)

#define THROW_TODO\
    throw std::runtime_error("TODO(Line " STRINGIZE_MACRO(__LINE__) "): Not implemented yet")

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
        AT,
        PLUS,
        MINUS, // can be unary or binary, depending on context
        TIMES,
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
            case Type::AT:
                return "@";
            case Type::PLUS:
                return "+";
            case Type::MINUS:
                return "-";
            case Type::TIMES:
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
                    type = Token::Type::TIMES;
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
        NExprVar, // possible children: none, name: yes, op: KW_AUTO/KW_REGISTER
        NExprNum, // possible children: none, value: yes
        NExprCall, // possible children: expr*, name: yes
        NExprUnOp, // possible children: expr, op: yes (MINUS/TILDE/AMPERSAND/LOGICAL_NOT)
        NExprBinOp, // possible children: expr, expr, op: yes (all the binary operators possible)
        NSubscript, // possible children: expr(addr), expr(index), num (sizespec, 1/2/4/8)
    };

    class Hash{
        public:
        size_t operator()(const ASTNode& node) const{
            return std::hash<string>()(node.uniqueDotIdentifier);
        }
    };

    Type type;

    string uniqueDotIdentifier;

    static const std::unordered_map<Type, string> nodeTypeToDotIdentifier; // initialization below
    static const std::unordered_map<Type, std::vector<string>> nodeTypeToDotStyling; // initialization below

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
            if(type == Type::NExprVar && op != Token::Type::EMPTY){
                return  nodeTypeToDotIdentifier.at(type) + "(" + Token::toString(op) + ")" + ": " + name;
            }else{
                return nodeTypeToDotIdentifier.at(type) + ": " + name;
            }
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
    {Token::Type::TIMES,            TUP(12, false)},
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
            decl->children.push_back(std::make_unique<ASTNode>(ASTNode::Type::NExprVar, varName.data(), registerOrAuto));

            DEBUGLOG("for declaration of " << decl->children[0]->name << ": decl op: " << Token::toString(decl->op) << ", var op: " << Token::toString(decl->children[0]->op));

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
                            paramlist->children.emplace_back(std::make_unique<ASTNode>(ASTNode::Type::NExprVar, paramname.data(), Token::Type::KW_REGISTER /* params are always registers */));
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
    std::unordered_map<string, unsigned int> externalFunctionsToNumParams{};
    std::unordered_set<string> declaredFunctions{};

#define SEMANTIC_ERROR(msg) \
    std::cerr << "Semantic Analysis error: " << msg << std::endl; \
    failed = true

    // decls only contains variable (name, isRegister), because ASTNodes have no copy constructor and using a unique_ptr<>& doesn't work for some unknown reason
    // feels like artemis man
    // quadratic in the number of variables/scopes (i.e. if both are arbitrarily large)
    void analyzeNode(ASTNode& node, std::vector<std::tuple<string,bool>> decls = {}) noexcept {
        //checks that declaratiosn happen before use
        if(node.type == ASTNode::Type::NFunction){
            // add to declared, remove from external
            declaredFunctions.insert(node.name);
            externalFunctionsToNumParams.erase(node.name);

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
        }else if(node.type == ASTNode::Type::NExprCall){
            if(!declaredFunctions.contains(node.name)){
                if(externalFunctionsToNumParams.contains(node.name)){
                    if(externalFunctionsToNumParams[node.name] != node.children.size()){
                        SEMANTIC_ERROR("Function \"" << node.name << "\" was once called with " << externalFunctionsToNumParams[node.name] << " arguments, but was now called with " << node.children.size() << " arguments");
                    }
                }else{
                    externalFunctionsToNumParams[node.name] = node.children.size();
                }
            }
            for(auto& arg : node.children){
                analyzeNode(*arg,decls);
            }
        }else if(node.type == ASTNode::Type::NExprVar){
            // check if var is declared
            bool found = false;
            for(auto& decl : decls){
                auto& [declName, isRegister] = decl;
                if(declName == node.name){
                    found = true;
                    // add info about if its register/auto to node
                    node.op = isRegister?Token::Type::KW_REGISTER:Token::Type::KW_AUTO;
                    DEBUGLOG("found var \"" << node.name << "\", added op: " << Token::toString(node.op));
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
        analyzeNode(ast.root);
    }

    void reset(){
        failed = false;
        declaredFunctions.clear();
        externalFunctionsToNumParams.clear();
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
// - semantic analysis has been changed in order to determine external functions
// - other minor changes in the parser
// ------------------------------------------------------------------------------------------------------

namespace Codegen{
    llvm::LLVMContext ctx;
    auto moduleUP = std::make_unique<llvm::Module>("mod", ctx);
    llvm::Type* i64 = llvm::Type::getInt64Ty(ctx);
    llvm::Function* currentFunction = nullptr;

    struct BasicBlockInfo{
        bool sealed{false};
        std::vector<llvm::BasicBlock*> parents{};
        std::unordered_map<string, llvm::Value*> varmap{};
    };

    std::unordered_map<llvm::BasicBlock*, BasicBlockInfo> blockInfo;

    // TODO: can the fact that this is cached in the "earlier" varmaps lead to cache invalidation problems (someone please quote Phil Karlton)?
    //       because of this potential problem, caching has been disabled for now, TODO enable it again after this has been thought through
    // automatically creates phi nodes on demand
    llvm::Value*& varmapLookup(llvm::BasicBlock* block, string& name) /* TODO add this as soon as everything is implemented: noexcept */{
        auto& [sealed, parents, varmap] = blockInfo[block];
        if(varmap.contains(name)){
            return varmap[name];
        }else{
            if(sealed){
                // cache it, so we don't have to look it up every time
                if(parents.size() == 1){
                    return varmap[name] = varmapLookup(parents[0], name);
                }else if(parents.size() > 1){
                    // create phi node to merge it
                    llvm::IRBuilder<> irb(block);
                    auto nonphi = block->getFirstNonPHI();
                    // if there is no nonphi node, we can just insert at the end, which should be where the irb starts
                    if(nonphi!=nullptr){
                        irb.SetInsertPoint(nonphi); // insertion is before the instruction, so this is the correct position
                    }
                    llvm::PHINode* phi = irb.CreatePHI(i64, parents.size(), name); //num reserved values here is only a hint, 0 is fine "[...] if you really have no idea", it's at least one because of our algo
                    
                    // block is sealed -> we have all the information -> we can add all the incoming values
                    for(auto& parent: parents){
                        phi->addIncoming(varmapLookup(parent, name),parent);
                    }
                    return varmap[name] = phi;
                }else{
                    //TODO think about this case again
                    throw std::runtime_error("Sealed block without parents was queried for undefined variable");
                }
            }else{
                // we need a phi node in this case
                // TODO maybe use another value for num reserved values here, if possible
                // can there be multiple IRBs? like this?:
                llvm::IRBuilder<> irb(block);
                auto nonphi = block->getFirstNonPHI();
                // if there is no nonphi node, we can just insert at the end, which should be where the irb starts
                if(nonphi!=nullptr){
                    irb.SetInsertPoint(nonphi); // insertion is before the instruction, so this is the correct position
                }
                llvm::PHINode* phi = irb.CreatePHI(i64, parents.size(), name); //num reserved values here is only a hint, 0 is fine "[...] if you really have no idea", it's at least one because of our algo
                for(unsigned i = 0; i < parents.size(); ++i){
                    phi->setIncomingBlock(i, parents[i]);
                }
                return varmap[name] = phi;
            }
        }
    }

    llvm::Value*& varmapLookup(llvm::BasicBlock* block, llvm::StringRef name) /* TODO add this as soon as everything is implemented: noexcept */{
        string stringName{name.str()}; // TODO is it okay that the lifetime of this is just this function? Should be, right? Because the map saves the key independently anyway and for the time of the lookup, the string is still alive
        return varmapLookup(block, stringName); // for whatever reason, using name.str() directly resuts in it calling itself...
    }

    //just for convenience
    inline llvm::Value*& updateVarmap(llvm::BasicBlock* block, string& name, llvm::Value* val){
        auto& [sealed, parents, varmap] = blockInfo[block];
        return varmap[name] = val;
    }
    inline llvm::Value*& updateVarmap(llvm::IRBuilder<>& irb, string& name, llvm::Value* val){
        return updateVarmap(irb.GetInsertBlock(), name, val);
    }

    // fills phi nodes with correct values, assumes block is sealed
    inline void fillPHIs(llvm::BasicBlock* block){
        auto& [sealed, parents, varmap] = blockInfo[block];
        for(auto& phi: block->phis()){
            for(unsigned i = 0; i < parents.size(); i++){
                phi.setIncomingValue(i, varmapLookup(parents[i], phi.getName()));
            }
        }
    }
    
    // Seals the block and fills phis
    inline void sealBlock(llvm::BasicBlock* block){
        auto& [sealed, parents, varmap] = blockInfo[block];
        sealed = true;
        fillPHIs(block);
    }

#define PARLIST(x) std::vector<llvm::Type*>(x, i64),


    llvm::Type* sizespecToLLVMType(ASTNode& sizespecNode, llvm::IRBuilder<>& irb){
        // TODO i hope the addr here is right
        // TODO is there any endianness problem here?
        auto sizespecInt = sizespecNode.value;
        llvm::Type* type;
        if(sizespecInt == 1){
            type = irb.getInt8Ty();
        }else if(sizespecInt == 2){
            type = irb.getInt16Ty();
        }else if(sizespecInt == 4){
            type = irb.getInt32Ty();
        }else if(sizespecInt == 8){
            type = irb.getInt64Ty();
        }else{
            throw std::runtime_error("Something has gone seriously wrong here, got a sizespec of " + std::to_string(sizespecInt) + " bytes");
        }
        return type;
    }


    llvm::Value* genExpr(ASTNode& exprNode, llvm::IRBuilder<>& irb){
        switch(exprNode.type){
            case ASTNode::Type::NExprVar:
                // TODO do different things depending on if its a register/auto var
                if(exprNode.op == Token::Type::KW_REGISTER){
                    // this should be what we want for register vars, for auto vars we aditionally need to look up the alloca (and store it back if its an assignment, see the assignment below)
                    return varmapLookup(irb.GetInsertBlock(), exprNode.name); // NOTE to self: this does work even though there is a pointer to the value in the varmap, because if the mapping gets updated, that whole pointer isn't the value anymore, nothing is changed about what it's pointing to.
                }else if(exprNode.op == Token::Type::KW_AUTO){
                    return irb.CreateLoad(i64, varmapLookup(irb.GetInsertBlock(), exprNode.name), exprNode.name);
                }else{
                    throw std::runtime_error("Something has gone seriously wrong here, got a variable node wich is neither register nor auto, its is " + exprNode.toString());
                }

            case ASTNode::Type::NExprNum:
                return llvm::ConstantInt::get(i64, exprNode.value);
            case ASTNode::Type::NExprCall:
                {
                    // TODO when i create this, does it copy the args vector? Because its lifetime is limited to this function, so if it doesnt, thats a problem
                    std::vector<llvm::Value*> args(exprNode.children.size());
                    for(unsigned int i = 0; i < exprNode.children.size(); ++i){
                        args[i] = genExpr(*exprNode.children[i], irb);
                    }
                    return irb.CreateCall(currentFunction, args);
                }
            case ASTNode::Type::NExprUnOp:
                {
                    auto& operandNode = *exprNode.children[0];
                    auto operand = genExpr(operandNode, irb);
                    switch(exprNode.op){
                        //can be TILDE, MINUS, AMPERSAND, LOGICAL_NOT
                        case Token::Type::TILDE:
                            return irb.CreateNot(operand); //i hope this is the right kind of not
                            break;
                        case Token::Type::MINUS:
                            return irb.CreateNeg(operand);
                        case Token::Type::LOGICAL_NOT:
                            {
                                // this seems incredibly stupid, i hope its fine
                                // TODO should this be an i1? or do we just keep to i64 all the way through?
                                //  after a bit of thinking: I think it's more adivsable to keep everything an i64 until it's absolutely necessary to convert it to an i1. Allows the operations that the language defines etc.
                                auto cmp = irb.CreateICmp(llvm::CmpInst::Predicate::ICMP_EQ, operand, irb.getInt64(0));
                                return irb.CreateSelect(cmp, irb.getInt1(true), irb.getInt1(false));
                            }
                        case Token::Type::AMPERSAND:
                            //TODO
                            // get the alloca
                            // then somehow get the address from that/ a ptr type obj pointing to it
                            throw std::runtime_error("Not implemented yet");
                        default:
                            throw std::runtime_error("Something has gone seriously wrong here, got a " + Token::toString(exprNode.op) + " as unary operator");
                    }
                }
            case ASTNode::Type::NExprBinOp:
                {
                    auto& lhsNode = *exprNode.children[0];
                    auto& rhsNode = *exprNode.children[1];
                    // assignment needs special handling:
                    // TODO before this switch and CRUCIALLY (!!!) before the lhs gets evaluated, check if exprNode.op is an assign and if the left hand side is a subscript. in that case, we need to generate a store instruction for the assignment
                    //  we also need to generate a store, if the lhs is an auto variable
                    auto rhs = genExpr(rhsNode, irb);
                    if(exprNode.op == Token::Type::ASSIGN){
                        if(lhsNode.type == ASTNode::Type::NSubscript){
                            auto addr = genExpr(*lhsNode.children[0], irb); // TODO I think this has to be cast to a ptr
                                                        // TODO what if this is an auto variable? then genExpr returns the load instruction, not sure if that's right
                            auto index = genExpr(*lhsNode.children[1], irb);
                            auto& sizespecNode = *lhsNode.children[2];

                            llvm::Type* type = sizespecToLLVMType(sizespecNode, irb);
                            // first cast, then store, so that the right amount is stored
                            // this could be done with a trunc, but that is only allowed if the type is strictly smaller, the CreateIntCast distinguishes these cases and takes care of it for us
                            auto cast = irb.CreateIntCast(rhs, type, true);
                            auto getelementpointer = irb.CreateGEP(type, addr, {index});
                            irb.CreateStore(cast, getelementpointer);

                            return rhs; // just as before, return the result, not the store
                        }else if(/* lhs node has to be var if we're here */ lhsNode.op == Token::Type::KW_AUTO){
                            irb.CreateStore(rhs, varmapLookup(irb.GetInsertBlock(), lhsNode.name));
                            return rhs; // just as before, return the result, not the store
                        }
                    }
                    auto lhs = genExpr(lhsNode, irb);

                    // for all cases except the assign this is a post order traversal of the epxr tree

                    switch(exprNode.op){
                        case Token::Type::BITWISE_OR:
                            return irb.CreateOr(lhs,rhs);
                        case Token::Type::BITWISE_XOR:
                            return irb.CreateXor(lhs,rhs);
                        case Token::Type::AMPERSAND:
                            return irb.CreateAnd(lhs,rhs);
                        case Token::Type::PLUS:
                            return irb.CreateAdd(lhs,rhs);
                        case Token::Type::MINUS:
                            return irb.CreateSub(lhs,rhs);
                        case Token::Type::TIMES:
                            return irb.CreateMul(lhs,rhs);
                        case Token::Type::DIV:
                            return irb.CreateSDiv(lhs,rhs);
                        case Token::Type::MOD:
                            return irb.CreateSRem(lhs,rhs);
                        case Token::Type::SHIFTL:
                            return irb.CreateShl(lhs,rhs);
                        case Token::Type::SHIFTR:
                            return irb.CreateAShr(lhs,rhs);
                        case Token::Type::LESS:
                            return irb.CreateICmp(llvm::CmpInst::ICMP_SLT, lhs, rhs);
                        case Token::Type::GREATER:
                            return irb.CreateICmp(llvm::CmpInst::ICMP_SGT, lhs, rhs);
                        case Token::Type::LESS_EQUAL:
                            return irb.CreateICmp(llvm::CmpInst::ICMP_SLE, lhs, rhs);
                        case Token::Type::GREATER_EQUAL:
                            return irb.CreateICmp(llvm::CmpInst::ICMP_SGE, lhs, rhs);
                        case Token::Type::EQUAL:
                            return irb.CreateICmp(llvm::CmpInst::ICMP_EQ, lhs, rhs);
                        case Token::Type::NOT_EQUAL:
                            return irb.CreateICmp(llvm::CmpInst::ICMP_NE, lhs, rhs);
                        case Token::Type::LOGICAL_AND:
                            // TODO this is kinda right, but also kinda not: These instrs expect an i1 for obvious reasons, but we have i64s, so we need to convert them here
                            // TODO
                            return irb.CreateLogicalAnd(lhs,rhs); //i have no idea how this works, cant find a 'logical and' instruction...
                        case Token::Type::LOGICAL_OR:
                            return irb.CreateLogicalOr(lhs,rhs); //i have no idea how this works, cant find a 'logical and' instruction...
                        case Token::Type::ASSIGN:
                            {
                                // in lhs: "old" varname of the var we're assigning to -> update mapping
                                // in rhs: value to assign to it
                                updateVarmap(irb, lhsNode.name, rhs);

                                // TODO is this enough? NO! READ ABOVE!
                                // TODO additionally: if the lhs is an auto var, we also need to generate a store instruction for it

                                // return rhs
                                return rhs;
                            }
                            break;
                        default:
                            throw std::runtime_error("Something has gone seriously wrong here, got a " + Token::toString(exprNode.op) + " as binary operator");
                    }
                }
            case ASTNode::Type::NSubscript:
                {
                    //this can *ONLY* be a "load" (getelementpointer) subscript, store has been handled in the special case for assignments above
                    auto& addrNode = *exprNode.children[0];
                    auto& indexNode = *exprNode.children[1];
                    auto& sizespecNode = *exprNode.children[2];

                    auto addr = genExpr(addrNode, irb); // TODO I think this has to be cast to a ptr
                                                        // TODO what if this is an auto variable? then genExpr returns the load instruction, not sure if that's right
                    auto index = genExpr(indexNode, irb);

                    llvm::Type* type = sizespecToLLVMType(sizespecNode, irb);
                    auto getelementpointer = irb.CreateGEP(type, addr, {index});
                    auto load = irb.CreateLoad(type, getelementpointer);

                    //we only have i64s, thus we need to convert our extracted value back to an i64
                    // after reading up on IntCast vs SExt/Trunc (in the source code... Why can't they just document this stuff properly?), it seems that CreateIntCast is a wrapper around CreateSExt/CreateZExt, but in this case we know exactly what we need, so I think CreateSExt would be fine, except that is only allowed if the type is strictly larger, the CreateIntCast distinguishes these cases and takes care of it for us
                    //auto castResult = irb.CreateSExt(load, i64);
                    auto castResult = irb.CreateIntCast(load, i64, true);

                    return castResult;
                }
            default:
                throw std::runtime_error("Something has gone seriously wrong here");
        }
    }

    void genStmts(std::vector<unique_ptr<ASTNode>>& stmts, llvm::IRBuilder<>& irb);

    // I think this doesn't need a return type
    void genStmt(ASTNode& stmtNode, llvm::IRBuilder<>& irb){
        switch(stmtNode.type){
            case ASTNode::Type::NStmtDecl:
                {
                    auto initializer = genExpr(*stmtNode.children[1], irb);
                    initializer->setName(stmtNode.children[0]->name); // TODO i hope this doesn't hurt performance, but it wouldn't make sense if it did
                    if(stmtNode.op == Token::Type::KW_AUTO){
                        // TODO i have to basically use a load/store every time i use this, right?
                        //  if so: TODO do that
                        llvm::IRBuilder<> entryIRB(currentFunction->getEntryBlock().getFirstNonPHI()); // i hope this is correct
                        auto alloca = entryIRB.CreateAlloca(i64); // this returns the ptr to the alloca'd memory
                        irb.CreateStore(initializer, alloca);

                        THROW_TODO; // TODO because of the load/store thing
                                    // TODO think about the next thing: Does this varmapping actually work?
                        blockInfo[irb.GetInsertBlock()].varmap[stmtNode.children[0]->name] = alloca; // we actually want the alloca'd memory, not the initializer
                    }else if(stmtNode.op == Token::Type::KW_REGISTER){
                        updateVarmap(irb, stmtNode.children[0]->name, initializer);
                    }else{
                        throw std::runtime_error("Something has gone seriously wrong here, got a " + Token::toString(stmtNode.op) + " as decl type");
                    }
                }
                break;
            case ASTNode::Type::NStmtReturn:
                irb.CreateRet(genExpr(*stmtNode.children[0], irb));
                break;
            case ASTNode::Type::NStmtBlock:
                // TODO how do we handle new scopes? Can't just make a basic block for it, right?
                //  possible answer: I think we don't really. I think its fine to simply parse it as it is, and keep it in the same BasicBlock, var declarations get overriden anyway and are already checked to be semantically valid
                genStmts(stmtNode.children, irb);
                break;
            case ASTNode::Type::NStmtIf:
                {
                    bool hasElse = stmtNode.children.size() == 3;

                    llvm::BasicBlock* then =         llvm::BasicBlock::Create(ctx, "then", currentFunction); // TODO can this name be "then"? or can this clash?
                    llvm::BasicBlock* els  = hasElse?llvm::BasicBlock::Create(ctx, "else", currentFunction): nullptr; // its generated this way around, so that the cont block is always after the else block
                    llvm::BasicBlock* cont =         llvm::BasicBlock::Create(ctx, "cont", currentFunction);
                    if(!hasElse){
                        els = cont;
                    }

                    auto condition = genExpr(*stmtNode.children[0], irb);
                    irb.CreateCondBr(condition, then, els);
                    // block is now finished
                    
                    auto& [thenSealed, thenParents, thenVarmap] = blockInfo[then];
                    thenSealed = true;
                    thenParents = {irb.GetInsertBlock()};
                    // var map is queried recursively anyway, would be a waste to copy it here

                    llvm::IRBuilder<> thenIRB(then);
                    genStmt(*stmtNode.children[1], thenIRB);
                    bool thenBranchesToCont = !(thenIRB.GetInsertBlock()->getTerminator());
                    if(thenBranchesToCont){
                        thenIRB.CreateBr(cont);
                    }
                    // now if is generated -> we can seal else

                    auto& [elseSealed, elseParents, elseVarmap] = blockInfo[els];
                    elseSealed = true; // if this is cont: then it's sealed. If this is else, then it's sealed too (but then cont is not sealed yet!).
                    if(hasElse){
                        elseParents = {irb.GetInsertBlock()};

                        llvm::IRBuilder<> elseIRB(els);
                        genStmt(*stmtNode.children[2], elseIRB);

                        bool elseBranchesToCont = !(elseIRB.GetInsertBlock()->getTerminator());
                        if(elseBranchesToCont){
                            elseIRB.CreateBr(cont);
                        }

                        auto& [_0, contParents, _1] = blockInfo[cont];
                        if(thenBranchesToCont) contParents.push_back(then);
                        if(elseBranchesToCont) contParents.push_back(els);
                    }else{
                        auto& contParents = elseParents; // purely for readability
                        contParents = {irb.GetInsertBlock()};
                        if(thenBranchesToCont) contParents.push_back(then);
                    }

                    sealBlock(cont);

                    // now that stuff is sealed, fill phi nodes

                    fillPHIs(then);
                    fillPHIs(els);

                    // now that we've generated the if, we can 'preceed as before' in the parent call, so just set the irb to the right place
                    irb.SetInsertPoint(cont); 
                }
                break;
            case ASTNode::Type::NStmtWhile:
                THROW_TODO; // TODO
                break;

            case ASTNode::Type::NExprVar:
            case ASTNode::Type::NExprNum:
            case ASTNode::Type::NExprCall:
            case ASTNode::Type::NExprUnOp:
            case ASTNode::Type::NExprBinOp:
            case ASTNode::Type::NSubscript:
                // this can lead to empty blocks. because assignments don't necessarily really generate instructions, if the only thing in the block is an assignment, it can be empty. We need to somehow forefully insert the instruction

                irb.CreateIntrinsic(llvm::Intrinsic::donothing, {}, {}); // TODO this looks very hacky currently, but inserting the generated expression doesn't work either
                DEBUGLOG("Inserted donothing intrinsic to prevent empty blocks, did it work: " << !irb.GetInsertBlock()->empty());
                // inserted manually here, so assignments actually fill blocks with stuff
                // TODO this seems kinda wrong
                //irb.Insert(
                        genExpr(stmtNode, irb)
                        //)
                        ; 
                break;


                //hopefully impossible
            default:
                throw std::runtime_error("Something has gone seriously wrong here" STRINGIZE_MACRO(__LINE__));
        }
    }

    void genStmts(std::vector<unique_ptr<ASTNode>>& stmts, llvm::IRBuilder<>& irb){
        for(auto& stmt : stmts){
            genStmt(*stmt, irb);
            if(stmt->type == ASTNode::Type::NStmtReturn){
                // stop the generation for this block
                break;
            }
        }
    }

    void genFunction(ASTNode& fnNode){
        auto paramNum = fnNode.children[0]->children.size();
        auto typelist = llvm::SmallVector<llvm::Type*, 8>(paramNum, i64);
        llvm::FunctionType* fnTy = llvm::FunctionType::get(i64, typelist, false);
        llvm::Function* fn = llvm::Function::Create(fnTy, llvm::GlobalValue::PrivateLinkage, fnNode.name, moduleUP.get()); //TODO is private linkage correct?
        currentFunction = fn;
        llvm::BasicBlock* entryBB = llvm::BasicBlock::Create(ctx, "entry", fn); // is this fine? do the labels have to have unique names only inside the function or inside the module?
        // simply use getEntryBlock() on the fn when declarations need to be added
        blockInfo[entryBB].sealed = true;
        llvm::IRBuilder<> irb(entryBB);
        for(unsigned int i = 0; i < paramNum; i++){
            llvm::Argument* arg = fn->getArg(i);
            auto& name = fnNode.children[0]->children[i]->name;
            arg->setName(name);
            updateVarmap(irb, name, arg);

        }

        auto& blockNode = *fnNode.children[1];
        genStmts(blockNode.children, irb);

        auto insertBlock = irb.GetInsertBlock();
        if(insertBlock->hasNUses(0) && insertBlock!=&currentFunction->getEntryBlock()){
            // if the block is unused (for example unreachable because of return statements in all theoretical predecessors), we discard it
            insertBlock->eraseFromParent(); // unlink and delete block
        }else if(insertBlock->empty() || (insertBlock->getTerminator() == nullptr)){
            // if the block is empty, or otherwise doesn't have a terminator, we need to either delete the block (happend in the case where its not used, above), or add one, this case:
            // This should only be possible if the block is the last block in the function anyway, all other blocks should always branch or return
            // in this case the block is reachable, but either empty or without terminator
            // I tried to find it in the C standard, but I couldn't find any defined behavior about functions with return types not returning, undefined behavior galore
            // clang simply inserts an unreachable here (even though it is totally reachable, we have in fact proven at this point, that if its empty, it has at least has >0 uses, and otherwise it should always have uses), so that's what we'll do too
            irb.CreateUnreachable();
        }
    }

    void genRoot(ASTNode& root){
        for(auto& [fnName, fnParamCount]: SemanticAnalysis::externalFunctionsToNumParams){
            std::vector<llvm::Type*> params(fnParamCount, i64);
            llvm::FunctionType* fnTy = llvm::FunctionType::get(i64, params, false);
            llvm::Function::Create(fnTy, llvm::GlobalValue::ExternalLinkage, fnName, moduleUP.get()); // TODO is this enough for external functions?
        }

        auto& children = root.children;

        for(auto& child:children){
            genFunction(*child);
        }
    }

    bool generate(AST& ast){
        //example code

        //llvm::FunctionType* fnTy = llvm::FunctionType::get(i64, {i64}, false);
        //llvm::Function* fn = llvm::Function::Create(fnTy,
        //llvm::GlobalValue::ExternalLinkage, "addOne", moduleUP.get());
        //llvm::BasicBlock* entryBB = llvm::BasicBlock::Create(ctx, "entry", fn);
        //llvm::IRBuilder<> irb(entryBB);
        //llvm::Value* add = irb.CreateAdd(fn->getArg(0), irb.getInt64(1));
        //
        //llvm::IRBuilder<> irb2(entryBB); // to see where the irb starts inserting
        //auto maybenull = &*entryBB->getInstList().begin();
        //if(maybenull == nullptr){
        //    std::cout << "null" << std::endl;
        //    exit(1);
        //}
        //irb2.SetInsertPoint(maybenull);
        //// much easier alternative:
        //irb2.SetInsertPoint(entryBB, entryBB->getFirstInsertionPt());
        //irb2.CreateRet(fn->getArg(0));


        genRoot(ast.root);

        //for(auto& fn : moduleUP->getFunctionList()){
        //    bool isErronous = llvm::verifyFunction(fn, &llvm::errs());
        //    llvm::errs() << "Function " << fn.getName() << " is " << (isErronous ? "erronous" : "fine") << "
        //}

        bool moduleIsBroken = llvm::verifyModule(*moduleUP, &llvm::errs());
        if(moduleIsBroken){
            moduleUP->print(llvm::errs(), nullptr);
        }else{
            // TODO change this to the correct output stream at the end
            moduleUP->print(llvm::outs(), nullptr);
        }
        return !moduleIsBroken;
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
                SemanticAnalysis::reset();
                SemanticAnalysis::analyze(*ast);
            }
        }
        auto semanalyzeEnd = std::chrono::high_resolution_clock::now();

        if(parser.failed || SemanticAnalysis::failed){
            return EXIT_FAILURE;
        }

        bool genSuccess = false;

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
            if(!(genSuccess = Codegen::generate(*ast))){
                std::cout << "Codegen failed, errors displayed above" << std::endl;
            }
            //TODO think about output method
        }
        //print execution times
        if(args.contains(ArgParse::possible[8])){
            std::cout << "Average parse time (over "              << iterations << " iterations): " << (1e-9*std::chrono::duration_cast<std::chrono::nanoseconds>(parseEnd - parseStart).count())/((double)iterations)           << "s"  << std::endl;
            std::cout << "Average semantic analysis time: (over " << iterations << " iterations): " << (1e-9*std::chrono::duration_cast<std::chrono::nanoseconds>(semanalyzeEnd - semanalyzeStart).count())/((double)iterations) << "s"  << std::endl;
            std::cout << "Memory usage: "                         << 1e-6*(ast->getRoughMemoryFootprint())                                                                                                                       << "MB" << std::endl;
        }
        if(!genSuccess) return EXIT_FAILURE;
    }catch(std::exception& e){
        std::cerr << "Error (instance of '"<< typeid(e).name() << "'): " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
