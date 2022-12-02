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
#include <queue>
#pragma GCC diagnostic push 
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Verifier.h>
#pragma GCC diagnostic pop

using std::string;
using std::unique_ptr;

#ifndef NDEBUG
#define DEBUGLOG(x) llvm::errs() << x << "\n"; fflush(stderr);
#else
#define DEBUGLOG(x)
#endif

#define STRINGIZE(x) #x
#define STRINGIZE_MACRO(x) STRINGIZE(x)

#define THROW_TODO\
    throw std::runtime_error("TODO(Line " STRINGIZE_MACRO(__LINE__) "): Not implemented yet")


// search for "HW 6" to find the start of the new code


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

    void assertNotToken(Token::Type type, bool advance = true){
        if(matchToken(type, advance)){
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
        NExprSubscript, // possible children: expr(addr), expr(index), num (sizespec, 1/2/4/8)
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
    {ASTNode::Type::NExprSubscript,   "Subscript"},
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
            tok.assertToken(Token::Type::SEMICOLON);

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
            try{
                num->value = std::stoll(tok.matched.value);;
            }catch(std::out_of_range& e){
                num->value = 0;
                std::cerr << "Line " << tok.getLineNum() << ": Warning: number " << tok.matched.value << " is out of range and will be truncated to 0" << std::endl;
            }
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
                while(!tok.matchToken(Token::Type::R_PAREN)){
                    call->children.emplace_back(parseExpr());
                    if(tok.matchToken(Token::Type::COMMA)){
                        tok.assertNotToken(Token::Type::R_PAREN, false);
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

                auto subscript = std::make_unique<ASTNode>(ASTNode::Type::NExprSubscript);
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
    std::unordered_map<string, int> externalFunctionsToNumParams{};
#define EXTERNAL_FUNCTION_VARARGS -1

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
                    // right side needs to be evaluated first (with current decls!), then left side can be annotated
                    analyzeNode(*stmt->children[1],decls);

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

                    analyzeNode(*stmt->children[0],decls);
                }else{
                    try{
                        analyzeNode(*stmt,decls);
                    }catch(std::runtime_error& e){
                        SEMANTIC_ERROR(e.what());
                    }
                }
            }
        }else if(node.type == ASTNode::Type::NExprCall){
            if(!declaredFunctions.contains(node.name)){
                if(externalFunctionsToNumParams.contains(node.name)){
                    if(externalFunctionsToNumParams[node.name] !=  static_cast<int>(node.children.size())){
                        // we seem to have ourselves a vararg function we don't know anything about, so indicate that by setting the number of params to -1
                        externalFunctionsToNumParams[node.name] = EXTERNAL_FUNCTION_VARARGS;
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
            if(node.children[0]->type != ASTNode::Type::NExprSubscript && node.children[0]->type != ASTNode::Type::NExprVar){
                SEMANTIC_ERROR("LHS of assignment/addrof must be a variable or subscript array access, got node which prints as: " << std::endl << node);
            }
            for(auto& child : node.children){
                analyzeNode(*child, decls);
            }
        }else if(node.type == ASTNode::Type::NStmtIf){
            // forbid declarations as sole stmt of if/else
            if(node.children[1]->type == ASTNode::Type::NStmtDecl || (node.children.size() > 2 && node.children[2]->type == ASTNode::Type::NStmtDecl)){
                SEMANTIC_ERROR("Declarations are not allowed as the sole statement in if/else");
            }
            for(auto& child : node.children){
                analyzeNode(*child, decls);
            }
        }else if(node.type == ASTNode::Type::NStmtWhile){
            // forbid declarations as sole stmt of while
            if(node.children[1]->type == ASTNode::Type::NStmtDecl){
                SEMANTIC_ERROR("Declarations are not allowed as the sole statement in while");
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

    std::map<Arg, std::string> parsedArgs{};
    
    // struct for all possible arguments
    const struct {
        const Arg help{"h", "help", 0, "Show this help message and exit",  false, true};
        const Arg input{"i", "input", 1, "Input file",  true, false};
        const Arg dot{"d", "dot", 0, "Output AST in GraphViz DOT format (to stdout by default, or file using -o) (overrides -p)", false, true};
        const Arg output{"o", "output", 2, "Output file for AST (requires -p)", false, false};
        const Arg print{"p", "print", 0, "Print AST (-d for DOT format highly recommended instead)", false, true};
        const Arg preprocess{"E", "preprocess", 0, "Run the C preprocessor on the input file before parsing it", false, true};
        const Arg url{"u", "url", 0, "Instead of printing the AST in DOT format to the console, print a URL to visualize it in the browser (requires -d or -p)", false, true};
        const Arg nosemantic{"n", "nosemantic", 0, "Don't run semantic analysis on the AST", false, true};
        const Arg benchmark{"b", "benchmark", 0, "Time execution time for parsing and semantic analysis and print memory footprint", false, true};
        const Arg iterations{"",  "iterations", 0, "Number of iterations to run the benchmark for (default 1, requires -b)", false, false};
        const Arg llvm{"l", "llvm", 0, "Output LLVM IR (mutually exclusive with p/d/u), by default to stdout, except if an output file is specified using -o", false, true};
        const Arg nowarn{"w", "nowarn", 0, "Do not generate warnings during the LLVM codegeneration phase (has no effect unless -l is specified)", false, true};
        const Arg isel{"s", "isel", 0, "Run the custom ARM instruction selector (has no effect unless -l is specified)", false, true};


        const Arg sentinel{"", "", 0, "", false, false};

        const Arg* const all[14] = {&help, &input, &dot, &output, &print, &preprocess, &url, &nosemantic, &benchmark, &iterations, &llvm, &nowarn, &isel, &sentinel};
        
        // iterator over all
        const Arg* begin() const{
            return all[0];
        }

        const Arg* end() const{
            return all[13];
        }
    } possible;

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
    std::map<Arg, std::string>& parse(int argc, char *argv[]){
        std::stringstream ss;
        ss << " ";
        for (int i = 1; i < argc; ++i) {
            ss << argv[i] << " ";
        }

        string argString = ss.str();


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
                    parsedArgs[arg] = positionalArgs[arg.pos-1];
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
                    parsedArgs[arg] = match[1];
                }else if(arg.longOpt!="" && std::regex_search(argString, match, matchLong)){
                    parsedArgs[arg] = match[2];
                }else if(arg.required && !parsedArgs.contains(arg)){
                    std::cerr << "Missing required argument: -" << arg.shortOpt << "/--" << arg.longOpt << std::endl;
                    missingRequired = true;
                }
            }else{
                std::regex matchFlagShort{" -[a-zA-z]*"+arg.shortOpt};
                std::regex matchFlagLong{" --"+arg.longOpt};
                if(std::regex_search(argString, matchFlagShort) || std::regex_search(argString, matchFlagLong)){
                    parsedArgs[arg] = ""; // empty string for flags, will just be checked using .contains
                }
            };
        }

        if(missingRequired){
            printHelp();
            exit(EXIT_FAILURE);
        }
        return parsedArgs;
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

namespace Codegen{
    bool warningsGenerated{false};

    llvm::LLVMContext ctx{};
    auto moduleUP = std::make_unique<llvm::Module>("mod", ctx);
    llvm::Type* i64 = llvm::Type::getInt64Ty(ctx);
    llvm::Type* voidTy = llvm::Type::getVoidTy(ctx);
    llvm::Function* currentFunction = nullptr;

    void warn(const std::string& msg, llvm::Instruction* instr = nullptr){
        if(!ArgParse::parsedArgs.contains(ArgParse::possible.nowarn)){
            llvm::errs() << "Warning: " << msg;
            if(instr != nullptr){
                llvm::errs() << " at " << *instr;
            }
            llvm::errs() << "\n";
            warningsGenerated = true;
        }
    }

#ifndef NDEBUG
    string stringizeVal(llvm::Value* val){
        string buf;
        llvm::raw_string_ostream ss{buf};
        ss << *val;
        return buf;
    }
#endif

    struct BasicBlockInfo{
        bool sealed{false};
        std::unordered_map<string, llvm::Value*/*, string_hash, string_hash::transparent_key_equal*/> varmap{}; // couldn't get transparent lookup to work (string_hash has since been deleted)
    };

    std::unordered_map<llvm::BasicBlock*, BasicBlockInfo> blockInfo{};
    std::unordered_map<llvm::PHINode*, ASTNode*> phisToResolve{};


    llvm::Function* findFunction(string name){
        auto fnIt = llvm::find_if(moduleUP->functions(), [&name](auto& func){return func.getName() == name;});
        if(fnIt == moduleUP->functions().end()){
            return nullptr;
        }else{
            return &*fnIt;
        }
    }

    llvm::Value* varmapLookup(llvm::BasicBlock* block, ASTNode& node) noexcept;

    llvm::Value* wrapVarmapLookupForUse(llvm::IRBuilder<>& irb, ASTNode& node){
        if(node.op == Token::Type::KW_REGISTER){
            // this should be what we want for register vars, for auto vars we aditionally need to look up the alloca (and store it back if its an assignment, see the assignment below)
            return varmapLookup(irb.GetInsertBlock(), node); // NOTE to self: this does work even though there is a pointer to the value in the varmap, because if the mapping gets updated, that whole pointer isn't the value anymore, nothing is changed about what it's pointing to.
        }else {
            return irb.CreateLoad(i64, varmapLookup(irb.GetInsertBlock(), node), node.name);
        }
    }

    // for different block for lookup/insert
    llvm::Value* wrapVarmapLookupForUse(llvm::BasicBlock* block, llvm::IRBuilder<>& irb, ASTNode& node){
        if(node.op == Token::Type::KW_REGISTER){
            // this should be what we want for register vars, for auto vars we aditionally need to look up the alloca (and store it back if its an assignment, see the assignment below)
            return varmapLookup(block, node); // NOTE to self: this does work even though there is a pointer to the value in the varmap, because if the mapping gets updated, that whole pointer isn't the value anymore, nothing is changed about what it's pointing to.
        }else {
            return irb.CreateLoad(i64, varmapLookup(block, node), node.name);
        }
    }

    // automatically creates phi nodes on demand
    llvm::Value* varmapLookup(llvm::BasicBlock* block, ASTNode& node) noexcept {
        string& name = node.name;
        auto& [sealed, varmap] = blockInfo[block];
        auto valueType = (node.op == Token::Type::KW_REGISTER?i64:llvm::PointerType::get(ctx, 0));
        if(varmap.contains(name)){
            return varmap[name];
        }else{
            if(sealed){
                // cache it, so we don't have to look it up every time
                if(block->hasNPredecessors(1)){
                    return varmap[name] = varmapLookup(block->getSinglePredecessor(), node);
                }else if(block->hasNPredecessors(0)){
                    // returning poison is quite reasonable, as anything here should never be used, or looked up (entry block case)
                    warn("Variable " + name + " is used before it is defined");
                    return varmap[name] = llvm::PoisonValue::get(valueType);
                }else{ // > 1 predecessors
                    // create phi node to merge it
                    llvm::IRBuilder<> irb(block);
                    auto nonphi = block->getFirstNonPHI();
                    // if there is no nonphi node, we can just insert at the end, which should be where the irb starts
                    if(nonphi!=nullptr){
                        irb.SetInsertPoint(nonphi); // insertion is before the instruction, so this is the correct position
                    }
                    llvm::PHINode* phi = irb.CreatePHI(valueType, 2, name); // num reserved values here is only a hint, 0 is fine "[...] if you really have no idea", it's at least one because of our algo, 2 because we have >1 preds
                    varmap[name] = phi; // doing this here breaks a potential cycle in the following lookup, if we look up this same name and it points back here, we don't create another phi node, we use the existing one we just created
                    
                    // block is sealed -> we have all the information -> we can add all the incoming values
                    for(auto pred:llvm::predecessors(block)){
                        phi->addIncoming(varmapLookup(pred, node), pred);
                    }
                    return phi;
                }
            }else{
                // we need a phi node in this case
                llvm::IRBuilder<> irb(block);
                auto nonphi = block->getFirstNonPHI();
                // if there is no nonphi node, we can just insert at the end of the block, which should be where the irb starts
                if(nonphi!=nullptr){
                    irb.SetInsertPoint(nonphi); // insertion is before the instruction, so this is the correct position
                }
                llvm::PHINode* phi = irb.CreatePHI(valueType, 2, name); // num reserved values here is only a hint, 0 is fine "[...] if you really have no idea", it's at least one because of our algo
                phisToResolve[phi] = &node;
                
                // incoming values/blocks get added by fillPHIs later
                return varmap[name] = phi;
            }
        }
    }

    // just for convenience
    inline llvm::Value*& updateVarmap(llvm::BasicBlock* block, ASTNode& node, llvm::Value* val) noexcept{
        auto& [sealed, varmap] = blockInfo[block];
        return varmap[node.name] = val;
    }

    inline llvm::Value*& updateVarmap(llvm::IRBuilder<>& irb, ASTNode& node, llvm::Value* val) noexcept{
        return updateVarmap(irb.GetInsertBlock(), node, val);
    }

    // fills phi nodes with correct values, assumes block is sealed
    inline void fillPHIs(llvm::BasicBlock* block) noexcept{
        for(auto& phi: block->phis()){
            for(auto pred: llvm::predecessors(block)){
                phi.addIncoming(varmapLookup(pred, *(phisToResolve[&phi])), pred);
            }
        }
    }
    
    // Seals the block and fills phis
    inline void sealBlock(llvm::BasicBlock* block){
        auto& [sealed, varmap] = blockInfo[block];
        sealed = true;
        fillPHIs(block);
    }

    llvm::Type* sizespecToLLVMType(ASTNode& sizespecNode, llvm::IRBuilder<>& irb){
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
                return wrapVarmapLookupForUse(irb, exprNode);
            case ASTNode::Type::NExprNum:
                return llvm::ConstantInt::get(i64, exprNode.value);
            case ASTNode::Type::NExprCall: 
                {
                    std::vector<llvm::Value*> args(exprNode.children.size());
                    for(unsigned int i = 0; i < exprNode.children.size(); ++i){
                        args[i] = genExpr(*exprNode.children[i], irb);
                    }
                    auto callee = findFunction(exprNode.name);
                    if(callee == nullptr) throw std::runtime_error("Something has gone seriously wrong here, got a call to a function that doesn't exist, but was neither forward declared, nor implicitly declared, its name is " + exprNode.name);
                    if(args.size() != callee->arg_size()){
                        // hw02.txt: "Everything else is handled as in ANSI C", hw04.txt: "note that parameters/arguments do not need to match"
                        // but: from the latest C11 standard draft: "the number of arguments shall agree with the number of parameters"
                        // (i just hope thats the same as in C89/ANSI C, can't find that standard anywhere online, Ritchie & Kernighan says this:
                        // "The effect of the call is undefined if the number of arguments disagrees with the number of parameters in the
                        // definition of the function", which is basically the same)
                        // so technically this is undefined behavior >:)
                        if(SemanticAnalysis::externalFunctionsToNumParams.contains(exprNode.name) && SemanticAnalysis::externalFunctionsToNumParams[exprNode.name] == EXTERNAL_FUNCTION_VARARGS){
                            // in this case we can create a normal call
                            return irb.CreateCall(&*callee, args);
                        }else{
                            // otherwise, there is something weird going on
                            std::stringstream ss{};
                            ss << "Call to function " << exprNode.name << " with " << args.size() << " arguments, but function has " << callee->arg_size() << " parameters";
                            DEBUGLOG(ss.str());
                            warn(ss.str());
                            return llvm::PoisonValue::get(i64);
                        }
                    }else{
                        return irb.CreateCall(&*callee, args);
                    }
                }
                break;
            case ASTNode::Type::NExprUnOp:
                {
                    auto& operandNode = *exprNode.children[0];
                    switch(exprNode.op){
                        // can be TILDE, MINUS, AMPERSAND, LOGICAL_NOT
                        case Token::Type::TILDE:
                            return irb.CreateNot(genExpr(operandNode, irb));
                        case Token::Type::MINUS:
                            return irb.CreateNeg(genExpr(operandNode, irb)); // creates a sub with 0 as first op
                        case Token::Type::LOGICAL_NOT:
                            {
                                auto cmp = irb.CreateICmp(llvm::CmpInst::ICMP_NE, genExpr(operandNode, irb), irb.getInt64(0));
                                return irb.CreateZExt(cmp, i64);
                            }
                        case Token::Type::AMPERSAND:
                            {
                                // get the ptr to the alloca then cast that to an int, because everything (except the auto vars stored in the varmap) is an i64
                                if(operandNode.type == ASTNode::Type::NExprVar){
                                    auto ptr = varmapLookup(irb.GetInsertBlock(), operandNode);
                                    // TODO this is wrong, addrof is also possible on subscripts
                                    return irb.CreatePtrToInt(ptr, i64);
                                }else{ /* has to be a subscript in this case, because of the SemanticAnalysis constarints */
                                    // REFACTOR: would be nice to merge this with the code for loading from subscripts at some point, its almost the same
                                    auto& addrNode = *operandNode.children[0];
                                    auto& indexNode = *operandNode.children[1];
                                    auto& sizespecNode = *operandNode.children[2];

                                    auto addr = genExpr(addrNode, irb); 
                                    auto addrPtr  = irb.CreateIntToPtr(addr, irb.getPtrTy()); // opaque ptrs galore!

                                    auto index = genExpr(indexNode, irb);

                                    llvm::Type* type = sizespecToLLVMType(sizespecNode, irb);
                                    auto getelementpointer = irb.CreateGEP(type, addrPtr, {index});

                                    return irb.CreatePtrToInt(getelementpointer, i64);
                                }
                            }
                        default:
                            throw std::runtime_error("Something has gone seriously wrong here, got a " + Token::toString(exprNode.op) + " as unary operator");
                    }
                }
            case ASTNode::Type::NExprBinOp:
                {
                    auto& lhsNode = *exprNode.children[0];
                    auto& rhsNode = *exprNode.children[1];

                        // all the following logical ops need to return i64s to conform to the C like behavior we want

#define ICAST(irb, x) irb.CreateIntCast((x), i64, false) // unsigned cast because we want 0 for false and 1 for true (instead of -1)

                    // 2 edge cases: assignment, and short circuiting logical ops:
                    // short circuiting logical ops: conditional evaluation

                    // can get preds using: llvm::pred_begin()/ llvm::predecessors()
                    bool isAnd;
                    if((isAnd = (exprNode.op == Token::Type::LOGICAL_AND)) || exprNode.op == Token::Type::LOGICAL_OR){
                        // we need to generate conditional branches in these cases, clang does it the same way
                        // but the lhs is always evaluated anyway, so we can just do that first
                        auto lhs = genExpr(lhsNode, irb);
                        auto lhsi1 = irb.CreateICmp(llvm::CmpInst::ICMP_NE, lhs, irb.getInt64(0)); // if its != 0, then it's true/1, otherwise false/0

                        auto startBB = irb.GetInsertBlock();
                        auto evRHS = llvm::BasicBlock::Create(ctx, "evRHS", currentFunction);
                        auto cont = llvm::BasicBlock::Create(ctx, "shortCircuitCont", currentFunction);

                        if(isAnd){
                            // for an and, we need to evaluate the other side iff its true
                            irb.CreateCondBr(lhsi1, evRHS, cont);
                        }else{
                            // for an or, we need to evaluate the other side iff its false
                            irb.CreateCondBr(lhsi1, cont, evRHS);
                        }

                        // create phi node *now*, because blocks might be split/etc. later
                        irb.SetInsertPoint(cont);
                        auto phi = irb.CreatePHI(irb.getInt1Ty(), 2);
                        phi->addIncoming(irb.getInt1(!isAnd), startBB); // if we skipped (= short circuited), we know that the value is false if it was an and, true otherwise

                        auto& [rhsSealed, rhsVarmap] = blockInfo[evRHS];
                        rhsSealed = true;
                        // don't need to fill phi's for RHS later, because it cant generate phis: is sealed, and has single parent

                        // var map is queried recursively anyway, would be a waste to copy it here

                        irb.SetInsertPoint(evRHS);
                        auto rhs = genExpr(rhsNode, irb);
                        auto rhsi1 = irb.CreateICmp(llvm::CmpInst::ICMP_NE, rhs, irb.getInt64(0));
                        auto& compResult = rhsi1;
                        irb.CreateBr(cont);

                        auto& [contSealed, contVarmap] = blockInfo[cont];
                        contSealed = true;
                        // we don't need to fill the phis of this block either, because it is sealed basically from the beginning, thus any phi nodes generated are complete already

                        auto rhsParentBlock = irb.GetInsertBlock();
                        phi->addIncoming(compResult, rhsParentBlock); // otherwise, if we didnt skip, we need to know what the rhs evaluated to

                        irb.SetInsertPoint(cont);

                        return ICAST(irb, phi);
                    };

                    // assignment needs special handling:
                    // before this switch and CRUCIALLY (!!!) before the lhs gets evaluated, check if exprNode.op is an assign and if the left hand side is a subscript. in that case, we need to generate a store instruction for the assignment
                    //  we also need to generate a store, if the lhs is an auto variable
                    auto rhs = genExpr(rhsNode, irb);
                    if(exprNode.op == Token::Type::ASSIGN){
                        if(lhsNode.type == ASTNode::Type::NExprSubscript){
                            auto addr = genExpr(*lhsNode.children[0], irb);
                            auto index = genExpr(*lhsNode.children[1], irb);
                            auto& sizespecNode = *lhsNode.children[2];

                            llvm::Type* type = sizespecToLLVMType(sizespecNode, irb);
                            // first cast, then store, so that the right amount is stored
                            // this could be done with a trunc, but that is only allowed if the type is strictly smaller, the CreateIntCast distinguishes these cases and takes care of it for us
                            auto cast = irb.CreateIntCast(rhs, type, true);

                            auto addrPtr = irb.CreateIntToPtr(addr, irb.getPtrTy()); // opaque ptrs galore!

                            auto getelementpointer = irb.CreateGEP(type, addrPtr, {index});
                            irb.CreateStore(cast, getelementpointer);
                        }else if(/* lhs node has to be var if we're here */ lhsNode.op == Token::Type::KW_AUTO){
                            irb.CreateStore(rhs, varmapLookup(irb.GetInsertBlock(), lhsNode));
                        }else{/* in this case it has to be a register variable */
                            // in lhs: "old" varname of the var we're assigning to -> update mapping
                            // in rhs: value to assign to it
                            updateVarmap(irb, lhsNode, rhs);
                        }
                        return rhs; // just as before, return the result, not the store/assign/etc.
                    }
                    auto lhs = genExpr(lhsNode, irb);

                    // for all other cases this is a post order traversal of the epxr tree

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
                            return ICAST(irb,irb.CreateICmp(llvm::CmpInst::ICMP_SLT, lhs, rhs));
                        case Token::Type::GREATER:
                            return ICAST(irb,irb.CreateICmp(llvm::CmpInst::ICMP_SGT, lhs, rhs));
                        case Token::Type::LESS_EQUAL:
                            return ICAST(irb,irb.CreateICmp(llvm::CmpInst::ICMP_SLE, lhs, rhs));
                        case Token::Type::GREATER_EQUAL:
                            return ICAST(irb,irb.CreateICmp(llvm::CmpInst::ICMP_SGE, lhs, rhs));
                        case Token::Type::EQUAL:
                            return ICAST(irb,irb.CreateICmp(llvm::CmpInst::ICMP_EQ, lhs, rhs));
                        case Token::Type::NOT_EQUAL:
                            return ICAST(irb,irb.CreateICmp(llvm::CmpInst::ICMP_NE, lhs, rhs));
                        /* non-short circuiting variants of the logical ops
                        case Token::Type::LOGICAL_AND:
                            {
                                // These instrs expect an i1 for obvious reasons, but we have i64s, so we need to convert them here
                                // but because of the C like semantics, we need to zext them back to i64 afterwards
                                auto lhsi1 = irb.CreateICmp(llvm::CmpInst::ICMP_NE, lhs, irb.getInt64(0)); // if its != 0, then it's true/1, otherwise false/0
                                auto rhsi1 = irb.CreateICmp(llvm::CmpInst::ICMP_NE, rhs, irb.getInt64(0)); // if its != 0, then it's true/1, otherwise false/0
                                return ICAST(irb,irb.CreateLogicalAnd(lhsi1,rhsi1)); // i have no idea how this works, cant find a 'logical and' instruction...
                            }
                        case Token::Type::LOGICAL_OR:
                            {
                                // These instrs expect an i1 for obvious reasons, but we have i64s, so we need to convert them here
                                // but because of the C like semantics, we need to zext them back to i64 afterwards
                                auto lhsi1 = irb.CreateICmp(llvm::CmpInst::ICMP_NE, lhs, irb.getInt64(0)); // if its != 0, then it's true/1, otherwise false/0
                                auto rhsi1 = irb.CreateICmp(llvm::CmpInst::ICMP_NE, rhs, irb.getInt64(0)); // if its != 0, then it's true/1, otherwise false/0
                                return ICAST(irb,irb.CreateLogicalOr(lhsi1,rhsi1)); // i have no idea how this works, cant find a 'logical or' instruction...
                            }
                            */
#undef ICAST
                        default:
                            throw std::runtime_error("Something has gone seriously wrong here, got a " + Token::toString(exprNode.op) + " as binary operator");
                    }
                }
            case ASTNode::Type::NExprSubscript:
                {
                    // this can *ONLY* be a "load" (getelementpointer) subscript, store has been handled in the special case for assignments above
                    auto& addrNode = *exprNode.children[0];
                    auto& indexNode = *exprNode.children[1];
                    auto& sizespecNode = *exprNode.children[2];

                    auto addr = genExpr(addrNode, irb); 
                    auto addrPtr  = irb.CreateIntToPtr(addr, irb.getPtrTy()); // opaque ptrs galore!

                    auto index = genExpr(indexNode, irb);

                    llvm::Type* type = sizespecToLLVMType(sizespecNode, irb);
                    auto getelementpointer = irb.CreateGEP(type, addrPtr, {index});

                    auto load = irb.CreateLoad(type, getelementpointer);

                    // we only have i64s, thus we need to convert our extracted value back to an i64
                    // after reading up on IntCast vs SExt/Trunc (in the source code... Why can't they just document this stuff properly?), it seems that CreateIntCast is a wrapper around CreateSExt/CreateZExt, but in this case we know exactly what we need, so I think CreateSExt would be fine, except that is only allowed if the type is strictly larger, the CreateIntCast distinguishes these cases and takes care of it for us
                    //auto castResult = irb.CreateSExt(load, i64);
                    auto castResult = irb.CreateIntCast(load, i64, true);

                    return castResult;
                }
            default:
                throw std::runtime_error("Something has gone seriously wrong here");
        }
    }

    void genStmts(std::vector<unique_ptr<ASTNode>>& stmts, llvm::IRBuilder<>& irb, std::unordered_set<std::string_view>& scopeDecls);

    void genStmt(ASTNode& stmtNode, llvm::IRBuilder<>& irb, std::unordered_set<std::string_view>& scopeDecls){
        switch(stmtNode.type){
            case ASTNode::Type::NStmtDecl:
                {
                    auto initializer = genExpr(*stmtNode.children[1], irb);

                    // so we can remove them on leaving the scope
                    scopeDecls.emplace(stmtNode.children[0]->name);
                    // i hope setting names doesn't hurt performance, but it wouldn't make sense if it did
                    if(stmtNode.op == Token::Type::KW_AUTO){
                        auto entryBB = &currentFunction->getEntryBlock();
                        auto nonphi = entryBB->getFirstNonPHI();
                        llvm::IRBuilder<> entryIRB(entryBB, entryBB->getFirstInsertionPt()); // i hope this is correct
                        if(nonphi != nullptr){
                            entryIRB.SetInsertPoint(nonphi);
                        }

                        auto alloca = entryIRB.CreateAlloca(i64); // this returns the ptr to the alloca'd memory
                        alloca->setName(stmtNode.children[0]->name);
                        irb.CreateStore(initializer, alloca);

                        updateVarmap(irb, *stmtNode.children[0], alloca); // we actually want to save the ptr (cast to an int, because everything is an int, i hope there arent any provenance problems here) to the alloca'd memory, not the initializer
                    }else if(stmtNode.op == Token::Type::KW_REGISTER){
                        updateVarmap(irb, *stmtNode.children[0], initializer);
                    }else{
                        throw std::runtime_error("Something has gone seriously wrong here, got a " + Token::toString(stmtNode.op) + " as decl type");
                    }
                }
                break;
            case ASTNode::Type::NStmtReturn:
                // technically returning "nothing" is undefined behavior, so we can just return 0 in that case
                if(stmtNode.children.size() == 0){
                    //irb.CreateRet(irb.getInt64(0));
                    // actually I think returning poison/undef is better, and warning about it
                    auto retrn = irb.CreateRet(llvm::PoisonValue::get(i64));
                    warn("Returning nothing is undefined behavior", retrn);
                }else{
                    irb.CreateRet(genExpr(*stmtNode.children[0], irb));
                }
                break;
            case ASTNode::Type::NStmtBlock:
                {
                    // safe because the strings (variable names) are constant -> can't invalidate set invariants/hashes
                    // I know this is quite slow and in retrospect I would have designed my datastructure differently to retain information about scopes during the semantic analysis, but thats not really easily possible anymore at this stage
                    std::unordered_set<std::string_view> scopeDecls{};

                    auto varmapCopy = blockInfo[irb.GetInsertBlock()].varmap;
                    genStmts(stmtNode.children, irb, scopeDecls);
                    // leaving the scope == leaving the block, so for every scope we leave, we can handle it here
                    // new idea: keep track of all declarations made in the current scope, and upon leaving:
                    // 1. if they are not present in the varmapCopy: remove them from the varmap
                    // 2. if they are present there: copy them from the varmapCopy
                    // this is also not the fastest, but it should work

                    auto& [_, varmap] = blockInfo[irb.GetInsertBlock()];

                    for(std::string_view decl: scopeDecls){
                        // I sadly couldn't get heterogeneous/transparent lookup for the varmap to work, so we have to do this, which makes it even slower :(
                        string declStr{decl};
                        if(varmapCopy.contains(declStr)){
                            varmap[declStr] = varmapCopy[declStr];
                        }else{
                            varmap.erase(declStr);
                        }
                    }

                    // after leaving the scope, the decls are thrown away
                }
                break;
            case ASTNode::Type::NStmtIf:
                {
                    bool hasElse = stmtNode.children.size() == 3;

                    llvm::BasicBlock* thenBB =         llvm::BasicBlock::Create(ctx, "then",   currentFunction);
                    llvm::BasicBlock* elseBB = hasElse?llvm::BasicBlock::Create(ctx, "else",   currentFunction): nullptr; // its generated this way around, so that the cont block is always after the else block
                    llvm::BasicBlock* contBB =         llvm::BasicBlock::Create(ctx, "ifCont", currentFunction);
                    if(!hasElse){
                        elseBB = contBB;
                    }

                    auto condition = genExpr(*stmtNode.children[0], irb); // as everything in our beautiful C like language, this is an i64, so "cast" it to an i1
                    condition = irb.CreateICmp(llvm::CmpInst::ICMP_NE, condition, irb.getInt64(0));
                    irb.CreateCondBr(condition, thenBB, elseBB);
                    // block is now finished
                    
                    auto& [thenSealed, thenVarmap] = blockInfo[thenBB];
                    thenSealed = true;
                    // var map is queried recursively anyway, would be a waste to copy it here

                    irb.SetInsertPoint(thenBB);
                    genStmt(*stmtNode.children[1], irb, scopeDecls);
                    auto thenBlock = irb.GetInsertBlock();

                    bool thenBranchesToCont = !(thenBlock->getTerminator());
                    if(thenBranchesToCont){
                        irb.CreateBr(contBB);
                    }
                    // now if is generated -> we can seal else

                    auto& [elseSealed, elseVarmap] = blockInfo[elseBB];
                    elseSealed = true; // if this is cont: then it's sealed. If this is else, then it's sealed too (but then cont is not sealed yet!).
                    if(hasElse){
                        irb.SetInsertPoint(elseBB);
                        genStmt(*stmtNode.children[2], irb, scopeDecls);
                        auto elseBlock = irb.GetInsertBlock();

                        bool elseBranchesToCont = !(elseBlock->getTerminator());
                        if(elseBranchesToCont){
                            irb.CreateBr(contBB);
                        }
                    }

                    // now that stuff is sealed, can also seal cont
                    blockInfo[contBB].sealed = true;

                    // then/else cannot generate phi nodes, because they were sealed from the start and have a single predecessor
                    //fillPHIs(thenBB);
                    //fillPHIs(elseBB);

                    // now that we've generated the if, we can 'preceed as before' in the parent call, so just set the irb to the right place
                    irb.SetInsertPoint(contBB); 
                }
                break;
            case ASTNode::Type::NStmtWhile:
                {
                    llvm::BasicBlock* condBB = llvm::BasicBlock::Create(ctx, "whileCond", currentFunction);
                    llvm::BasicBlock* bodyBB = llvm::BasicBlock::Create(ctx, "whileBody", currentFunction);
                    llvm::BasicBlock* contBB = llvm::BasicBlock::Create(ctx, "whileCont", currentFunction);

                    irb.CreateBr(condBB);
                    // block is now finished

                    blockInfo[condBB].sealed = false; // can get into condition block from body, which is not generated yet

                    irb.SetInsertPoint(condBB);

                    auto conditionExpr = genExpr(*stmtNode.children[0], irb); // as everything in our beautiful C like language, this is an i64, so "cast" it to an i1
                    conditionExpr = irb.CreateICmp(llvm::CmpInst::ICMP_NE, conditionExpr, irb.getInt64(0));
                    irb.CreateCondBr(conditionExpr, bodyBB, contBB);

                    blockInfo[bodyBB].sealed = true; // can only get into body block from condition block -> all predecessors are known

                    irb.SetInsertPoint(bodyBB);
                    genStmt(*stmtNode.children[1], irb, scopeDecls);
                    auto bodyBlock = irb.GetInsertBlock();

                    bool bodyBranchesToCond = !(bodyBlock->getTerminator());
                    if(bodyBranchesToCond){
                        irb.CreateBr(condBB);
                    }

                    // seal condition now that all its predecessors (start block and body) are fully known and generated
                    sealBlock(condBB);

                    // body cannot generate phi nodes because its sealed from the start and has a single predecessor
                    //fillPHIs(bodyBB);

                    blockInfo[contBB].sealed = true;

                    irb.SetInsertPoint(contBB);
                }
                break;

            case ASTNode::Type::NExprVar:
            case ASTNode::Type::NExprNum:
            case ASTNode::Type::NExprCall:
            case ASTNode::Type::NExprUnOp:
            case ASTNode::Type::NExprBinOp:
            case ASTNode::Type::NExprSubscript:
                genExpr(stmtNode, irb); 
                break;

            // hopefully impossible
            default:
                throw std::runtime_error("Something has gone seriously wrong here" STRINGIZE_MACRO(__LINE__));
        }
    }

    void genStmts(std::vector<unique_ptr<ASTNode>>& stmts, llvm::IRBuilder<>& irb, std::unordered_set<std::string_view>& scopeDecls){
        for(auto& stmt : stmts){
            genStmt(*stmt, irb, scopeDecls);
            if(stmt->type == ASTNode::Type::NStmtReturn){
                // stop the generation for this block
                break;
            }
        }
    }

    void genFunction(ASTNode& fnNode){
        auto paramNum = fnNode.children[0]->children.size();
        auto typelist = llvm::SmallVector<llvm::Type*, 8>(paramNum, i64);
        llvm::Function* fn = findFunction(fnNode.name);
        currentFunction = fn;
        llvm::BasicBlock* entryBB = llvm::BasicBlock::Create(ctx, "entry", fn);
        // simply use getEntryBlock() on the fn when declarations need to be added
        blockInfo[entryBB].sealed = true;
        llvm::IRBuilder<> irb(entryBB);

        for(unsigned int i = 0; i < paramNum; i++){
            llvm::Argument* arg = fn->getArg(i);
            auto& name = fnNode.children[0]->children[i]->name;
            arg->setName(name);
            updateVarmap(irb, *fnNode.children[0]->children[i], arg);
        }

        auto& blockNode = *fnNode.children[1];
        std::unordered_set<std::string_view> empty; // its fine to leave it uninitialized, because this child is always a block -> generates a new and actually useful scope decls anyway
        genStmt(blockNode, irb, empty); // calls gen stmts on the blocks children, but does additional stuff

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
        // declare implicitly declared functions
        for(auto& [fnName, fnParamCount]: SemanticAnalysis::externalFunctionsToNumParams){
            llvm::SmallVector<llvm::Type*, 8> params;
            if(fnParamCount == EXTERNAL_FUNCTION_VARARGS){
                params = {};
            }else{
                params = llvm::SmallVector<llvm::Type*, 8>(fnParamCount, i64);

            }
            llvm::FunctionType* fnTy = llvm::FunctionType::get(i64, params, fnParamCount == EXTERNAL_FUNCTION_VARARGS);
            llvm::Function::Create(fnTy, llvm::GlobalValue::ExternalLinkage, fnName, moduleUP.get()); 
        }

        // declare all functions in the file, to easily allow forward declarations
        auto& children = root.children;
        for(auto& fnNode : children){
            auto paramNum = fnNode->children[0]->children.size();
            auto typelist = llvm::SmallVector<llvm::Type*, 8>(paramNum, i64);
            llvm::FunctionType* fnTy = llvm::FunctionType::get(i64, typelist, false);
            llvm::Function* fn = llvm::Function::Create(fnTy, llvm::GlobalValue::ExternalLinkage, fnNode->name, moduleUP.get());
            for(unsigned int i = 0; i < paramNum; i++){
                llvm::Argument* arg = fn->getArg(i);
                auto& name = fnNode->children[0]->children[i]->name;
                arg->setName(name);
            }
        }

        for(auto& child:children){
            genFunction(*child);
        }
    }

    bool generate(AST& ast, llvm::raw_ostream& out){
        genRoot(ast.root);


        bool moduleIsBroken = llvm::verifyModule(*moduleUP, &llvm::errs());
        if(moduleIsBroken){
            moduleUP->print(llvm::errs(), nullptr);
        }else{
            moduleUP->print(out, nullptr);
        }
        if(warningsGenerated){
            llvm::errs() << "Warnings were generated during code generation, please check the output for more information\n";
        }
        return !moduleIsBroken;
    }

}


/*
 ------------------------------------------------------------------------------------------------------
 HW 6 START
 Exceptions:
 ------------------------------------------------------------------------------------------------------
 Subset of LLVM IR used (checks/not checks: done/not done):
 -     Format: <instr>  [<operand type(s)>]
 - [x] Alloca           [i64]
 - [x] Load             [ptr]
 - [x] Store            [i64, ptr]
 - [x] Bitwise Not      [i64] (transformed by llvm)
 - [x] Negate           [i64] (transformed by llvm)
 - [x] ICmp             [i64] (with ICMP_EQ, ICMP_NE, ICMP_SLT, ICMP_SLE, ICMP_SGT, ICMP_SGE)
 - [x] ZExt             [i64]
 - [x] SExt             [i64] (used in exactly 2 places: storing from subscripts, and loading from subscripts)
 - [x] PtrToInt         [ptr]
 - [x] IntToPtr         [i64]
 - [x] Or               [i64]
 - [x] And              [i64]
 - [x] Xor              [i64]
 - [x] Add              [i64]
 - [x] Sub              [i64]
 - [x] Mul              [i64]
 - [x] SDiv             [i64]
 - [x] SRem             [i64]
 - [x] Shl              [i64]
 - [x] AShr             [i64]
 - [x] GetElementPtr    [ptr, i64]
 - [x] Br (cond/uncond) [i1]
 - [ ] Poison Values    [i64/ptr]

 Not relevant for this task, but used:
 - Call             [return: i64, args: i64...]
 - PHIs             [i64, ptr]
 - Ret              [i64]
 - Unreachable

 ----------------------------------------------------
 ARM (v8-A) subset used:
 Control Flow:
      - Conditional Branches: CBNZ, CBZ (branch if not zero, branch if zero)
      - Unconditional Branches: B (immediate), BR (Branch to register), RET (return from subroutine)
 Load/Store:
      - Load: LDR (load register)
      - Store: STR (store register)


 ----------------------------------------------------
 useful stuff im putting somewhere
 llvm::isa<llvm::ReturnInst> (or isa_and_nonnull<>) and similar things can be used for checking if a value is of a certain type
 llvm::dyn_cast<> should not be used for large chains of ifs, there is the InstVisitor class for that purpose
 -> For the patterns and matching, use an InstVisitor, where each method depends on the root value type of the pattern,
    possibly hardcode which patterns are tried based on their root class, or try to do it with some compile time programming
 Scratch this, the llvm::Instruction::... enum is much easier, and the InstVisitor is not really needed but maybe filtering the patterns
 at compile time is still useful

 */

namespace Codegen::ISel{

    unsigned currentFunctionBytesToFreeAtEnd{0};

    // TODO probably delete
    //template<typename T>
    //consteval std::vector<llvm::Value*> patternsMatchingType(std::vector<llvm::Value*> patterns [> this will be a list of patterns which in turn contain root values, but this is easier for now<]){
    //    for(auto it = patterns.begin(); it!= patterns.end(); ){
    //        auto pattern = *(it++);
    //        if(!llvm::isa<T>(pattern)){
    //            it = patterns.erase(it);
    //        }
    //    }
    //    return patterns;
    //}

    // test, for the pattern matching
    struct Pattern{
    public:
        static const Pattern emptyPattern;

        const unsigned type{0};       // basically an enum member for type checks, will be initialized with llvm::Instruction::<type>
                                      // find instruction types in include/llvm/IR/Instruction.def
                                      // 0 means no type check with this requirement (but children/constants etc. still need to match)
        const std::vector<Pattern> children{};
        const unsigned totalSize;
        const bool root{false};
        const bool constant{false};  // checked value has to be a constant
        const bool noDelete{false};  // even though the pattern matches, the value should not be deleted

        struct PatternHash{
            std::size_t operator()(const Pattern& p) const{
                std::size_t hash = 0;
                hash ^= std::hash<unsigned>{}(p.type);
                hash ^= std::hash<bool>{}(p.root);
                hash ^= std::hash<bool>{}(p.constant);
                for(auto& child:p.children){
                    hash ^= PatternHash{}(child);
                }
                return hash;
            }
        };

        // == operator for hashing
        bool operator==(const Pattern& other) const{
            if(type != other.type || root != other.root || constant != other.constant || children.size() != other.children.size()){
                return false;
            }
            for(unsigned int i = 0; i < children.size(); i++){
                if(children[i] != other.children[i]){
                    return false;
                }
            }
            return true;
        }

        // map the roots of the patterns to their ARM replacement instructions (calls)
        static std::unordered_map<Pattern, llvm::CallInst* (*)(llvm::IRBuilder<>&), PatternHash> replacementCalls;

        void replaceWithARM(llvm::Instruction* instr){

#define PUSH_OPS(inst,currentPattern)                                                                  \
    /* iterating over children means, that if they are empty, we ignore them, which is what we want */ \
    for(unsigned i = 0; i < currentPattern->children.size(); i++){                                     \
        if(currentPattern->children[i].type!=0 && !currentPattern->children[i].noDelete){              \
            auto op = instr->getOperand(i);                                                            \
            auto opInstr = llvm::dyn_cast<llvm::Instruction>(op);                                      \
            assert(opInstr != nullptr);                                                                \
            toRemove.push(opInstr);                                                                    \
            patternQueue.push(currentPattern->children.data()+i);                                      \
        }                                                                                              \
    }

            llvm::IRBuilder<> irb{instr->getParent()};
            auto root = instr;

            irb.SetInsertPoint(root);
            // generate the appropriate call instruction
            auto call = Pattern::replacementCalls[*this](irb); 

            // remove all operands of the instruction from the program
            std::queue<llvm::Instruction*> toRemove{};
            std::queue<const Pattern*> patternQueue{}; // TODO validate the pattern queue thing
            PUSH_OPS(instr, this);
            while(!toRemove.empty()){ // TODO i can already see this making problems with the circularly referent PHIs
                instr = toRemove.front();
                toRemove.pop();

                PUSH_OPS(instr, patternQueue.front());
                patternQueue.pop();

                DEBUGLOG("removing " << *instr);
#ifndef NDEBUG
                // normally it should only have one use. But because we also use it in the replacement call, it has two uses, so error on >=3
                if(instr->hasNUsesOrMore(3)){ // TODO Uses() or Users() ?? Uses, right?
                    llvm::errs() << "Critical: Instruction has more than one use, cannot be removed, pattern matching severly wrong, instr: " << *instr << ", num uses: " << instr->getNumUses() << "\n";
                    // print users:
                    for(auto user: instr->users()){
                        llvm::errs() << "user: " << *user << "\n";
                    }
                    fflush(stderr);
                    abort();
                }
#endif
                instr->eraseFromParent();
                DEBUGLOG("success!")
            }

            if(!noDelete){
                root->replaceAllUsesWith(call);
                root->eraseFromParent();
            }

#undef PUSH_OPS
        }

    private:
        // TODO it might be worth having an alternative std::function for matching, which simply gets the llvm value as an argument and returns true if it matches,
        // this would allow for arbitrary matching

        unsigned calcTotalSize(std::vector<Pattern> children) const noexcept {
            unsigned res = 1;
            for(auto& child:children) if(child.type!=0 || child.constant) res+=child.totalSize;
            return res;
        }

        Pattern(unsigned isMatching, std::vector<Pattern> children, bool constant, bool root, bool noDelete) :
            type(isMatching),
            children(children),
            totalSize(calcTotalSize(children)),
            root(root),
            constant(constant),
            noDelete(noDelete)
            {}

    public:
        Pattern(unsigned isMatching = 0, std::vector<Pattern> children = {}, bool noDelete = false) :
            type(isMatching), children(children), totalSize(calcTotalSize(children)), noDelete(noDelete)
            {}

        static const Pattern constantPattern;

        /// constructor like method for making a constant requirement
        static Pattern make_constant(){
            return constantPattern;
        }

        static Pattern make_root(llvm::CallInst* (*replacementCall)(llvm::IRBuilder<>&), unsigned isMatching = 0, std::vector<Pattern> children = {}, bool noDelete = false){
        //static Pattern make_root(std::function<llvm::CallInst*(llvm::IRBuilder<>&)> replacementCall, unsigned isMatching = 0, std::vector<Pattern> children = {}, std::initializer_list<Pattern> alternatives = {}){
            Pattern rootPattern{isMatching, children, false, true, noDelete};
            Pattern::replacementCalls[rootPattern] = replacementCall;
            return rootPattern;
        }

        // TODO delete
        // example replacementCall function
        //[](llvm::IRBuilder<>& irb){
        //    // TODO this part changes for every pattern, do that
        //    // TODO add args
        //    return irb.CreateCall(llvm::Function::Create(llvm::FunctionType::get(i64, true), llvm::GlobalValue::ExternalLinkage, "STR", moduleUP.get()), {});
        //};

        /**
         * The matching works thusly:
         * - if there are alternatives, to this pattern, the pattern matches iff any of the alternatives match
         * - 0 as an instruction type does not check the type of the instruction, but other checks are still performed
         * - if the instruction type is not 0, it must match the instruction type of the value, if the value is not an instruction, it does not match
         * - the number of children must match the number of operands to the instruction, except if there are 0 children, which indicates that the instruction has arbitrary operands
         * - if the pattern (node) has a constant requirement, the value must be a constant and the constant must match
         * - if the pattern (node) is neither the root, nor a leaf, this value must only have one user (which is the instruction that we are trying to match), so it can be deleted without affecting other instructions
         * - all children patterns must also match their respective operands
         *
         * This basically boils down to tree matching with edge splitting on DAGs (any instruction except the root and those which are ignored, because they are empty require exactly one predecessor to match), with the DAG being the basic blocks of an llvm function
         */
        bool match(llvm::Value* val) const {
            if(type!=0 && !root && val->hasNUsesOrMore(2)) return false; // TODO check again. This requirement is important for all 'real' inner nodes, i.e. all except the root and leaves
                                                           // leaves should have type 0
                                                           // TODO hasOneUse() vs hasOneUser()

            // constant check
            if(constant){
                return llvm::isa_and_nonnull<llvm::ConstantInt>(val);
            }

            auto inst = llvm::dyn_cast_or_null<llvm::Instruction>(val); // this also propagates nullptrs, and it just returns a null pointer if the cast failed, this saves us an isa<> check
            if(type!=0 && 
                 (inst==nullptr || inst->getOpcode() != type || (children.size() != 0 && children.size() != inst->getNumOperands()))){
                return false;
            }
            for(unsigned int i = 0; i < children.size(); i++){
                if(!children[i].match(inst->getOperand(i))){
                    return false;
                }
            }
            return true;
        }
    };

    const Pattern Pattern::emptyPattern = Pattern{0,{}, false, false, true};;
    std::unordered_map<Pattern, llvm::CallInst* (*)(llvm::IRBuilder<>&), Pattern::PatternHash> Pattern::replacementCalls{};
    const Pattern Pattern::constantPattern = Pattern(0, {}, true, false, true);

    std::unordered_set<unsigned> skippableTypes{
        llvm::Instruction::Ret,
        llvm::Instruction::Call,
        llvm::Instruction::Unreachable,
        llvm::Instruction::PHI,
    };

    /// for useful matching the patterns need to be sorted by totalSize (descending) here -> TODO: do this at compile time wherever the patterns are stored
    std::unordered_map<llvm::Instruction*, Pattern&> matchPatterns(llvm::Function* func, std::vector<Pattern>& patterns){
        std::unordered_map<llvm::Instruction*, Pattern&> matches{};
        std::unordered_set<llvm::Instruction*> covered{};
        for(auto& block:*func){
            // iterate over the instructions in a bb in reverse order, to iterate over the dataflow top down
            for(auto& instr:llvm::reverse(block)){

                // skip returns branches, calls, etc.
                if(skippableTypes.contains(instr.getOpcode())) continue;

                // skip instructions that are already matched
                if(covered.contains(&instr)) continue;
                
                // find largest pattern that matches
                for(auto& pattern:patterns){
                    if(pattern.match(&instr)){
                        matches.emplace(&instr, pattern);
                        // TODO this definitely would mess up the iterator, we need to somehow remove them afterwards, and skip them while iterating
                        /*
                          the following code boils down to:

                          addCovered(&covered, &pattern, &instr){
                              for(auto& child:instr->operands()){
                                  covered[child] = pattern;
                                  addCovered(covered, pattern, child);
                              }
                          }

                        */

                        std::queue<const Pattern*> patternQueue{}; // TODO validate the pattern queue thing
                        patternQueue.push(&pattern);

                        // TODO same problem as above in the instruction deletion: this just uses all operands and does not stop at the children of the pattern
                        std::queue<llvm::Instruction*> coveredInsertionQueue{}; // lets not do this recursively...
                        coveredInsertionQueue.push(&instr);
                        while(!coveredInsertionQueue.empty()){
                            // TODO i can already see this making problems with the circularly referent PHIs

                            auto current = coveredInsertionQueue.front();
                            auto currentPattern = patternQueue.front();
                            coveredInsertionQueue.pop();
                            patternQueue.pop();

                            covered.insert(current);
                            
                            // add remaining operands to queue
                            // iterating over children means, that if they are empty, we ignore them, which is what we want
                            for(unsigned i = 0; i < currentPattern->children.size(); i++){
                                if(currentPattern->children[i].type!=0){ // this also guarantees that this is not a constant
                                    auto op = current->getOperand(i);
                                    auto opInstr = llvm::dyn_cast<llvm::Instruction>(op);
                                    coveredInsertionQueue.push(opInstr);
                                    patternQueue.push(currentPattern->children.data()+i);
                                }
                            }
                        }

                        goto cont;
                    }
                }

                llvm::errs() << "no pattern matched for instruction: " << instr << "\n";
                THROW_TODO;

cont:
                continue; // for verbosity
            }
        }

        // rewrite IR with the matched patterns
        for(auto& [matchedInstr, matchedPattern]:matches){
            matchedPattern.replaceWithARM(matchedInstr);
        }

        return matches;
    }


#define CREATE_INST_FN(name, ret, ...)                          \
    {                                                           \
        name,                                                   \
        llvm::Function::Create(                                 \
            llvm::FunctionType::get(ret, {__VA_ARGS__}, false), \
            llvm::GlobalValue::ExternalLinkage,                 \
            name,                                               \
            *moduleUP                                           \
        )                                                       \
    }

#define CREATE_INST_FN_VARARGS(name, ret)                  \
    {                                                           \
        name,                                                   \
        llvm::Function::Create(                                 \
            llvm::FunctionType::get(ret, {}, true), \
            llvm::GlobalValue::ExternalLinkage,                 \
            name,                                               \
            *moduleUP                                           \
        )                                                       \
    }

    /// functions to serve as substitute for actual ARM instructions
    static std::unordered_map<string, llvm::Function*> instructionFunctions;

    static void initInstructionFunctions(){
        // for some obscure reason, adding this in a static initializer stops the pattern matching from working
        instructionFunctions = {
            CREATE_INST_FN("ARM_add",       i64, i64, i64),
            CREATE_INST_FN("ARM_add_SP",    voidTy, i64), // simulate add to stack pointer
            CREATE_INST_FN("ARM_add_SHIFT", voidTy, i64, i64,  i64),
            CREATE_INST_FN("ARM_sub",       i64, i64, i64),
            CREATE_INST_FN("ARM_sub_SP",    llvm::Type::getInt64PtrTy(ctx), i64), // simulate sub from stack pointer
            CREATE_INST_FN("ARM_sub_SHIFT", llvm::Type::getInt64PtrTy(ctx), i64, i64,  i64),
            CREATE_INST_FN("ARM_madd",      i64, i64, i64,  i64),
            CREATE_INST_FN("ARM_msub",      i64, i64, i64,  i64),
            CREATE_INST_FN("ARM_sdiv",      i64, i64, i64),

            // TODO take care of cmp things, try to merge flag setting with flag consuming, but I don't think this is universally possible
            CREATE_INST_FN("ARM_cmp",      i64, i64, i64),
            CREATE_INST_FN_VARARGS("ARM_csel",       i64), // condition is represented as string, so use varargs
            CREATE_INST_FN_VARARGS("ARM_csel_i1",    llvm::Type::getInt1Ty(ctx)), // condition is represented as string, so use varargs

            // shifts by variable and immediate amount
            CREATE_INST_FN("ARM_lsl_imm",  i64,  i64, i64),
            CREATE_INST_FN("ARM_lsl_var",  i64,  i64, i64),
            CREATE_INST_FN("ARM_asr_imm",  i64,  i64, i64),
            CREATE_INST_FN("ARM_asr_var",  i64,  i64, i64),

            // bitwise
            CREATE_INST_FN("ARM_and",      i64, i64, i64),
            CREATE_INST_FN("ARM_orr",      i64, i64, i64),
            CREATE_INST_FN("ARM_eor",      i64, i64, i64), // XOR

            // mov
            CREATE_INST_FN("ARM_mov",       i64, i64),
            CREATE_INST_FN("ARM_mov_shift", i64, i64,  i64),

            // memory access
            // (with varargs to be able to simulate the different addressing modes)
            CREATE_INST_FN_VARARGS("ARM_ldr",      i64),
            CREATE_INST_FN_VARARGS("ARM_ldr_sb",   i64),
            CREATE_INST_FN_VARARGS("ARM_ldr_sh",   i64),
            CREATE_INST_FN_VARARGS("ARM_ldr_sw",   i64),
            CREATE_INST_FN_VARARGS("ARM_str",      voidTy),
            CREATE_INST_FN_VARARGS("ARM_str32",    voidTy),
            CREATE_INST_FN_VARARGS("ARM_str32_b",  voidTy),
            CREATE_INST_FN_VARARGS("ARM_str32_h",  voidTy),

            // control flow/branches
            CREATE_INST_FN_VARARGS("ARM_b_cond", llvm::Type::getInt1Ty(ctx)),
            CREATE_INST_FN_VARARGS("ARM_b",      voidTy),
            CREATE_INST_FN("ARM_b_cbnz",         llvm::Type::getInt1Ty(ctx),  i64),

            // 'metadata' calls
            CREATE_INST_FN_VARARGS("ZExt_handled_in_Reg_Alloc", i64),
        };
    }

#undef CREATE_INST_FN

    // ARM zero register, technically not necessary, but its nice programatically, in order not to use immediate operands where its not possible
    auto XZR = llvm::ConstantInt::get(i64, 0);

    // TODO delete/move
    // TODO do this for multiple functions
    void test(){

        initInstructionFunctions();

        // TODO whats still missing is something to consume the GEP generated for subscripts with & operands before them

        // TODO
        // check that all ops which are possibly encoded as immediates, actually fit into the respective immediate encoding

#define MAYBE_MAT_CONST(value)                                                                           \
        (llvm::isa<llvm::ConstantInt>(value) && (!llvm::dyn_cast<llvm::ConstantInt>(value)->isZero())) ? \
            irb.CreateCall(instructionFunctions["ARM_mov"], {value}) :                                   \
            (value)

#define MAT_CONST(value) irb.CreateCall(instructionFunctions["ARM_mov"], {value})
            
/// gets the nth operand of the instruction and materializes it if necessary
#define OP_N_MAT(instr, N) \
        MAYBE_MAT_CONST(instr->getOperand(N))

/// get the nth operand of the instruction
#define OP_N(instr, N) \
        instr->getOperand(N)


        std::vector<Pattern> patterns{
            // TODO how do i deal with root alternatives? There the root wouldn't be a root and wouldn't be in the replacementCall map, but still match
            // first idea: return which pattern matched in the match function, and then use that to replace the instruction
            // second idae: just remove alternatives...

            // madd
            Pattern::make_root(
                [](llvm::IRBuilder<>& irb){
                    auto instr = &*irb.GetInsertPoint();
                    // first 2 args is mul, last one is add
                    auto* mul = llvm::dyn_cast<llvm::Instruction>(OP_N(instr,0));
                    auto mulOp1 = OP_N_MAT(mul, 0);
                    auto mulOp2 = OP_N_MAT(mul, 1);

                    auto addOp2 = OP_N_MAT(instr, 1);

                    auto fn = instructionFunctions["ARM_madd"];
                    return irb.CreateCall(fn, {mulOp1, mulOp2, addOp2});
                },
                llvm::Instruction::Add,
                {
                    {llvm::Instruction::Mul, {}, },
                    {},
                }
            ),
            Pattern::make_root(
                [](llvm::IRBuilder<>& irb){
                    auto instr = &*irb.GetInsertPoint();

                    // first 2 args is mul, last one is add
                    auto* mul = llvm::dyn_cast<llvm::Instruction>(OP_N(instr,1));
                    auto mulOp1 = OP_N_MAT(mul,0);
                    auto mulOp2 = OP_N_MAT(mul,1);

                    auto addOp1 = OP_N_MAT(instr,0);

                    auto fn = instructionFunctions["ARM_madd"];
                    return irb.CreateCall(fn, {mulOp1, mulOp2, addOp1});
                },
                llvm::Instruction::Add,
                {
                    {},
                    {llvm::Instruction::Mul, {}},
                }
            ),

            // msub
            Pattern::make_root(
                [](llvm::IRBuilder<>& irb){
                    auto instr = &*irb.GetInsertPoint();

                    // first 2 args is mul, last one is sub
                    auto* mul = llvm::dyn_cast<llvm::Instruction>(OP_N(instr,1));
                    auto mulOp1 = OP_N_MAT(mul,0);
                    auto mulOp2 = OP_N_MAT(mul,1);

                    auto subOp1 = OP_N_MAT(instr,0);

                    auto fn = instructionFunctions["ARM_msub"];
                    return irb.CreateCall(fn, {mulOp1, mulOp2, subOp1});
                },
                llvm::Instruction::Sub,
                {
                    {},
                    {llvm::Instruction::Mul, {}},
                }
            ),

#define TWO_OPERAND_INSTR_PATTERN(llvmInstr, armInstr)                                                               \
            Pattern::make_root(                                                                                      \
                [](llvm::IRBuilder<>& irb){                                                                          \
                    auto instr = &*irb.GetInsertPoint();                                                             \
                    return irb.CreateCall(instructionFunctions[(armInstr)], {OP_N_MAT(instr,0), OP_N_MAT(instr,1)}); \
                },                                                                                                   \
                llvm::Instruction::llvmInstr,                                                                        \
                {}                                                                                                   \
            ),


            // add, sub, mul, div
            TWO_OPERAND_INSTR_PATTERN(Add, "ARM_add") // TODO their second operand can be an immediate, handle that
            TWO_OPERAND_INSTR_PATTERN(Sub, "ARM_sub") 
            Pattern::make_root(
                [](llvm::IRBuilder<>& irb){
                    auto instr = &*irb.GetInsertPoint();
                    return irb.CreateCall(instructionFunctions["ARM_madd"], {OP_N_MAT(instr,0), OP_N_MAT(instr,1), XZR});
                },
                llvm::Instruction::Mul,
                {}
            ),
            TWO_OPERAND_INSTR_PATTERN(SDiv, "ARM_sdiv")
            // remainder
            Pattern::make_root(
                [](llvm::IRBuilder<>& irb){
                    auto instr = &*irb.GetInsertPoint();
                    auto quotient = irb.CreateCall(instructionFunctions["ARM_sdiv"],  {OP_N_MAT(instr,0), OP_N_MAT(instr,1)});
                    return irb.CreateCall(instructionFunctions["ARM_msub"], {quotient, OP_N_MAT(instr,1), OP_N_MAT(instr,0)}); // remainder = numerator - (quotient * denominator)
                },
                llvm::Instruction::SRem,
                {}
            ),

            // shifts
            // logical left shift
            Pattern::make_root(
                [](llvm::IRBuilder<>& irb){
                    auto instr = &*irb.GetInsertPoint();
                    return irb.CreateCall(instructionFunctions["ARM_lsl_imm"], {OP_N_MAT(instr,0), OP_N(instr,1)});
                },
                llvm::Instruction::Shl,
                {
                    {},
                    {Pattern::make_constant()}
                }
            ),
            TWO_OPERAND_INSTR_PATTERN(Shl, "ARM_lsl_var")

            // arithmetic shift right
            Pattern::make_root(
                [](llvm::IRBuilder<>& irb){
                    auto instr = &*irb.GetInsertPoint();
                    return irb.CreateCall(instructionFunctions["ARM_asr_imm"], {OP_N_MAT(instr,0), OP_N(instr,1)});
                },
                llvm::Instruction::AShr,
                {
                    {},
                    {Pattern::make_constant()}
                }
            ),
            TWO_OPERAND_INSTR_PATTERN(AShr, "ARM_asr_var")

            // bitwise ops
            TWO_OPERAND_INSTR_PATTERN(And, "ARM_and")
            TWO_OPERAND_INSTR_PATTERN(Or, "ARM_orr")
            TWO_OPERAND_INSTR_PATTERN(Xor, "ARM_eor")

            // memory

            // alloca
            Pattern::make_root(
                [](llvm::IRBuilder<>& irb){
                    auto stackVar = irb.CreateCall(instructionFunctions["ARM_sub_SP"], {irb.getInt64(8)}); // always exactly 8
                    currentFunctionBytesToFreeAtEnd+=8;

                    return stackVar;
                },
                llvm::Instruction::Alloca // always allocates an i64 in our case
                                           // there is always a store that uses this alloca, but to not delete/replace it, we match the alloca itself
            ),

            // sign extends can only happen after loadsA
            // truncation can only happen before stores
            // load with sign extension
            Pattern::make_root(
                [](llvm::IRBuilder<>& irb){
                    auto* sextInstr     = llvm::dyn_cast<llvm::SExtInst>(&*irb.GetInsertPoint());
                    auto* loadInstr     = llvm::dyn_cast<llvm::LoadInst>(OP_N(sextInstr,0));
                    auto* gepInstr      = llvm::dyn_cast<llvm::GetElementPtrInst>(loadInstr->getPointerOperand());
                    auto* intToPtrInstr = llvm::dyn_cast<llvm::IntToPtrInst>(gepInstr->getPointerOperand());
                    auto bitwidthOfLoad = loadInstr->getType()->getIntegerBitWidth();


                    switch(bitwidthOfLoad){
                    // args: base, offset, offsetshift
                        case 8:
                            return irb.CreateCall(instructionFunctions["ARM_ldr_sb"], {OP_N_MAT(intToPtrInstr,0), OP_N_MAT(gepInstr,1), irb.getInt8(0)});
                        case 16:
                            return irb.CreateCall(instructionFunctions["ARM_ldr_sh"], {OP_N_MAT(intToPtrInstr,0), OP_N_MAT(gepInstr,1), irb.getInt8(1)});
                        case 32:
                            return irb.CreateCall(instructionFunctions["ARM_ldr_sw"], {OP_N_MAT(intToPtrInstr,0), OP_N_MAT(gepInstr,1), irb.getInt8(2)});
                        default: // on 64, the sext is not created by llvm:
                            // TODO some error
                            exit(1);
                    }
                },
                llvm::Instruction::SExt,
                {
                    {
                        llvm::Instruction::Load,
                        {
                            {
                                llvm::Instruction::GetElementPtr,
                                {
                                    {
                                        llvm::Instruction::IntToPtr // int to ptr arg is an arbitrary expression
                                    },
                                    {} // index
                                }
                            }
                        }
                    }
                }
            ),
            // same pattern as above, just without the sign extension
            Pattern::make_root(
                [](llvm::IRBuilder<>& irb){
                    auto* loadInstr     = llvm::dyn_cast<llvm::LoadInst>(&*irb.GetInsertPoint());
                    auto* gepInstr      = llvm::dyn_cast<llvm::GetElementPtrInst>(loadInstr->getPointerOperand());
                    auto* intToPtrInstr = llvm::dyn_cast<llvm::IntToPtrInst>(gepInstr->getPointerOperand());

                    // because it doesn't have a sign extension, it is guaranteed to be a 64 bit load
                    return irb.CreateCall(instructionFunctions["ARM_ldr"], {OP_N_MAT(intToPtrInstr,0), OP_N_MAT(gepInstr,1), irb.getInt8(3)}); // shift by 3 i.e. times 8
                },
                llvm::Instruction::Load,
                {
                    {
                        llvm::Instruction::GetElementPtr,
                        {
                            {
                                llvm::Instruction::IntToPtr // int to ptr arg is an arbitrary expression
                            },
                            {} // index
                        }
                    }
                }
            ),

            // store with truncation
            Pattern::make_root(
                [](llvm::IRBuilder<>& irb){
                    auto* storeInst     = llvm::dyn_cast<llvm::StoreInst>(&*irb.GetInsertPoint());
                    auto* gepInstr      = llvm::dyn_cast<llvm::GetElementPtrInst>(storeInst->getPointerOperand());
                    auto* intToPtrInstr = llvm::dyn_cast<llvm::IntToPtrInst>(gepInstr->getPointerOperand());
                    auto* truncInstr    = llvm::dyn_cast<llvm::TruncInst>(storeInst->getValueOperand());
                    auto bitwidthOfStore = truncInstr->getType()->getIntegerBitWidth();
                    
                    switch(bitwidthOfStore){
                    // args: value, base, offset, offsetshift
                        case 8:
                            return irb.CreateCall(instructionFunctions["ARM_str32_b"], {OP_N_MAT(truncInstr,0), OP_N_MAT(intToPtrInstr,0), OP_N_MAT(gepInstr,1), irb.getInt8(0)});
                        case 16:
                            return irb.CreateCall(instructionFunctions["ARM_str32_h"], {OP_N_MAT(truncInstr,0), OP_N_MAT(intToPtrInstr,0), OP_N_MAT(gepInstr,1), irb.getInt8(1)});
                        case 32:
                            return irb.CreateCall(instructionFunctions["ARM_str32"], {OP_N_MAT(truncInstr,0), OP_N_MAT(intToPtrInstr,0), OP_N_MAT(gepInstr,1), irb.getInt8(2)});
                        default: // on 64, the trunc is not created by llvm:
                            // TODO some error
                            exit(1);
                    }
                },
                llvm::Instruction::Store,
                { 
                    // store arg is cast to the target type, so truncated, or target is already i64. This case handles truncation
                    {
                        llvm::Instruction::Trunc
                    },
                    {
                        llvm::Instruction::GetElementPtr,
                        {
                            {
                                llvm::Instruction::IntToPtr
                            },
                            {} // index
                        }
                    },
                }
            ),
            // store without truncation
            Pattern::make_root(
                [](llvm::IRBuilder<>& irb){
                    auto* storeInst     = llvm::dyn_cast<llvm::StoreInst>(&*irb.GetInsertPoint());
                    auto* gepInstr      = llvm::dyn_cast<llvm::GetElementPtrInst>(storeInst->getPointerOperand());
                    auto* intToPtrInstr = llvm::dyn_cast<llvm::IntToPtrInst>(gepInstr->getPointerOperand());
                    // without truncaton -> no bitwidth check necessary
                    return irb.CreateCall(instructionFunctions["ARM_str"], {OP_N_MAT(storeInst, 0), OP_N_MAT(intToPtrInstr, 0), OP_N_MAT(gepInstr,1), irb.getInt8(3)}); // shift by 3 for multiplying by 8
                },
                llvm::Instruction::Store,
                { 
                    // store arg is cast to the target type, so truncated, or target is already i64. This case handles non-truncation i.e. arbitrary first (value) arg
                    {},
                    {
                        llvm::Instruction::GetElementPtr,
                        {
                            {
                                llvm::Instruction::IntToPtr
                            },
                            {} // index
                        }
                    },
                }
            ),

            Pattern::make_root(
                [](llvm::IRBuilder<>& irb){
                    auto instr = &*irb.GetInsertPoint();
                    return irb.CreateCall(instructionFunctions["ARM_ldr"], {OP_N_MAT(instr,0)});
                },
                llvm::Instruction::Load,
                {}
            ),
            Pattern::make_root(
                [](llvm::IRBuilder<>& irb){
                    auto instr = &*irb.GetInsertPoint();
                    return irb.CreateCall(instructionFunctions["ARM_str"], {OP_N_MAT(instr,0), OP_N_MAT(instr,1)});
                },
                llvm::Instruction::Store,
                {}
            ),

            // subscript with addrof
            Pattern::make_root(
                [](llvm::IRBuilder<>& irb){
                    // add instruction for address calculation
                    auto* ptrToInt = llvm::dyn_cast<llvm::PtrToIntInst>(irb.GetInsertPoint());
                    auto* gep = llvm::dyn_cast<llvm::GetElementPtrInst>(ptrToInt->getPointerOperand());
                    auto* intToPtr = llvm::dyn_cast<llvm::IntToPtrInst>(gep->getOperand(0));

                    unsigned shiftInt = 0;
                    switch(gep->getSourceElementType()->getIntegerBitWidth()){
                        case 8: shiftInt = 0; break;
                        case 16: shiftInt = 1; break;
                        case 32: shiftInt = 2; break;
                        case 64: shiftInt = 3; break;
                        default:
                            //TODO error
                            exit(1);
                    }

                    auto indexOp = OP_N(gep, 1);
                    DEBUGLOG("indexOp: " << *indexOp);
                    llvm::ConstantInt* indexConst;
                    // TODO is there any problem with the constant not being deleted?
                    if((indexConst = llvm::dyn_cast_or_null<llvm::ConstantInt>(indexOp))!= nullptr){
                        auto index = indexConst->getSExtValue();
                        return irb.CreateCall(instructionFunctions["ARM_add"], {OP_N_MAT(intToPtr,0), irb.getInt64(index << shiftInt)});
                    } else {
                        return irb.CreateCall(instructionFunctions["ARM_add_SHIFT"], {OP_N_MAT(intToPtr, 0), indexOp, irb.getInt64(shiftInt)});
                    }
                },
                llvm::Instruction::PtrToInt,
                {
                    {
                        llvm::Instruction::GetElementPtr,
                        {
                            {
                                llvm::Instruction::IntToPtr
                            },
                            {} // index operand can be arbitrary
                        }
                    }
                }
            ),

            // ptr to int 
            Pattern::make_root(
                [](llvm::IRBuilder<>& irb){
                    auto instr = &*irb.GetInsertPoint();
                    return irb.CreateCall(instructionFunctions["ARM_mov"], {OP_N_MAT(instr,0)}); // in reality, it would just be deleted
                },
                llvm::Instruction::PtrToInt,
                {}
            ),


            // control flow/branches
            // conditional branches always have an icmp NE as their condition, if we match them before the unconditional ones, the plain Br match without children always matches only unconditional ones
            Pattern::make_root(
                [](llvm::IRBuilder<>& irb){
                    auto instr      = &*irb.GetInsertPoint();
                    auto cond       = OP_N_MAT(instr,0);
                    auto innerCond  =
                        llvm::dyn_cast<llvm::ICmpInst>(
                            llvm::dyn_cast<llvm::ZExtInst>(
                                llvm::dyn_cast<llvm::ICmpInst>(cond)->getOperand(0))->getOperand(0));
                    auto pred       = innerCond->getPredicate();
                    auto predStr    = llvm::ICmpInst::getPredicateName(pred);

                    // get llvm string literal which displayes predStr
                    auto predStrLiteral = llvm::ConstantDataArray::getString(ctx, predStr);

                    auto trueBlock  = llvm::dyn_cast<llvm::BasicBlock>(OP_N(instr,1));
                    auto falseBlock = llvm::dyn_cast<llvm::BasicBlock>(OP_N(instr,2));

                    auto cmp = irb.CreateCall(instructionFunctions["ARM_cmp"], {OP_N_MAT(innerCond,0), OP_N_MAT(innerCond,1)});
                    auto call = irb.CreateCall(instructionFunctions["ARM_b_cond"], {cmp, predStrLiteral});
                    // reinsert the branch (with its operand as the call), this is an exception to the rule, it cannot be removed, it might seem stupid, but its only a consequence of modeling acutal arm assembly in llvm
                    irb.CreateCondBr(call, trueBlock, falseBlock); // TODO this could make problems with use checking before deletion...
                    return call;
                },
                llvm::Instruction::Br, // conditional branch after 'normal' comparison
                {
                    {
                        llvm::Instruction::ICmp, 
                        {
                            {
                                llvm::Instruction::ZExt,
                                {
                                    llvm::Instruction::ICmp
                                }
                            },
                            {}
                        }
                    },
                    {},
                    {},
                }
            ),
            // icmp before conditional branch
            Pattern::make_root(
                [](llvm::IRBuilder<>& irb){
                    auto instr      = &*irb.GetInsertPoint();
                    auto cond       = llvm::dyn_cast<llvm::ICmpInst>(OP_N(instr,0));
                    auto condInner  = OP_N_MAT(cond,0);
                    auto trueBlock  = llvm::dyn_cast<llvm::BasicBlock>(OP_N(instr,1));
                    auto falseBlock = llvm::dyn_cast<llvm::BasicBlock>(OP_N(instr,2));

                    auto call = irb.CreateCall(instructionFunctions["ARM_b_cbnz"], {condInner});
                    // reinsert the branch (with its operand as the call), this is an exception to the rule, it cannot be removed, it might seem stupid, but its only a consequence of modeling acutal arm assembly in llvm
                    irb.CreateCondBr(call, trueBlock, falseBlock); // TODO this could make problems with use checking before deletion...
                    return call;
                },
                llvm::Instruction::Br,
                {
                    {llvm::Instruction::ICmp}, // no requirements, because it can only be NE
                    {},
                    {},
                }
            ),
            Pattern::make_root(
                [](llvm::IRBuilder<>& irb){
                    auto instr = dyn_cast<llvm::BranchInst>(&*irb.GetInsertPoint());
                    auto call = irb.CreateCall(instructionFunctions["ARM_b"], {OP_N(instr,0)});
                    // cannot be conditional branch, because that always has an icmp NE as its condition
                    // reinsert the branch (with its operand as the call), this is an exception to the rule, it cannot be removed, it might seem stupid, but its only a consequence of modeling acutal arm assembly in llvm
                    //irb.CreateBr(instr->getSuccessor(0)); // TODO this could make problems with use checking before deletion...
                    return call;
                },
                llvm::Instruction::Br, 
                {},
                true
            ),

            // icmp (also almost all possible ZExts, they're (almost) exclusively used for icmps, only once for a phi in the short circuiting logical ops)
            Pattern::make_root(
                [](llvm::IRBuilder<>& irb){
                    auto* zextInstr = llvm::dyn_cast<llvm::ZExtInst>(&*irb.GetInsertPoint());
                    auto* icmpInstr = llvm::dyn_cast<llvm::ICmpInst>(OP_N(zextInstr, 0));

                    auto pred       = icmpInstr->getPredicate();
                    auto predStr    = llvm::ICmpInst::getPredicateName(pred);

                    // get llvm string literal which displayes predStr
                    auto predStrLiteral = llvm::ConstantDataArray::getString(ctx, predStr);

                    irb.CreateCall(instructionFunctions["ARM_cmp"], {OP_N_MAT(icmpInstr, 0), OP_N_MAT(icmpInstr, 1)});

                    return irb.CreateCall(instructionFunctions["ARM_csel"], {MAT_CONST(irb.getInt64(1)), XZR, predStrLiteral}); // TODO args
                },
                llvm::Instruction::ZExt,
                {
                    {llvm::Instruction::ICmp}
                }
            ),
            // raw icmp without ZExt
            Pattern::make_root(
                [](llvm::IRBuilder<>& irb){
                    auto* icmpInstr = llvm::dyn_cast<llvm::ICmpInst>(&*irb.GetInsertPoint());

                    auto pred       = icmpInstr->getPredicate();
                    auto predStr    = llvm::ICmpInst::getPredicateName(pred);

                    // get llvm string literal which displayes predStr
                    auto predStrLiteral = llvm::ConstantDataArray::getString(ctx, predStr);

                    irb.CreateCall(instructionFunctions["ARM_cmp"], {OP_N_MAT(icmpInstr, 0), OP_N_MAT(icmpInstr, 1)});

                    return irb.CreateCall(instructionFunctions["ARM_csel_i1"], {MAT_CONST(irb.getInt64(1)), XZR, predStrLiteral});
                },
                llvm::Instruction::ICmp,
                {}
            ),

            // TODO test
            // ZExt/PHI: only matched in order to match something for this ZExt, it will become a register later anyway
            Pattern::make_root(
                [](llvm::IRBuilder<>& irb){
                    auto* zextInstr = llvm::dyn_cast<llvm::ZExtInst>(&*irb.GetInsertPoint());
                    auto* phiInstr = llvm::dyn_cast<llvm::PHINode>(OP_N(zextInstr, 0)); // noDelete = true -> we can use this as is


                    return irb.CreateCall(instructionFunctions["ZExt_handled_in_Reg_Alloc"], {phiInstr});
                },
                llvm::Instruction::ZExt,
                {
                    {llvm::Instruction::PHI, {}, true}
                }
            ),

            // TODO ZExt for short circuiting
        };

        // TODO sort by size descending
        //std::sort(patterns.begin(), patterns.end(), [](auto& a, auto& b){
        //    return a.totalSize > b.totalSize;
        //});


        for(auto& fn : moduleUP->functions()){
            if(fn.isDeclaration()) continue;

            currentFunctionBytesToFreeAtEnd = 0;
            matchPatterns(&fn, patterns);
            // irb starts at terminator of last block

            if(currentFunctionBytesToFreeAtEnd==0) continue;

            // inserting add sp, sp, #bytesToFreeAtEnd before all returns
            llvm::IRBuilder<> irb{ctx};
            for(auto& block : fn){
                auto term = block.getTerminator();
                if(llvm::isa_and_present<llvm::ReturnInst>(term)){
                    irb.SetInsertPoint(term);
                    irb.CreateCall(instructionFunctions["ARM_add_SP"], {irb.getInt64(currentFunctionBytesToFreeAtEnd)}); // takes immediate
                }
            }
        }

        moduleUP->print(llvm::outs(), nullptr);
        bool moduleIsBroken = llvm::verifyModule(*moduleUP, &llvm::errs());
        if(moduleIsBroken) llvm::errs() << "ISel broke module :(\n";
    }
}


int main(int argc, char *argv[])
{
    try{
        auto parsedArgs = ArgParse::parse(argc, argv);

        if(parsedArgs.contains(ArgParse::possible.help)){
            ArgParse::printHelp();
            return 0;
        }

        std::string inputFilename = parsedArgs.at(ArgParse::possible.input);
        int iterations = 1;
        if(parsedArgs.contains(ArgParse::possible.iterations)){
            iterations = std::stoi(parsedArgs.at(ArgParse::possible.iterations));
        }

        if(access(inputFilename.c_str(),R_OK) != 0){
            perror("Could not open input file");
            return 1;
        }

        std::ifstream inputFile{inputFilename};
        if(parsedArgs.contains(ArgParse::possible.preprocess)){
            //this is a bit ugly, but it works
            std::stringstream ss;

            auto epochsecs = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count(); //cpp moment
            ss << "cpp -E -P " << parsedArgs.at(ArgParse::possible.input) << " > /tmp/" << epochsecs << ".bpreprocessed";
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

        if(parsedArgs.contains(ArgParse::possible.preprocess)) system("rm /tmp/*.bpreprocessed");

        auto semanalyzeStart = std::chrono::high_resolution_clock::now();
        if(!parsedArgs.contains(ArgParse::possible.nosemantic)){
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

        if(parsedArgs.contains(ArgParse::possible.print) || parsedArgs.contains(ArgParse::possible.dot)){
            if(parsedArgs.contains(ArgParse::possible.url)){
                std::stringstream ss;
                ast->printDOT(ss);
                auto compactSpacesRegex = std::regex("\\s+");
                auto str = std::regex_replace(ss.str(), compactSpacesRegex, " ");
                std::cout << "https://dreampuf.github.io/GraphvizOnline/#" << url_encode(str) << std::endl;
            }else if(parsedArgs.contains(ArgParse::possible.dot)){
                if(parsedArgs.contains(ArgParse::possible.output)){
                    std::ofstream outputFile{parsedArgs.at(ArgParse::possible.output)};
                    ast->printDOT(outputFile);
                    outputFile.close();
                }else{
                    ast->printDOT(std::cout);
                }
            }else{
                ast->print(std::cout);
            }
        }else if(parsedArgs.contains(ArgParse::possible.llvm)){
            if(parsedArgs.contains(ArgParse::possible.output)){
                std::error_code errorCode;
                llvm::raw_fd_ostream outputFile{parsedArgs.at(ArgParse::possible.output), errorCode};
                if(!(genSuccess = Codegen::generate(*ast, outputFile))){
                    llvm::errs() << "Codegen failed\nIndividual errors displayed above\n";
                }
                outputFile.close();
            }else{
                if(!(genSuccess = Codegen::generate(*ast, llvm::outs()))){
                    llvm::errs() << "Codegen failed\nIndividual errors displayed above\n";
                }
                // TODO call ISel
                Codegen::ISel::test();
            }
        }
        //print execution times
        if(parsedArgs.contains(ArgParse::possible.benchmark)){
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
