#include <iostream>
#include <cassert>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <regex>
#include <tuple>
#include <fstream>
#include <charconv>
#include <optional>
#include <utility>
#include <filesystem>

#include <err.h>
#include <sys/wait.h>

#include "frontend.h"

// ---- Token ----

string Token::toString() {
    if (type == Type::NUM || type == Type::IDENTIFIER) {
        return string{value};
    } else {
        return Token::toString(type);
    }
}

bool Token::operator==(const Token& other) const {
    return type == other.type && value == other.value;
}

string Token::toString(Token::Type type) {
    switch (type) {
        case Type::EMPTY:         return "EMPTY";
        case Type::NUM:           return "NUM";
        case Type::IDENTIFIER:    return "IDENTIFIER";
        case Type::KW_AUTO:       return "auto";
        case Type::KW_REGISTER:   return "register";
        case Type::KW_IF:         return "if";
        case Type::KW_ELSE:       return "else";
        case Type::KW_WHILE:      return "while";
        case Type::KW_RETURN :    return " return ";
        case Type::SEMICOLON:     return ";";
        case Type::COMMA:         return ",";
        case Type::L_PAREN:       return "(";
        case Type::R_PAREN:       return ")";
        case Type::L_BRACKET:     return "[";
        case Type::R_BRACKET:     return "]";
        case Type::L_BRACE:       return "{";
        case Type::R_BRACE:       return "}";
        case Type::LOGICAL_NOT:   return "!";
        case Type::TILDE:         return "~";
        case Type::AMPERSAND:     return "&";
        case Type::BITWISE_OR:    return "|";
        case Type::BITWISE_XOR:   return "^";
        case Type::AT:            return "@";
        case Type::PLUS:          return "+";
        case Type::MINUS:         return "-";
        case Type::TIMES:         return "*";
        case Type::DIV:           return "/";
        case Type::MOD:           return "%";
        case Type::SHIFTL:        return "<<";
        case Type::SHIFTR:        return ">>";
        case Type::LESS:          return "<";
        case Type::GREATER:       return ">";
        case Type::LESS_EQUAL:    return "<=";
        case Type::GREATER_EQUAL: return ">=";
        case Type::EQUAL:         return "==";
        case Type::NOT_EQUAL:     return "!=";
        case Type::ASSIGN:        return "=";
        case Type::LOGICAL_AND:   return "&&";
        case Type::LOGICAL_OR:    return "||";
        case Type::EOP:           return "EOP";
    }
    return "UNKNOWN";
}

const Token emptyToken{
    Token::Type::EMPTY,
    ""
};

ParsingException::ParsingException(string msg) : std::runtime_error(msg) {}

// ---- Tokenizer ----

Tokenizer::UnexpectedTokenException::UnexpectedTokenException(Tokenizer& tokenizer, Token::Type expected)
    : ParsingException(exceptionString(tokenizer, expected)) {}

std::string Tokenizer::UnexpectedTokenException::exceptionString(Tokenizer& tokenizer, Token::Type expected) {
    tokenizer.peekToken();
    std::string typeHint = "";
    if ((tokenizer.peeked).type == Token::Type::NUM || (tokenizer.peeked).type == Token::Type::IDENTIFIER) 
        typeHint = " (type: " + Token::toString((tokenizer.peeked).type) + ")";

    std::string expectedHint = "";
    if (expected != Token::Type::EMPTY) 
        expectedHint = ", expected: " + Token::toString(expected);

    return "Line " + std::to_string(tokenizer.getLineNum()) + ": " +
        "Unexpected token: " + (tokenizer.peeked).toString() + typeHint +
        expectedHint;
}

Tokenizer::Tokenizer(std::ifstream& inputFile) : matched(emptyToken), prog(initProg(inputFile)), peeked(emptyToken) {}

Tokenizer::Tokenizer(string&& prog) : matched(emptyToken), prog(std::move(prog)), peeked(emptyToken) {}

string Tokenizer::initProg(std::ifstream& inputFile) {
    return string(std::istreambuf_iterator(inputFile), {}) +
        " "; // to safely peek for two char ops at the end
}

const std::regex Tokenizer::numberRegex{"[0-9]+"};
const std::regex Tokenizer::identifierRegex{"[a-zA-Z_][a-zA-Z0-9_]*"};
const std::unordered_map<string, Token::Type> Tokenizer::keywords = {
    {"auto",                     Token::Type::KW_AUTO},
    {"register",                 Token::Type::KW_REGISTER},
    {"if",                       Token::Type::KW_IF},
    {"else",                     Token::Type::KW_ELSE},
    {"while",                    Token::Type::KW_WHILE},
    {"return",                   Token::Type::KW_RETURN},
};

const string Tokenizer::newline{"\n"};

Token Tokenizer::peekToken() {
    using Type = Token::Type;

    if (peeked != emptyToken) return peeked;

    // return EOP token
    if (progI == string::npos || progI >= prog.size())
        return peeked = Type::EOP;


    // skip whitespace & comments
    while (true) {
        progI = prog.find_first_not_of(" \f\n\r\t\v", progI); // same chars that isspace uses
        if (progI == string::npos || progI >= prog.size()) 
            return peeked = Type::EOP;

        if (prog[progI] == '/') {
            if (progI + 1 < prog.size()&& prog[progI + 1] == '/') {
                // single line comment
                progI += 2;

                progI = prog.find(newline, progI);
                progI += newline.size();
            } else if (progI + 1 >= prog.size()) {
                return peeked = Type::EOP;
            } else {
                break;
            }
        } else {
            break;
        }
    }

    if (prog.size() - progI <= 0)
        return peeked = Type::EOP;

    // NUM
    if (isdigit(prog[progI])) {
        std::smatch match;
        if (std::regex_search(prog.cbegin() + progI, prog.cend(), match,
                    numberRegex)) {
            auto numStrMatch = match[0];
            auto len = numStrMatch.length();
            progI += len;
            return peeked = {Type::NUM,
                string_view(numStrMatch.first.base(), len)};
        }
    }

    // IDENTIFIER
    if (isalpha(prog[progI]) || prog[progI] == '_') {
        std::smatch match;
        if (std::regex_search(prog.cbegin() + progI, prog.cend(), match,
                    identifierRegex)) {
            auto idStrMatch = match[0];
            auto len = idStrMatch.length();
            progI += len;
            // check if it's a keyword
            if (auto keywordTokenIt = keywords.find(idStrMatch.str());
                    keywordTokenIt != keywords.end()) {
                return peeked = keywordTokenIt->second;
            } else {
                return peeked = {Type::IDENTIFIER,
                    string_view(idStrMatch.first.base(), len)};
            }
        }
    }

    Type type = Token::Type::EMPTY;

    switch (prog[progI]) {
        // single characters
        // parentheses, brackets, braces, unabiguous operators, ...
        case '(': type = Type::L_PAREN;     break;
        case ')': type = Type::R_PAREN;     break;
        case '[': type = Type::L_BRACKET;   break;
        case ']': type = Type::R_BRACKET;   break;
        case '{': type = Type::L_BRACE;     break;
        case '}': type = Type::R_BRACE;     break;
        case '~': type = Type::TILDE;       break;
        case '^': type = Type::BITWISE_XOR; break;
        case '@': type = Type::AT;          break;
        case '+': type = Type::PLUS;        break;
        case '-': type = Type::MINUS;       break;
        case '*': type = Type::TIMES;       break;
        case '/': type = Type::DIV;         break;
        case '%': type = Type::MOD;         break;
        case ';': type = Type::SEMICOLON;   break;
        case ',': type = Type::COMMA;       break;

            // ambiguous one/two character operators
        case '<':
            if (prog[progI + 1] == '=')      type = Type::LESS_EQUAL, progI++;
            else if (prog[progI + 1] == '<') type = Type::SHIFTL,     progI++;
            else                             type = Type::LESS;
            break;
        case '>':
            if (prog[progI + 1] == '=')      type = Type::GREATER_EQUAL, progI++;
            else if (prog[progI + 1] == '>') type = Type::SHIFTR,        progI++;
            else                             type = Type::GREATER;
            break;
        case '=':
            if (prog[progI + 1] == '=')      type = Type::EQUAL,       progI++;
            else                             type = Type::ASSIGN;
            break;
        case '&':
            if (prog[progI + 1] == '&')      type = Type::LOGICAL_AND, progI++;
            else                             type = Type::AMPERSAND;
            break;
        case '|':
            if (prog[progI + 1] == '|')      type = Type::LOGICAL_OR,  progI++;
            else                             type = Type::BITWISE_OR;
            break;
        case '!':
            if (prog[progI + 1] == '=')      type = Type::NOT_EQUAL,   progI++;
            else                             type = Type::LOGICAL_NOT;
            break;
    }

    if (type != Type::EMPTY) {
        progI++;
        return peeked = type;
    }

    // invalid character
    throw std::runtime_error("Invalid character: "s + prog[progI + 0]);
}

std::uint64_t Tokenizer::getLineNum() {
    if (progI == string::npos) {
        return std::count(prog.begin(), prog.end(), '\n') + 1;
    }
    // this should work on windows too, because '\r\n' also contains '\n', but
    // honestly if windows users have wrong line numbers in their errors, so be
    // it :P
    return std::count(prog.begin(), prog.begin() + progI, '\n') + 1;
}

Token Tokenizer::nextToken() {
    Token tok = peekToken();
    peeked = emptyToken;
    return tok;
}

bool Tokenizer::matchToken(Token::Type type, bool advance) {
    Token tok = peekToken();
    if (tok.type == type) {
        matched = tok;
        if (advance)
                nextToken();
        return true;
    } else {
        matched = emptyToken;
        return false;
    }
}

void Tokenizer::assertToken(Token::Type type, bool advance) {
    if (!matchToken(type, advance))
        throw UnexpectedTokenException(*this, type);
}

void Tokenizer::assertNotToken(Token::Type type, bool advance) {
    if (matchToken(type, advance))
        throw UnexpectedTokenException(*this, type);
}

void Tokenizer::reset() {
    matched = emptyToken;
    peeked = emptyToken;
    progI = 0;
}

// ---- IdentifierInfo ----

IdentifierInfo::Type IdentifierInfo::fromTokenType(const Token::Type t) {
    switch (t) {
        case Token::Type::KW_AUTO:     return AUTO;
        case Token::Type::KW_REGISTER: return REGISTER;
        default:                       return UNKNOWN;
    }
}

string IdentifierInfo::toString(Type t) {
    switch (t) {
        case REGISTER: return "register";
        case AUTO:     return "auto";
        case FUNCTION: return "function";
        case UNKNOWN:
        default:       return "unknown";
    }
}

// ---- ASTNode ----

size_t ASTNode::Hash::operator()(const ASTNode& node) const {
    return node.nodeID;
}

int ASTNode::nodeCounter = 0;

const InsertOnceQueryAfterwardsMap<ASTNode::Type, int> ASTNode::numberOfChildren{{
    {ASTNode::Type::NRoot,          -1},
    {ASTNode::Type::NFunction,       2},
    {ASTNode::Type::NParamList,     -1},
    {ASTNode::Type::NStmtDecl,       1},
    {ASTNode::Type::NStmtReturn,    -1},
    {ASTNode::Type::NStmtBlock,     -1},
    {ASTNode::Type::NStmtWhile,      2},
    {ASTNode::Type::NStmtIf,        -1},
    {ASTNode::Type::NExprVar,        0},
    {ASTNode::Type::NExprNum,        0},
    {ASTNode::Type::NExprCall,      -1},
    {ASTNode::Type::NExprUnOp,       1},
    {ASTNode::Type::NExprBinOp,      2},
    {ASTNode::Type::NExprSubscript,  3},
}};

const InsertOnceQueryAfterwardsMap<ASTNode::Type, std::string> ASTNode::nodeTypeToDotIdentifier{{
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
}};

const InsertOnceQueryAfterwardsMap<ASTNode::Type, llvm::SmallVector<std::string,4>> ASTNode::nodeTypeToDotStyling{{
    {ASTNode::Type::NRoot,        {"shape=house",          "style=filled",        "fillcolor=lightgrey"}},
    {ASTNode::Type::NFunction,    {"shape=box",            "style=filled",        "fillcolor=lightblue"}},
    {ASTNode::Type::NParamList,   {"shape=invtriangle"}},
    {ASTNode::Type::NStmtBlock,   {"shape=invtriangle",    "style=filled",        "fillcolor=grey"}},
    {ASTNode::Type::NExprUnOp,    {"style=filled",         "color=chocolate3"}},
    {ASTNode::Type::NExprBinOp,   {"style=filled",         "color=chocolate1"}},
    {ASTNode::Type::NExprVar,     {"style=filled",         "color=lightblue1"}},
    {ASTNode::Type::NStmtDecl,    {"shape=rectangle"}},
    {ASTNode::Type::NStmtIf,      {"shape=rectangle"}},
    {ASTNode::Type::NStmtReturn,  {"shape=rectangle"}},
    {ASTNode::Type::NStmtWhile,   {"shape=rectangle"}},
}};

std::ostream& operator<<(std::ostream& out, ASTNode& node) {
    node.printDOT(out);
    return out;
}

ASTNode::ASTNode(Type type, string_view name, IdentifierInfo::Type t) : type(type), ident({name, 0, t}) {}
ASTNode::ASTNode(Type type, string_view name, Token::Type t)          : ASTNode(type, name, IdentifierInfo::fromTokenType(t)) {}
ASTNode::ASTNode(Type type, int64_t value)                            : type(type), value(value) {}
ASTNode::ASTNode(Type type, Token::Type op)                           : type(type), op(op) {}
ASTNode::ASTNode(Type type, llvm::ArrayRef<ASTNode> children)         : type(type), children(children) {}

string ASTNode::uniqueDotIdentifier() const {
    return nodeTypeToDotIdentifier[type] + "_" + std::to_string(nodeID);
}

string ASTNode::toString() const {
    if (type == Type::NExprNum) {
        return std::to_string(value);
    } else if (type == Type::NExprBinOp || type == Type::NExprUnOp) {
        return Token::toString(op);
    } else if (type == Type::NStmtDecl || type == Type::NExprVar) {
        return nodeTypeToDotIdentifier[type] + "(" + IdentifierInfo::toString(ident.type) + ")" + ": " + string(ident.name);
    } else {
        return nodeTypeToDotIdentifier[type];
    }
}

void ASTNode::printDOT(std::ostream& out, int indentDepth, bool descend) const {
    string indent(4 * (indentDepth + 1), ' ');

    out << indent << uniqueDotIdentifier() << " [label=\"" << toString()
        << "\"";
    if (nodeTypeToDotStyling.contains(type)) {
        for (auto& styleInstr : nodeTypeToDotStyling[type]) {
                out << ", " << styleInstr;
        }
    }
    out << "];" << std::endl;

    if (descend) {
        for (auto& child : children) {
                out << indent << uniqueDotIdentifier() << " -> "
                    << child.uniqueDotIdentifier() << ";" << std::endl;
                child.printDOT(out, indentDepth + 1);
        }
    }
}

// ---- AST ----
void AST::printDOT(std::ostream& out) {
    out << "digraph AST {" << std::endl;
    root.printDOT(out, 0, false);
    for (auto& child : root.children) {
        out << root.uniqueDotIdentifier() << " -> " << child.uniqueDotIdentifier() << ";" << std::endl;
        out << "subgraph cluster_" << child.uniqueDotIdentifier() << " {" << std::endl;
        // function cluster styling

        out << "style=filled;" << std::endl;
        out << "color=lightgrey;" << std::endl;
        out << "node [style=filled,color=white];" << std::endl;
        out << "label = \"" << child.toString() << "\";" << std::endl;
        child.printDOT(out, 1);
        out << "}" << std::endl;
    }

    out << "}" << std::endl;
}

size_t AST::getRoughMemoryFootprint() {
    size_t totalSize = 0;
    root.iterateChildren([&totalSize](ASTNode& node) {
        totalSize += sizeof(node);
        // seperately add the size of the children vector
        totalSize += node.children.capacity() * sizeof(ASTNode);
    });
    return totalSize;
}

bool AST::validate() {
    bool valid = true;
    root.iterateChildren([&valid](ASTNode& node) {
        auto number = ASTNode::numberOfChildren[node.type];
        if (number > 0 && node.children.size() != static_cast<size_t>(number)) {
            std::cerr << "Node " << node.uniqueDotIdentifier() << " has " << node.children.size() << " children, but should have " << number << "\n";
            valid = false;
        }else{
            auto atLeast = number-1;
            if(node.children.size() < static_cast<size_t>(atLeast)){
                std::cerr << "Node " << node.uniqueDotIdentifier() << " has " << node.children.size() << " children, but should have at least " << atLeast << "\n";
                valid = false;
            }
        }
    });
    return valid;
}

// ---- Parser ----

Parser::Parser(string&& prog) : tok{std::move(prog)} {}
Parser::Parser(std::ifstream& inputFile) : tok{inputFile} {}

ASTNode Parser::parseStmtDecl(){
    assert((tok.matched.type == Token::Type::KW_AUTO || tok.matched.type == Token::Type::KW_REGISTER) && "parseStmtDecl called on non-declaration token");

    auto registerOrAuto = tok.matched.type;

    tok.assertToken(Token::Type::IDENTIFIER);
    auto varName = tok.matched.value;

    tok.assertToken(Token::Type::ASSIGN);

    //if this returns normally, it means we have a valid initializer
    auto initializer = parseExpr();

    tok.assertToken(Token::Type::SEMICOLON);
    auto decl = ASTNode(ASTNode::Type::NStmtDecl, varName, registerOrAuto);

    decl.children.emplace_back(std::move(initializer));
    return decl;
}

ASTNode Parser::parseStmtReturn(){
    assert(tok.matched.type == Token::Type::KW_RETURN && "parseStmtReturn called on non-return token");

    auto returnStmt= ASTNode(ASTNode::Type::NStmtReturn);
    if(tok.matchToken(Token::Type::SEMICOLON)){
        return returnStmt;
    }else{
        returnStmt.children.emplace_back(parseExpr()); //parseExpr throws exception in the case of parsing error
        tok.assertToken(Token::Type::SEMICOLON);
        return returnStmt;
    }
}

ASTNode Parser::parseBlock(){
    assert(tok.matched.type == Token::Type::L_BRACE && "parseBlock called on non-block token");

    auto block = ASTNode(ASTNode::Type::NStmtBlock);
    while(!tok.matchToken(Token::Type::R_BRACE)){
        try{
            block.children.emplace_back(parseStmt());
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

ASTNode Parser::parseStmtIfWhile(bool isWhile){
    assert(((isWhile && tok.matched.type == Token::Type::KW_WHILE) || (!isWhile && tok.matched.type == Token::Type::KW_IF)) && "parseStmtIfWhile called on non-if/while token");

    tok.assertToken(Token::Type::L_PAREN);

    auto condition = parseExpr();

    tok.assertToken(Token::Type::R_PAREN);

    auto body = parseStmt();
    auto ifWhileStmt = ASTNode(isWhile ? ASTNode::Type::NStmtWhile : ASTNode::Type::NStmtIf);
    ifWhileStmt.children.push_back(std::move(condition));
    ifWhileStmt.children.push_back(std::move(body));

    if(!isWhile && tok.matchToken(Token::Type::KW_ELSE)){
        ifWhileStmt.children.emplace_back(parseStmt());
    }
    return ifWhileStmt;
}

ASTNode Parser::parseStmt(){
    if(tok.matchToken(Token::Type::KW_RETURN)){
        return parseStmtReturn();
    }else if(tok.matchToken(Token::Type::KW_IF) || tok.matchToken(Token::Type::KW_WHILE)){
        return parseStmtIfWhile(tok.matched.type == Token::Type::KW_WHILE);
    }else if(tok.matchToken(Token::Type::KW_REGISTER) || tok.matchToken(Token::Type::KW_AUTO)){
        return parseStmtDecl();
    }else if(tok.matchToken(Token::Type::L_BRACE)){
        return parseBlock();
    }else{
        auto expr = parseExpr();
        tok.assertToken(Token::Type::SEMICOLON);

        return expr;
    }
}

ASTNode Parser::parsePrimaryExpression(){
    //- numbers
    //- unary ops
    //- variables
    //- calls
    //- parenthesized expressions

    if(tok.matchToken(Token::Type::NUM)){
        auto num = ASTNode(ASTNode::Type::NExprNum);

        const auto str = tok.matched.value;
        auto [ptr, err] = std::from_chars(str.data(), str.data() + str.size(), num.value);

        if (!(err == std::errc{} && ptr == str.data() + str.size())) {
            num.value = 0;
            std::cerr << "Line " << tok.getLineNum() << ": Warning: number " << tok.matched.value << " is out of range and will be truncated to 0" << std::endl;
        }
        return num;
    }else if(tok.matchToken(Token::Type::TILDE)||tok.matchToken(Token::Type::MINUS)||tok.matchToken(Token::Type::LOGICAL_NOT)||tok.matchToken(Token::Type::AMPERSAND)){ 
        auto unOp = ASTNode(ASTNode::Type::NExprUnOp, tok.matched.type);
        unOp.children.emplace_back(parseExpr(13)); //unary ops have 13 prec, rassoc
        return unOp;
    }else if(tok.matchToken(Token::Type::IDENTIFIER)){
        auto identStrV = tok.matched.value;
        if(tok.matchToken(Token::Type::L_PAREN)){
            auto call = ASTNode(ASTNode::Type::NExprCall, identStrV);
            while(!tok.matchToken(Token::Type::R_PAREN)){
                call.children.emplace_back(parseExpr());
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
            return ASTNode(ASTNode::Type::NExprVar, identStrV);
        }
    }else if(tok.matchToken(Token::Type::L_PAREN)){
        auto expr = parseExpr(0);
        tok.assertToken(Token::Type::R_PAREN);
        return expr;
    }

    throw UnexpectedTokenException(tok);
}

ASTNode Parser::parseSubscript(ASTNode&& lhs){
    auto expr = parseExpr();
    std::optional<ASTNode> numOpt; // so this can be declared without being initialized
    if(tok.matchToken(Token::Type::AT)){
        // has to be followed by number
        tok.assertToken(Token::Type::NUM, false);

        // parse sizespec as number, validate it's 1/2/4/8
        numOpt = parsePrimaryExpression();
        auto& num = numOpt.value();
        if(
            !(
                num.value==1 ||
                num.value==2 ||
                num.value==4 ||
                num.value==8
             )
        ){
            throw ParsingException("Line " +std::to_string(tok.getLineNum())+": Expression containing " + expr.toString() + " was followed by @, but @ wasn't followed by 1/2/4/8");
        }
    }else{
        numOpt = ASTNode(ASTNode::Type::NExprNum);
        auto& num = numOpt.value();
        num.value = 8; //8 by default
    }

    tok.assertToken(Token::Type::R_BRACKET);

    auto subscript = ASTNode(ASTNode::Type::NExprSubscript, {
    std::move(lhs),
    std::move(expr),
    std::move(numOpt.value())});
    return subscript;
}

//adapted from the lecture slides
ASTNode Parser::parseExpr(int minPrec){
    ASTNode lhs = parsePrimaryExpression();
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
            // i hope this var def gets 'inlined'
            auto subscript = parseSubscript(std::move(lhs));
            lhs = std::move(subscript);
            continue; // continue at the next level up
        }
        tok.nextToken(); // advance the tokenizer, now that we have actually consumed the token

        int newPrec = rassoc ? prec : prec+1;
        ASTNode rhs = parseExpr(newPrec);
        auto newLhs = ASTNode(ASTNode::Type::NExprBinOp, token.type);
        newLhs.children.push_back(std::move(lhs));
        newLhs.children.push_back(std::move(rhs));
        lhs = std::move(newLhs);
    }
}

ASTNode Parser::parseFunction(){
    if(tok.matchToken(Token::Type::IDENTIFIER)){
        auto name = tok.matched.value;
        if(tok.matchToken(Token::Type::L_PAREN)){
            auto paramlist = ASTNode(ASTNode::Type::NParamList);
            try{
                while(true){
                    if(tok.matchToken(Token::Type::R_PAREN)){
                        break;
                    }else if(tok.matchToken(Token::Type::COMMA) && tok.matchToken(Token::Type::IDENTIFIER, /*advance*/ false)){
                        // a comma needs to be followed by an identifier, so this needs to be checked here, but let the next loop iteration actually handle the identifier
                        continue;
                    }else if(tok.matchToken(Token::Type::IDENTIFIER)){
                        auto paramname = tok.matched.value;
                        //identifiers need to be seperated by commas, not follow each other directly
                        if(tok.matchToken(Token::Type::IDENTIFIER, false)){
                            throw UnexpectedTokenException(tok, Token::Type::R_PAREN);
                        }
                        paramlist.children.emplace_back(ASTNode::Type::NExprVar, paramname, IdentifierInfo::REGISTER /* params are always registers */);
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
                if(tok.matched.type == Token::Type::EOP)
                    throw e;
                
            }
            // at this point either a consumed R_PAREN or syntax error is guaranteed
            tok.assertToken(Token::Type::L_BRACE);
            auto body = parseBlock();
            auto func = ASTNode(ASTNode::Type::NFunction, name);
            func.children.push_back(std::move(paramlist));
            func.children.push_back(std::move(body));
            return func;
        }
    }
    throw UnexpectedTokenException(tok);
}

unique_ptr<AST> Parser::parse(){
    auto ast = std::make_unique<AST>();
    auto& root = ast->root;

    // parse a bunch of functions
    while(!tok.matchToken(Token::Type::EOP, false)){
        root.children.emplace_back(parseFunction());
    }
    return ast;
}

// TODO maybe integrate this into the actual parsing at some point
namespace SemanticAnalysis{
    bool failed{false};
    llvm::StringMap<int> externalFunctionsToNumParams{};

    llvm::StringSet<> declaredFunctions{};

    // ---- Scopes ----

    class Scopes{
        struct DeclarationEntry{
            IdentifierInfo* info;
            uint16_t nestingLevel;
            uint32_t declarationBlockNumberAtNestingLevel;
        };

        uint16_t currentNestedLevel{0};
        /// describes how many blocks we have already visited in the current nesting level
        /// -> together with nesting level, uniquely identifies a block
        /// -> if at the declaration nesting level, this is the same as the current nesting level, then the declaration is a parent -> can use decl
        llvm::SmallVector<uint32_t> passedBlocksPerNestingLevel{0};
        llvm::StringMap<llvm::SmallVector<DeclarationEntry>> definitions{};

    public:
        void nest(){
            currentNestedLevel++;
            // if we visit the same nesting level repeatedly, we don't want to push an entry every time
            if(currentNestedLevel >= passedBlocksPerNestingLevel.size())
                passedBlocksPerNestingLevel.emplace_back(0);
        }

        void unnest(){
            assert(currentNestedLevel > 0 && "unnesting too much");

            // now that we unnest, we have passed a block in the current nesting level
            passedBlocksPerNestingLevel[currentNestedLevel]++;
            currentNestedLevel--;
        }

        void declareVariable(IdentifierInfo* info){
            static uint64_t nextVarUID = 1; // start at 1 so that 0 is invalid/not set yet

            assert((ArgParse::args.iterations() || info->uID == 0) && "Variable which should be declared in the current scope already has a UID");

            // we have to create a new UID for this variable
            info->uID = nextVarUID++;

            auto* decls = find(info->name);
            if(decls){
                // forbid same scope shadowing
                if(!decls->empty() && decls->back().nestingLevel == currentNestedLevel){
                    SEMANTIC_ERROR("Variable '" << info->name << "' already declared in this scope");
                    return;
                }
                decls->emplace_back(info, currentNestedLevel, passedBlocksPerNestingLevel[currentNestedLevel]);
            }else{
                bool worked = definitions.try_emplace(info->name, llvm::SmallVector<DeclarationEntry>(1, {info, currentNestedLevel, passedBlocksPerNestingLevel[currentNestedLevel]})).second;
                (void) worked;
                assert(worked && "should always succeed in inserting new decl list in this case");
            }
        }

        /// returns a pointer to the identifier info of the declaration, or nullptr if it is not declared in the current scope
        IdentifierInfo* operator[](const string_view name){
            if(auto* decls = find(name); decls && !decls->empty()){
                return decls->back().info;
            }

            SEMANTIC_ERROR("Variable '" << name << "' used but not declared");
            return nullptr;
        }

        void reset(){
            currentNestedLevel = 0;
            passedBlocksPerNestingLevel.clear();
            definitions.clear();

            passedBlocksPerNestingLevel.emplace_back(0);
        }

    private:
        /// returns a pointer to the vector of declaration entries which's back is the correct declaration for the name at the current scope
        /// returns nullptr if no declaration vector was found
        inline llvm::SmallVector<DeclarationEntry>* find(const string_view name){
            auto it = definitions.find(name);
            if(it == definitions.end()) return nullptr;

            auto& decls = it->second;
            while(decls.size() > 0){
                auto& [info, declaredNest, declaredBlockNum] = decls.back();
                if(declaredNest <= currentNestedLevel && declaredBlockNum == passedBlocksPerNestingLevel[declaredNest]){
                    /// we have to be sure that the declaration happend "above" the current block
                    /// and that it was declared in the same "parent/grandparent/great-grandparent/..." scope

                    // we found a declaration in the current scope, it's at the back of the vector we return
                    return &decls;
                }
                // we found a declaration, but it was either not in a scope above, or in a scope that we have already passed, so it's unusable now, get rid of it
                decls.pop_back();
            }

            // in this case, the list already exists, but we want to add to the now empty list
            return &decls;
        }

    } scopes;

    void analyzeNode(ASTNode& node) noexcept {
        //checks that declarations happen before use
        if(node.type == ASTNode::Type::NFunction){
            // add to declared, remove from external
            declaredFunctions.insert(node.ident.name);
            externalFunctionsToNumParams.erase(node.ident.name);

            // add params to decls
            for(auto& param : node.children[0].children){
                scopes.declareVariable(&param.ident);
            }
            analyzeNode(node.children[1]);

            scopes.reset();
            
            // don't analyze them again
            return;
        }else if(node.type == ASTNode::Type::NStmtBlock){
            // add local vars to decls
            scopes.nest();
            for(auto& stmt : node.children){
                if(stmt.type == ASTNode::Type::NStmtDecl){
                    // right side needs to be evaluated first, then left side can be annotated
                    analyzeNode(stmt.children[0]);

                    scopes.declareVariable(&stmt.ident);
                }else{
                    try{
                        analyzeNode(stmt);
                    }catch(std::runtime_error& e){
                        SEMANTIC_ERROR(e.what());
                    }
                }
            }
            scopes.unnest();

            // don't analyze them again
            return;
        }else if(node.type == ASTNode::Type::NExprCall){
            if(!declaredFunctions.contains(node.ident.name)){
                if(auto pairIt = externalFunctionsToNumParams.find(node.ident.name); pairIt != externalFunctionsToNumParams.end()){
                    if(pairIt->second !=  static_cast<int>(node.children.size())){
                        // we seem to have ourselves a vararg function we don't know anything about, so indicate that by setting the number of params to -1
                        pairIt->second = EXTERNAL_FUNCTION_VARARGS;
                    }
                }else{
                    externalFunctionsToNumParams[node.ident.name] = node.children.size();
                }
            }
        }else if(node.type == ASTNode::Type::NExprVar){
            auto* declInfo  = scopes[node.ident.name];
            if(!declInfo)
                return;

            // we could also try to pointer things up a bit and use the same identifier info for all nodes, but we would then just have a data union of an actual ident info and an ident info pointer, which doesn't save space. It only complicates the maintenance of things, and as this is only assigned once, there are also no redunandancy issues afterwards, so just duplicate it for simplicity.
            node.ident.uID  = declInfo->uID;
            node.ident.type = declInfo->type;

            return;
        }else if((node.type == ASTNode::Type::NExprBinOp && node.op == Token::Type::ASSIGN) || (node.type == ASTNode::Type::NExprUnOp && node.op == Token::Type::AMPERSAND)){
            if(node.type == ASTNode::Type::NExprUnOp && node.children[0].type == ASTNode::Type::NExprVar){
				assert(node.op == Token::Type::AMPERSAND);

                // register variables and parameters are not permitted as operands to the unary addrof & operator
                // subscript is fine and thus left out here
                if(auto declIdentInfo = scopes[node.children[0].ident.name]; !declIdentInfo || declIdentInfo->type!=IdentifierInfo::AUTO){
                    SEMANTIC_ERROR("Cannot take the address of a parameter or register variable");
                }
            }
            // lhs/only child must be subscript or identifier
            if(node.children[0].type != ASTNode::Type::NExprSubscript && node.children[0].type != ASTNode::Type::NExprVar){
                SEMANTIC_ERROR("LHS of assignment/addrof must be a variable or subscript array access, got node which prints as: " << std::endl << node);
            }
        }else if(node.type == ASTNode::Type::NStmtIf || node.type == ASTNode::Type::NStmtWhile){
            // forbid declarations as sole stmt of while/if/else
            if(node.children[1].type == ASTNode::Type::NStmtDecl || (node.children.size() > 2 && node.children[2].type == ASTNode::Type::NStmtDecl)){
                SEMANTIC_ERROR("Declarations are not allowed as the sole statement in while/if/else");
            }
        }else{
        }
        for(auto& child : node.children){
            analyzeNode(child);
        }
    }

    void analyze(AST& ast){
        analyzeNode(ast.root);
    }

    void reset(){
        failed = false;
        declaredFunctions.clear();
        externalFunctionsToNumParams.clear();
        scopes.reset();
    }
} // end namespace SemanticAnalysis
