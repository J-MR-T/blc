#pragma GCC diagnostic push 
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wcomment"
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/PostOrderIterator.h>
#pragma GCC diagnostic pop

#include "util.h"

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
    string_view value{""}; // only used for identifiers and numbers

    // turns the tokens back into their original strings for pretty printing
    static string toString(Token::Type type);

    string toString();

    bool operator==(const Token &other) const;

    // includes implicit conversion from type to Token
    Token(Type type, string_view value = "") : type(type), value(value) {}

};

// TODO yeet all those stupid exceptions and compile with -fno-exceptions
class ParsingException : public std::runtime_error {
    public:
      ParsingException(string msg);
};

class Tokenizer{
public:
    Token matched;

private:
    const string prog;
    std::size_t progI{0};
    Token peeked;

public:
    class UnexpectedTokenException :  public ParsingException {
        static std::string exceptionString(Tokenizer &tokenizer, Token::Type expected);

    public:
        UnexpectedTokenException(Tokenizer &tokenizer, Token::Type expected = Token::Type::EMPTY);
    };

    static string initProg(std::ifstream &inputFile);

    Tokenizer(string &&prog);

    Tokenizer(std::ifstream &inputFile);

    static const std::unordered_map<string, Token::Type> keywords;
    static const std::regex numberRegex;
    static const std::regex identifierRegex;

    static const string newline;

    Token peekToken();

    std::uint64_t getLineNum();

    Token nextToken();

    bool matchToken(Token::Type type, bool advance = true);

    void assertToken(Token::Type type, bool advance = true);

    void assertNotToken(Token::Type type, bool advance = true);

    // only used for performance tests with multiple iterations
    void reset();
};


struct IdentifierInfo{
    string_view name;
    uint64_t uID;
    enum Type {
        UNKNOWN,
        REGISTER,
        AUTO,
        FUNCTION,
    } type;

    static IdentifierInfo::Type fromTokenType(const Token::Type t);

    static string toString(Type t);
};

class ASTNode{
public:
    enum class Type{
        NRoot,          // possible children: [function*]
        NFunction,      // possible children: [paramlist, block], identifier: yes
        NParamList,     // possible children: [var*]
        NStmtDecl,      // possible children: [expr (*required* initializer)], identifier: yes
        NStmtReturn,    // possible children: [expr?]
        NStmtBlock,     // possible children: [statement*]
        NStmtWhile,     // possible children: [expr, stmt]
        NStmtIf,        // possible children: [expr, stmt, stmt (optional else)]
        NExprVar,       // possible children: [], identifier: yes
        NExprNum,       // possible children: [], value: yes
        NExprCall,      // possible children: [expr*], identifier: yes
        NExprUnOp,      // possible children: [expr], op: yes (MINUS/TILDE/AMPERSAND/LOGICAL_NOT)
        NExprBinOp,     // possible children: [expr, expr], op: yes (all the binary operators possible)
        NExprSubscript, // possible children: [expr(addr), expr(index), num (sizespec, 1/2/4/8)]
    };

    class Hash{
    public:
        size_t operator()(const ASTNode &node) const;
    };

    Type type;

    static int nodeCounter; // initialized to 0
    uint64_t nodeID = nodeCounter++;

    /// maps the type of a node to the number of children it should have, used for validation. -n with n>1 means at least (n-1) children -> for example -1 means 0 or more children (any number)
    static const InsertOnceQueryAfterwardsMap<ASTNode::Type, int> numberOfChildren;
    static const InsertOnceQueryAfterwardsMap<ASTNode::Type, std::string> nodeTypeToDotIdentifier;
    static const InsertOnceQueryAfterwardsMap<ASTNode::Type, llvm::SmallVector<std::string,4>> nodeTypeToDotStyling;

    /// node attributes
    union{
        int64_t value;
        Token::Type op; // for UnOp, BinOp, and distinguishing auto vars from register vars
        IdentifierInfo ident;
    };

    std::vector<ASTNode> children{};

    ASTNode(Type type, string_view name = "", IdentifierInfo::Type t = IdentifierInfo::UNKNOWN);

    ASTNode(Type type, string_view name, Token::Type t);

    ASTNode(Type type, int64_t value);

    ASTNode(Type type, Token::Type op);

    ASTNode(Type type, llvm::ArrayRef<ASTNode> children);

    string uniqueDotIdentifier() const;

    string toString() const;

    void printDOT(std::ostream& out, int indentDepth = 0, bool descend = true) const;

    inline void iterateChildren(std::function<void(ASTNode&)> f){
        for(auto& child : children){
            f(child);
            child.iterateChildren(f);
        }
    }

};

std::ostream& operator<<(std::ostream& out, ASTNode& node);

class AST{
public:
    ASTNode root{ASTNode::Type::NRoot};

    void printDOT(std::ostream& out);

    size_t getRoughMemoryFootprint();

    bool validate();
};

static const std::unordered_map<Token::Type, std::tuple<int /* precedence */, bool /* is right-assoc */>> operators = {
    {Token::Type::L_BRACKET,           {14, false}},
    // unary: 13 (handled seperately)
    {Token::Type::TIMES,               {12, false}},
    {Token::Type::DIV,                 {12, false}},
    {Token::Type::MOD,                 {12, false}},
    {Token::Type::PLUS,                {11, false}},
    {Token::Type::MINUS,               {11, false}},
    {Token::Type::SHIFTL,              {10, false}},
    {Token::Type::SHIFTR,              {10, false}},
    {Token::Type::LESS,                {9,  false}},
    {Token::Type::GREATER,             {9,  false}},
    {Token::Type::LESS_EQUAL,          {9,  false}},
    {Token::Type::GREATER_EQUAL,       {9,  false}},
    {Token::Type::EQUAL,               {8,  false}},
    {Token::Type::NOT_EQUAL,           {8,  false}},
    {Token::Type::AMPERSAND,           {7,  false}}, //bitwise AND in this case
    {Token::Type::BITWISE_XOR,         {6,  false}},
    {Token::Type::BITWISE_OR,          {5,  false}},
    {Token::Type::LOGICAL_AND,         {4,  false}},
    {Token::Type::LOGICAL_OR,          {3,  false}},
    {Token::Type::ASSIGN,              {1,  true}},
};

using UnexpectedTokenException = Tokenizer::UnexpectedTokenException;

class Parser{

    Tokenizer tok;

public:
    bool failed{false};

    Parser(string&& prog);

    Parser(std::ifstream& inputFile);

    // only called in case there is a return at the start of a statement -> throws exception if it fails
    ASTNode parseStmtDecl();

    ASTNode parseStmtReturn();

    ASTNode parseBlock();

    ASTNode parseStmtIfWhile(bool isWhile);

    ASTNode parseStmt();

    //avoid left recursion
    ASTNode parsePrimaryExpression();

    ASTNode parseSubscript(ASTNode&& lhs);

    ASTNode parseExpr(int minPrec = 0);

    ASTNode parseFunction();

    unique_ptr<AST> parse();

    void resetTokenizer(){
        tok.reset();
    }
};

namespace SemanticAnalysis{
    extern bool failed;
    extern llvm::StringMap<int> externalFunctionsToNumParams;
#define EXTERNAL_FUNCTION_VARARGS -1

    extern llvm::StringSet<> declaredFunctions;

#define SEMANTIC_ERROR(msg) do{                                   \
    std::cerr << "Semantic Analysis error: " << msg << std::endl; \
    failed = true;                                                \
    } while(0)


    void analyzeNode(ASTNode& node) noexcept ;
    
    void analyze(AST& ast);

    void reset();

} // end namespace SemanticAnalysis
