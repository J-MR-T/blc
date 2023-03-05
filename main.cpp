#include <iostream>
#include <cassert>
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
#include <list>
#include <utility>
#include <filesystem>

#include <err.h>
#include <sys/wait.h>

#pragma GCC diagnostic push 
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <llvm/Config/llvm-config.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/ValueMap.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <llvm/ADT/SmallSet.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/PostOrderIterator.h>
#include <llvm/IR/Dominators.h>
#include <llvm/Analysis/CFG.h>

// stuff I assume i need for writing object files...
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Support/Host.h>
#include <llvm/ADT/Triple.h>
#include <llvm/Object/Error.h>
#include <llvm/Object/MachO.h>
#include <llvm/Object/ObjectFile.h>
#include <llvm/Object/SymbolicFile.h>
#include <llvm/MC/MCObjectWriter.h>
#include <llvm/MC/MCAsmBackend.h>
#include <llvm/MC/MCAsmInfo.h>
#include <llvm/MC/MCCodeEmitter.h>
#include <llvm/MC/MCContext.h>
#include <llvm/MC/MCInstrInfo.h>
#include <llvm/MC/MCObjectWriter.h>
#include <llvm/MC/MCRegisterInfo.h>
#include <llvm/MC/MCSubtargetInfo.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/MC/MCTargetOptions.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/MC/MCTargetOptionsCommandFlags.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/ToolOutputFile.h>
#pragma GCC diagnostic pop

using std::string;
using std::string_view;
using std::unique_ptr;
using std::literals::string_literals::operator""s;

#ifndef NDEBUG
#define DEBUGLOG(x) llvm::errs() << x << "\n"; fflush(stderr);
#else
#define DEBUGLOG(x)
#endif

#define STRINGIZE(x) #x
#define STRINGIZE_MACRO(x) STRINGIZE(x)

// kind of an enum class but implicitly convertible to int
namespace ExitCode{
enum {
    SUCCESS       = 0,
    ERROR         = 1,
    TODO          = 2, // exit status 2 for 2do :)
    ERROR_IO      = ERROR | 1 << 2,
    ERROR_SYNTAX  = ERROR | 1 << 3,
    ERROR_CODEGEN = ERROR | 1 << 4,
    ERROR_LINK    = ERROR | 1 << 5,
};
}

#define EXIT_TODO_X(x) \
    errx(ExitCode::TODO, "TODO(Line " STRINGIZE_MACRO(__LINE__) "): " x "\n");

#define EXIT_TODO EXIT_TODO_X("Not implemented yet.")

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

        // returns whether the arg has a value/has been set
        bool operator()() const;

        // returns the args value
        std::string& operator*() const;
    };

    std::map<Arg, std::string> parsedArgs{};
    
    // struct for all possible arguments
    const struct {
        const Arg help{      "h", "help"      , 0, "Show this help message and exit"                                                                     , false, true};
        const Arg input{     "i", "input"     , 1, "Input file"                                                                                          ,  true, false};
        const Arg dot{       "d", "dot"       , 0, "Output AST in GraphViz DOT format (to stdout by default, or file using -o) (overrides -p)"           , false, true};
        const Arg output{    "o", "output"    , 2, "Output file for AST (requires -p)"                                                                   , false, false};
        const Arg preprocess{"E", "preprocess", 0, "Run the C preprocessor on the input file before parsing it"                                          , false, true};
        const Arg url{       "u", "url"       , 0, "Instead of printing the AST in DOT format to the console, print a URL to visualize it in the browser", false, true};
        const Arg benchmark{ "b", "benchmark" , 0, "Measure execution time and print memory footprint"                                                   , false, true};
        const Arg iterations{"" , "iterations", 0, "Number of iterations to run the benchmark for (default 1, requires -b)"                              , false, false};
        const Arg llvm{      "l", "llvm"      , 0, "Print LLVM IR if used without -o. Compiles to object file and links to executable if used with -o.\n"
                                                   "Disables the rest of the compilation process"                                                        , false, true};
        const Arg nowarn{    "w", "nowarn"    , 0, "Do not generate warnings during the LLVM codegeneration phase"                                       , false, true};
        const Arg isel{      "s", "isel"      , 0, "Output (ARM-) instruction selected LLVM-IR"                                                          , false, true};
        const Arg regalloc{  "r", "regalloc"  , 0, "Output (ARM-) register allocated LLVM-IR"                                                            , false, true};
        const Arg asmout{    "a", "asm"       , 0, "Output (ARM-) assembly"                                                                              , false, true};


        const Arg sentinel{"", "", 0, "", false, false};

        const Arg* const all[13] = {&help, &input, &dot, &output, &preprocess, &url, &benchmark, &iterations, &llvm, &nowarn, &isel, &regalloc, &sentinel};
        
        // iterator over all
        const Arg* begin() const{
            return all[0];
        }

        const Arg* end() const{
            return all[12];
        }
    } args;

    bool Arg::operator()() const{
        return parsedArgs.contains(*this);
    }

    std::string& Arg::operator*() const{
        return parsedArgs[*this];
    }

    void printHelp(const char* argv0){
        std::cerr << "A Compiler for a B like language" << std::endl;
        std::cerr << "Usage: " << std::endl;
        for(auto& arg:args){
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
            std::cerr << "    " << arg.description << std::endl; // TODO string replace all \n with \n \t here
        }

        std::cerr << 
        "\nExamples: \n"
        << "  " << argv0 << " -i input.b -p -d -o output.dot\n"
        << "  " << argv0 << " input.b -pd output.dot\n"
        << "  " << argv0 << " input.b -pdu\n"
        << "  " << argv0 << " -lE input.b\n"
        << "  " << argv0 << " -l main.b main\n"
        << "  " << argv0 << " -ls input.b\n"
        << "  " << argv0 << " -sr input.b\n"
        << "  " << argv0 << " -a samples/addressCalculations.b | aarch64-linux-gnu-gcc -g -x assembler -o test - && qemu-aarch64 -L /usr/aarch64-linux-gnu test hi\\ there\n";
    }

    //unordered_map doesnt work because of hash reasons (i think), so just define <, use ordered
    std::map<Arg, std::string>& parse(int argc, char *argv[]){
        std::stringstream ss;
        ss << " ";
        for (int i = 1; i < argc; ++i) {
            ss << argv[i] << " ";
        }

        string argString = ss.str();


        //handle positional args first, they have lower precedence
        //find them all, put them into a vector, then match them to the possible args
        std::vector<string> positionalArgs{};
        for(int i = 1; i < argc; ++i){
            for(const auto& arg : args){
                if(!arg.flag && (("-"+arg.shortOpt) == string{argv[i-1]} || ("--"+arg.longOpt) == string{argv[i-1]})){
                    //the current arg is the value to another argument, so we dont count it
                    goto cont;
                }
            }


            if(argv[i][0] != '-'){
                // now we know its a positional arg
                positionalArgs.emplace_back(argv[i]);
            }
cont:
            continue;
        }

        for(const auto& arg : args){
            if(arg.pos != 0){
                //this is a positional arg
                if(positionalArgs.size() > arg.pos-1){
                    parsedArgs[arg] = positionalArgs[arg.pos-1];
                }
            }
        }

        bool missingRequired = false;

        //long/short/flags
        for(const auto& arg : args){
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
            printHelp(argv[0]);
            exit(ExitCode::ERROR);
        }
        return parsedArgs;
    }

} // end namespace ArgParse

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
    static string toString(Token::Type type){
        switch(type){
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

    // includes implicit conversion from type to Token
    Token(Type type, string_view value = "") : type(type), value(value) {}

};

// TODO yeet all those stupid exceptions and compile with -fno-exceptions
class ParsingException : public std::runtime_error {
    public:
        ParsingException(string msg) : std::runtime_error(msg){}
};

const Token emptyToken{
    Token::Type::EMPTY,
    ""
};

class Tokenizer{
public:

class UnexpectedTokenException :  public ParsingException {

    static std::string exceptionString(Tokenizer& tokenizer, Token::Type expected){
        tokenizer.peekToken();
        std::string typeHint = "";
        if((tokenizer.peeked).type == Token::Type::NUM || (tokenizer.peeked).type == Token::Type::IDENTIFIER){
            typeHint = " (type: " + Token::toString((tokenizer.peeked).type) + ")";
        }
        std::string expectedHint = "";
        if(expected != Token::Type::EMPTY){
            expectedHint = ", expected: " + Token::toString(expected);
        }

        return "Line " + std::to_string(tokenizer.getLineNum()) + ": "
            + "Unexpected token: " + (tokenizer.peeked).toString() 
            + typeHint
            + expectedHint;
    }

    public:
        UnexpectedTokenException(Tokenizer& tokenizer, Token::Type expected = Token::Type::EMPTY) 
            : ParsingException(exceptionString(tokenizer, expected)) {}
};

    static string initProg(std::ifstream& inputFile){
        return string(std::istreambuf_iterator(inputFile), {})+" "; // to safely peek for two char ops at the end
    }

    Tokenizer(string&& prog) : matched(emptyToken), prog(std::move(prog)), peeked(emptyToken) {}

    Tokenizer(std::ifstream& inputFile) : matched(emptyToken), prog(initProg(inputFile)), peeked(emptyToken) {
    }

    static const std::unordered_map<string, Token::Type> keywords; // initialized below
    const std::regex numberRegex{"[0-9]+"};
    const std::regex identifierRegex{"[a-zA-Z_][a-zA-Z0-9_]*"};

    const string newline{"\n"};

    Token matched;

    Token peekToken(){
        using Type = Token::Type;

        if(peeked == emptyToken){
            if(progI == string::npos || progI>=prog.size()){
                //return EOP token
                return peeked = Type::EOP;
            }

            // skip whitespace & comments 
            while(true){
                progI=prog.find_first_not_of(" \f\n\r\t\v", progI); // same chars as isspace uses
                if(progI == string::npos || progI>=prog.size()){
                    return peeked = Type::EOP;
                }


                if(prog[progI] == '/'){
                    if(progI+1 < prog.size() && prog[progI+1] == '/'){
                        // single line comment
                        progI+=2;

                        progI=prog.find(newline, progI);
                        progI+=newline.size();
                    }else if(progI+1 >= prog.size()){
                        return peeked = Type::EOP;
                    }else{
                        break;
                    }
                }else{
                    break;
                }
            }

            if(prog.size()-progI <= 0){
                //return EOP token
                return peeked = Type::EOP;
            }


            //NUM
            if(isdigit(prog[progI])){
                std::smatch match;
                if(std::regex_search(prog.cbegin()+progI, prog.cend(), match, numberRegex)){
                    auto numStrMatch = match[0];
                    auto len = numStrMatch.length();
                    progI += len;
                    return peeked = {Type::NUM, string_view(numStrMatch.first.base(), len)};
                }
            }

            //IDENTIFIER
            if(isalpha(prog[progI]) || prog[progI] == '_'){
                std::smatch match;
                if(std::regex_search(prog.cbegin()+progI, prog.cend(), match, identifierRegex)){
                    auto idStrMatch = match[0];
                    auto len = idStrMatch.length();
                    progI += len;
                    //check if it's a keyword
                    if(auto keywordTokenIt = keywords.find(idStrMatch.str()); keywordTokenIt != keywords.end()){
                        return peeked = keywordTokenIt->second;
                    }else{
                        return peeked = {Type::IDENTIFIER, string_view(idStrMatch.first.base(), len)};
                    }
                }
            }

            Type type = Token::Type::EMPTY;

            switch(prog[progI]){
            //single characters
            //parentheses, brackets, braces, unabiguous operators, ...
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

            //ambiguous one/two character operators
                case '<': if(prog[progI+1] == '=')      type = Type::LESS_EQUAL, progI++; 
                          else if(prog[progI+1] == '<') type = Type::SHIFTL, progI++;
                          else type = Type::LESS;
                          break;
                case '>': if(prog[progI+1] == '=')      type = Type::GREATER_EQUAL, progI++; 
                          else if(prog[progI+1] == '>') type = Type::SHIFTR, progI++;
                          else type = Type::GREATER;
                          break;
                case '=': if(prog[progI+1] == '=')      type = Type::EQUAL, progI++;
                          else type = Type::ASSIGN;
                          break;
                case '&': if(prog[progI+1] == '&')      type = Type::LOGICAL_AND, progI++;
                          else type = Type::AMPERSAND;
                          break;
                case '|': if(prog[progI+1] == '|')      type = Type::LOGICAL_OR, progI++;
                          else type = Type::BITWISE_OR;
                          break;
                case '!': if(prog[progI+1] == '=')      type = Type::NOT_EQUAL, progI++;
                          else type = Type::LOGICAL_NOT;
                          break;
            }

            if(type!=Type::EMPTY){
                progI++;
                return peeked = type;
            }

            //invalid character
            throw std::runtime_error("Invalid character: "s + prog[progI+0]);

        }else{
            return peeked;
        }
    }

    std::uint64_t getLineNum(){
        if(progI == string::npos){
            return std::count(prog.begin(), prog.end(), '\n') +1;
        }
        // this should work on windows too, because '\r\n' also contains '\n', but honestly if windows users have wrong line numbers in their errors, so be it :P
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

struct IdentifierInfo{
    string_view name;
    uint64_t uID;
    enum Type {
        UNKNOWN,
        REGISTER,
        AUTO,
        FUNCTION,
    } type;

    static IdentifierInfo::Type fromTokenType(const Token::Type t){
        switch(t){
            case Token::Type::KW_AUTO:     return AUTO;
            case Token::Type::KW_REGISTER: return REGISTER;
            default:                       return UNKNOWN;
        }
    }

    static string toString(Type t){
        switch(t){
            case REGISTER:  return "register";
            case AUTO:      return "auto";
            case FUNCTION:  return "function";
            case UNKNOWN:
            default:        return "unknown";
        }
    }

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
        NExprCall,      // possible children: [expr*], name: yes
        NExprUnOp,      // possible children: [expr], op: yes (MINUS/TILDE/AMPERSAND/LOGICAL_NOT)
        NExprBinOp,     // possible children: [expr, expr], op: yes (all the binary operators possible)
        NExprSubscript, // possible children: [expr(addr), expr(index), num (sizespec, 1/2/4/8)]
    };

    /// maps the type of a node to the number of children it should have, used for error checking. -1 means arbitrary number of children
    static const std::unordered_map<Type, int> numberOfChildren;
    class Hash{
    public:
        size_t operator()(const ASTNode& node) const{
            return node.nodeID;
        }
    };

    Type type;

    static int nodeCounter; // initialized to 0 below
    uint64_t nodeID = nodeCounter++;

    static const std::unordered_map<Type, string> nodeTypeToDotIdentifier; // initialization below
    static const std::unordered_map<Type, std::vector<string>> nodeTypeToDotStyling; // initialization below

    /// node attributes
    union{
        int64_t value;
        Token::Type op; // for UnOp, BinOp, and distinguishing auto vars from register vars
        IdentifierInfo ident;
    };

    std::vector<ASTNode> children{};

    ASTNode(Type type, string_view name = "", IdentifierInfo::Type t = IdentifierInfo::UNKNOWN) : type(type), ident({name, 0, t}) {}

    ASTNode(Type type, string_view name, Token::Type t) : ASTNode(type, name, IdentifierInfo::fromTokenType(t)) {}

    ASTNode(Type type, int64_t value) : type(type), value(value) {}

    ASTNode(Type type, Token::Type op) : type(type), op(op) {}

    ASTNode(Type type, std::initializer_list<ASTNode> children) : type(type), children(children) {}

    string uniqueDotIdentifier() const {
        return nodeTypeToDotIdentifier.at(type) + "_" + std::to_string(nodeID);
    }

    string toString() const {
        if(type == Type::NExprNum){
            return std::to_string(value);
        } else if (type == Type::NExprBinOp || type == Type::NExprUnOp){
            return Token::toString(op);
        } else if (type == Type::NStmtDecl || type == Type::NExprVar){
            return nodeTypeToDotIdentifier.at(type) + "(" + IdentifierInfo::toString(ident.type) + ")" + ": " + string(ident.name);
        } else {
            return nodeTypeToDotIdentifier.at(type);
        }
    }

    void printDOT(std::ostream& out, int indentDepth = 0, bool descend = true) const {
        string indent(4*(indentDepth+1), ' ');

        out << indent << uniqueDotIdentifier() << " [label=\"" << toString() << "\"" ;
        if(nodeTypeToDotStyling.contains(type)){
            for(auto& styleInstr : nodeTypeToDotStyling.at(type)){
                out << ", " << styleInstr;
            }
        }
        out << "];" << std::endl;

        if(descend){
            for(auto& child : children){
                out << indent << uniqueDotIdentifier() << " -> " << child.uniqueDotIdentifier() << ";" << std::endl;
                child.printDOT(out, indentDepth+1);
            }
        }
    }

    inline void iterateChildren(std::function<void(ASTNode&)> f){
        for(auto& child : children){
            f(child);
            child.iterateChildren(f);
        }
    }

};

int ASTNode::nodeCounter = 0;

const std::unordered_map<ASTNode::Type, int> ASTNode::numberOfChildren = {
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
};

std::ostream& operator<< (std::ostream& out, ASTNode& node){
    node.printDOT(out);
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
            out << root.uniqueDotIdentifier() << " -> " << child.uniqueDotIdentifier() << ";" << std::endl;
            out << "subgraph cluster_" << child.uniqueDotIdentifier() << " {" << std::endl;
            //function cluster styling

            out << "style=filled;" << std::endl;
            out << "color=lightgrey;" << std::endl;
            out << "node [style=filled,color=white];" << std::endl;
            out << "label = \"" << child.toString() << "\";" << std::endl;
            child.printDOT(out, 1);
            out << "}" << std::endl;
        }

        out << "}" << std::endl;
    }

    size_t getRoughMemoryFootprint(){
        size_t totalSize = 0;
        root.iterateChildren([&totalSize](ASTNode& node){
            totalSize += sizeof(node);
            //seperately add the size of the children vector
            totalSize += node.children.capacity() * sizeof(ASTNode);
        });
        return totalSize;
    }

    bool validate(){
        bool valid = true;
        root.iterateChildren([&valid](ASTNode& node){
            auto number = ASTNode::numberOfChildren.at(node.type);
            if(number > 0 && node.children.size() != static_cast<size_t>(number)){
                std::cerr << "Node " << node.uniqueDotIdentifier() << " has " << node.children.size() << " children, but should have " << number << "\n";
                valid = false;
            }
        });
        return valid;
    }

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

    Parser(string&& prog) : tok{std::move(prog)} {}

    Parser(std::ifstream& inputFile) : tok{inputFile} {}
    // only called in case there is a return at the start of a statement -> throws exception if it fails
    ASTNode parseStmtDecl(){
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

    ASTNode parseStmtReturn(){
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

    ASTNode parseBlock(){
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

    ASTNode parseStmtIfWhile(bool isWhile){
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

    ASTNode parseStmt(){
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


    //avoid left recursion
    ASTNode parsePrimaryExpression(){
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
            auto ident = tok.matched.value;
            if(tok.matchToken(Token::Type::L_PAREN)){
                auto call = ASTNode(ASTNode::Type::NExprCall, ident);
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
                return ASTNode(ASTNode::Type::NExprVar, ident);
            }
        }else if(tok.matchToken(Token::Type::L_PAREN)){
            auto expr = parseExpr(0);
            tok.assertToken(Token::Type::R_PAREN);
            return expr;
        }

        throw UnexpectedTokenException(tok);
    }

    inline ASTNode parseSubscript(ASTNode&& lhs){
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
    ASTNode parseExpr(int minPrec = 0){
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

    ASTNode parseFunction(){
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
                    if(tok.matched.type == Token::Type::EOP){
                        throw e;
                    }
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

};

namespace SemanticAnalysis{
    bool failed{false};
    llvm::StringMap<int> externalFunctionsToNumParams{};
#define EXTERNAL_FUNCTION_VARARGS -1

    llvm::StringSet<> declaredFunctions{};

#define SEMANTIC_ERROR(msg) do{                                   \
    std::cerr << "Semantic Analysis error: " << msg << std::endl; \
    failed = true;                                                \
    } while(0)

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
                if(decls->back().nestingLevel == currentNestedLevel){
                    SEMANTIC_ERROR("Variable '" << info->name << "' already declared in this scope");
                    return;
                }
                decls->emplace_back(info, currentNestedLevel, passedBlocksPerNestingLevel[currentNestedLevel]);
            }else{
                definitions.try_emplace(info->name, llvm::SmallVector<DeclarationEntry>(1, {info, currentNestedLevel, passedBlocksPerNestingLevel[currentNestedLevel]}));
            }
        }

        IdentifierInfo* operator[](const string_view name){
            if(auto* decls = find(name)){
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
        /// returns nullptr if no declaration was found
        inline llvm::SmallVector<DeclarationEntry>* find(const string_view name){
            auto it = definitions.find(name);
            if(it == definitions.end()) return nullptr;

            auto& decls = it->second;
            while(decls.size() > 0){
                auto& [info, declaredNest, declaredBlockNum] = it->second.back();
                if(declaredNest <= currentNestedLevel && declaredBlockNum == passedBlocksPerNestingLevel[declaredNest]){
                    /// we have to be sure that the declaration happend "above" the current block
                    /// and that it was declared in the same "parent/grandparent/great-grandparent/..." scope

                    // we found a declaration in the current scope, it's at the back of the vector we return
                    return &it->second;
                }
                // we found a declaration, but it was either not in a scope above, or in a scope that we have already passed, so it's unusable now, get rid of it
                decls.pop_back();
            }

            return nullptr;
        }

    } scopes;

    // decls only contains variable (name, isRegister), because ASTNodes have no copy constructor and using a unique_ptr<>& doesn't work for some unknown reason
    // quadratic in the number of variables/scopes (i.e. if both are arbitrarily large)
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
            if(!declInfo){
                return;
            }
            // we could also try to pointer things up a bit and use the same identifier info for all nodes, but we would then just have a data union of an actual ident info and an ident info pointer, which doesn't save space. It only complicates the maintenance of things, and as this is only assigned once, there are also no redunandancy issues afterwards, so just duplicate it for simplicity.
            node.ident.uID  = declInfo->uID;
            node.ident.type = declInfo->type;

            return;
        }else if((node.type == ASTNode::Type::NExprBinOp && node.op == Token::Type::ASSIGN) || (node.type == ASTNode::Type::NExprUnOp && node.op == Token::Type::AMPERSAND)){
            if(node.type == ASTNode::Type::NExprUnOp && node.children[0].type == ASTNode::Type::NExprVar){
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

    void warn(const std::string& msg, llvm::Value* value = nullptr){
        if(!ArgParse::args.nowarn()){
            llvm::errs() << "Warning: " << msg;
            if(value)
                llvm::errs() << " at " << *value;
            llvm::errs() << "\n";
            warningsGenerated = true;
        }
    }

    llvm::StringRef llvmGetStringMetadata(llvm::Instruction* inst, llvm::StringRef mdName){
        assert(inst->hasMetadata(mdName) && ("call must have \"" + mdName + "\" metadata").str().c_str());
        return llvm::dyn_cast<llvm::MDString>(inst->getMetadata(mdName)->getOperand(0))->getString();
    }

    void llvmSetStringMetadata(llvm::Instruction* inst, llvm::StringRef mdName, llvm::StringRef val){
        llvm::MDNode* md = llvm::MDNode::get(ctx, llvm::MDString::get(ctx, val));
        inst->setMetadata(mdName, md);
    }

    void llvmSetEmptyMetadata(llvm::Instruction* inst, llvm::StringRef mdName){
        inst->setMetadata(mdName, llvm::MDNode::get(ctx, {}));
    }

    string llvmPredicateToARM(llvm::ICmpInst::Predicate pred){
        switch(pred){
            // we only use 6 of these:
            case llvm::CmpInst::ICMP_EQ:
                return "eq";
            case llvm::CmpInst::ICMP_NE:
                return "ne";
            case llvm::CmpInst::ICMP_SGT:
                return "gt";
            case llvm::CmpInst::ICMP_SGE:
                return "ge";
            case llvm::CmpInst::ICMP_SLT:
                return "lt";
            case llvm::CmpInst::ICMP_SLE:
                return "le";
            default:
                assert(false && "Invalid predicate");
                return "";
        }
    }

    struct BasicBlockInfo{
        bool sealed{false};
        llvm::DenseMap<uint64_t /* variable uid */, llvm::Value*> regVarmap{};
    };

    llvm::DenseMap<uint64_t /* variable uid */, llvm::AllocaInst*> autoVarmap{};

    llvm::DenseMap<llvm::BasicBlock*, BasicBlockInfo> blockInfo{};
    llvm::DenseMap<llvm::PHINode*, ASTNode*> phisToResolve{};


    // REFACTOR: cache the result in a hash map
    llvm::Function* findFunction(llvm::StringRef name){
        auto fnIt = llvm::find_if(moduleUP->functions(), [&name](auto& func){return func.getName() == name;});
        if(fnIt == moduleUP->functions().end()){
            return nullptr;
        }else{
            return &*fnIt;
        }
    }

    llvm::Value* varmapLookup(llvm::BasicBlock* block, ASTNode& node) noexcept;

    // for different block for lookup/insert
    llvm::Value* wrapVarmapLookupForUse(llvm::BasicBlock* block, llvm::IRBuilder<>& irb, ASTNode& node) noexcept{
        if(node.ident.type == IdentifierInfo::REGISTER){
            // this should be what we want for register vars, for auto vars we aditionally need to look up the alloca (and store it back if its an assignment, see the assignment below)
            return varmapLookup(block, node); // NOTE to self: this does work even though there is a pointer to the value in the varmap, because if the mapping gets updated, that whole pointer isn't the value anymore, nothing is changed about what it's pointing to.
        }else {
            return irb.CreateLoad(i64, varmapLookup(block, node), node.ident.name);
        }
    }

    llvm::Value* wrapVarmapLookupForUse(llvm::IRBuilder<>& irb, ASTNode& node) noexcept{
        return wrapVarmapLookupForUse(irb.GetInsertBlock(), irb, node);
    }

    // automatically creates phi nodes on demand
    llvm::Value* varmapLookup(llvm::BasicBlock* block, ASTNode& node) noexcept {
        uint64_t uID = node.ident.uID;
        if(node.ident.type == IdentifierInfo::AUTO){
            assert(autoVarmap.find(uID) != autoVarmap.end() && "auto var not found in varmap");
            return autoVarmap[uID];
        }

        auto& [sealed, varmap] = blockInfo[block];
        if(auto it = varmap.find(uID); it != varmap.end()){
            return it->getSecond();
        }else{
            if(sealed){
                // cache it, so we don't have to look it up every time
                if(block->hasNPredecessors(1)){
                    return varmap[uID] = varmapLookup(block->getSinglePredecessor(), node);
                }else if(block->hasNPredecessors(0)){
                    // returning poison is quite reasonable, as anything here should never be used, or looked up (either unreachable, or entry block)
                    return varmap[uID] = llvm::PoisonValue::get(i64);
                }else{ // > 1 predecessors
                    // create phi node to merge it
                    llvm::IRBuilder<> irb(block);
                    auto nonphi = block->getFirstNonPHI();
                    // if there is no nonphi node, we can just insert at the end, which should be where the irb starts
                    if(nonphi!=nullptr){
                        irb.SetInsertPoint(nonphi); // insertion is before the instruction, so this is the correct position
                    }
                    llvm::PHINode* phi = irb.CreatePHI(i64, 2, node.ident.name); // num reserved values here is only a hint, it's at least one because of our algo, 2 because we have >1 preds
                    varmap[uID] = phi; // doing this here breaks a potential cycle in the following lookup, if we look up this same name and it points back here, we don't create another phi node, we use the existing one we just created
                    
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
                // if there is no non-phi node, we can just insert at the end of the block, which should be where the irb starts
                if(nonphi!=nullptr){
                    irb.SetInsertPoint(nonphi); // insertion is before the instruction, so this is the correct position
                }
                llvm::PHINode* phi = irb.CreatePHI(i64, 2, node.ident.name); // num reserved values here is only a hint, 0 is fine "[...] if you really have no idea", it's at least one because of our algo
                phisToResolve[phi] = &node;
                
                // incoming values/blocks get added by fillPHIs later
                return varmap[uID] = phi;
            }
        }
    }

    /// just for convenience
    /// can *only* be called with register vars, as auto vars need to be looked up in the alloca map
    inline llvm::Value* updateVarmap(llvm::BasicBlock* block, ASTNode& node, llvm::Value* val) noexcept{
        assert(node.ident.type == IdentifierInfo::REGISTER && "can only update register vars with this method");

        auto& [sealed, varmap] = blockInfo[block];
        return varmap[node.ident.uID] = val;
    }

    inline llvm::Value* updateVarmap(llvm::IRBuilder<>& irb, ASTNode& node, llvm::Value* val) noexcept{
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
            errx(ExitCode::ERROR_CODEGEN, "Something has gone seriously wrong here, got a sizespec of %ld bytes", sizespecInt);
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
                        args[i] = genExpr(exprNode.children[i], irb);
                    }
                    auto callee = findFunction(exprNode.ident.name);
                    assert(callee && "got a call to a function that doesn't exist, but was neither forward declared, nor implicitly declared, this should never happen");
                    if(args.size() != callee->arg_size()){
                        // hw02.txt: "Everything else is handled as in ANSI C", hw04.txt: "note that parameters/arguments do not need to match"
                        // but: from the latest C11 standard draft: "the number of arguments shall agree with the number of parameters"
                        // (i just hope thats the same as in C89/ANSI C, can't find that standard anywhere online, Ritchie & Kernighan says this:
                        // "The effect of the call is undefined if the number of arguments disagrees with the number of parameters in the
                        // definition of the function", which is basically the same)
                        // so technically this is undefined behavior >:)
                        if(auto it = SemanticAnalysis::externalFunctionsToNumParams.find(exprNode.ident.name); 
                                it != SemanticAnalysis::externalFunctionsToNumParams.end() &&
                                SemanticAnalysis::externalFunctionsToNumParams[exprNode.ident.name] == EXTERNAL_FUNCTION_VARARGS){
                            // in this case we can create a normal call
                            return irb.CreateCall(&*callee, args);
                        }else{
                            // otherwise, there is something weird going on
                            std::stringstream ss{};
                            ss << "Call to function " << exprNode.ident.name << " with " << args.size() << " arguments, but function has " << callee->arg_size() << " parameters";
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
                    auto& operandNode = exprNode.children[0];
                    switch(exprNode.op){
                        // can be TILDE, MINUS, AMPERSAND, LOGICAL_NOT
                        case Token::Type::TILDE:
                            return irb.CreateNot(genExpr(operandNode, irb));
                        case Token::Type::MINUS:
                            return irb.CreateNeg(genExpr(operandNode, irb)); // creates a sub with 0 as first op
                        case Token::Type::LOGICAL_NOT:
                            {
                                auto cmp = irb.CreateICmp(llvm::CmpInst::ICMP_EQ, genExpr(operandNode, irb), irb.getInt64(0));
                                return irb.CreateZExt(cmp, i64);
                            }
                        case Token::Type::AMPERSAND:
                            {
                                // get the ptr to the alloca then cast that to an int, because everything (except the auto vars stored in the varmap) is an i64
                                if(operandNode.type == ASTNode::Type::NExprVar){
                                    auto ptr = varmapLookup(irb.GetInsertBlock(), operandNode);
                                    return irb.CreatePtrToInt(ptr, i64);
                                }else{ /* has to be a subscript in this case, because of the SemanticAnalysis constraints */
                                    // REFACTOR: would be nice to merge this with the code for loading from subscripts at some point, its almost the same
                                    auto& addrNode     = operandNode.children[0];
                                    auto& indexNode    = operandNode.children[1];
                                    auto& sizespecNode = operandNode.children[2];

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
                    auto& lhsNode = exprNode.children[0];
                    auto& rhsNode = exprNode.children[1];

                        // all the following logical ops need to return i64s to conform to the C like behavior we want

#define ICAST(irb, x) irb.CreateIntCast((x), i64, false) // unsigned cast because we want 0 for false and 1 for true (instead of -1)

                    // 2 edge cases: assignment, and short circuiting logical ops:
                    // short circuiting logical ops: conditional evaluation

                    // can get preds using: llvm::pred_begin()/ llvm::predecessors()
                    if(bool isAnd = exprNode.op == Token::Type::LOGICAL_AND; isAnd || exprNode.op == Token::Type::LOGICAL_OR){
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
                            auto addr = genExpr(lhsNode.children[0], irb);
                            auto index = genExpr(lhsNode.children[1], irb);
                            auto& sizespecNode = lhsNode.children[2];

                            llvm::Type* type = sizespecToLLVMType(sizespecNode, irb);
                            // first cast, then store, so that the right amount is stored
                            // this could be done with a trunc, but that is only allowed if the type is strictly smaller, the CreateIntCast distinguishes these cases and takes care of it for us
                            auto cast = irb.CreateIntCast(rhs, type, true);

                            auto addrPtr = irb.CreateIntToPtr(addr, irb.getPtrTy()); // opaque ptrs galore!

                            auto getelementpointer = irb.CreateGEP(type, addrPtr, {index});
                            irb.CreateStore(cast, getelementpointer);
                        }else if(/* lhs node has to be var if we're here */ lhsNode.ident.type == IdentifierInfo::AUTO){
                            irb.CreateStore(rhs, varmapLookup(irb.GetInsertBlock(), lhsNode));
                        }else{/* in this case it has to be a register variable */
                            // in lhs: "old" varname of the var we're assigning to -> update mapping
                            // in rhs: value to assign to it
                            assert(lhsNode.ident.type == IdentifierInfo::REGISTER && "if lhs is not AUTO, it has to be REGISTER");

                            updateVarmap(irb, lhsNode, rhs);
                        }
                        return rhs; // just as before, return the result, not the store/assign/etc.
                    }
                    auto lhs = genExpr(lhsNode, irb);

                    // for all other cases this is a post order traversal of the epxr tree

                    switch(exprNode.op){
                        case Token::Type::BITWISE_OR:    return irb.CreateOr(lhs,rhs);
                        case Token::Type::BITWISE_XOR:   return irb.CreateXor(lhs,rhs);
                        case Token::Type::AMPERSAND:     return irb.CreateAnd(lhs,rhs);
                        case Token::Type::PLUS:          return irb.CreateAdd(lhs,rhs);
                        case Token::Type::MINUS:         return irb.CreateSub(lhs,rhs);
                        case Token::Type::TIMES:         return irb.CreateMul(lhs,rhs);
                        case Token::Type::DIV:           return irb.CreateSDiv(lhs,rhs);
                        case Token::Type::MOD:           return irb.CreateSRem(lhs,rhs);
                        case Token::Type::SHIFTL:        return irb.CreateShl(lhs,rhs);
                        case Token::Type::SHIFTR:        return irb.CreateAShr(lhs,rhs);

                        case Token::Type::LESS:          return ICAST(irb,irb.CreateICmp(llvm::CmpInst::ICMP_SLT, lhs, rhs));
                        case Token::Type::GREATER:       return ICAST(irb,irb.CreateICmp(llvm::CmpInst::ICMP_SGT, lhs, rhs));
                        case Token::Type::LESS_EQUAL:    return ICAST(irb,irb.CreateICmp(llvm::CmpInst::ICMP_SLE, lhs, rhs));
                        case Token::Type::GREATER_EQUAL: return ICAST(irb,irb.CreateICmp(llvm::CmpInst::ICMP_SGE, lhs, rhs));
                        case Token::Type::EQUAL:         return ICAST(irb,irb.CreateICmp(llvm::CmpInst::ICMP_EQ, lhs, rhs));
                        case Token::Type::NOT_EQUAL:     return ICAST(irb,irb.CreateICmp(llvm::CmpInst::ICMP_NE, lhs, rhs));
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
                    assert(exprNode.children.size() == 3 && "Subscript load expression must have 3 children");

                    auto& addrNode     = exprNode.children[0];
                    auto& indexNode    = exprNode.children[1];
                    auto& sizespecNode = exprNode.children[2];

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

    void genStmts(std::vector<ASTNode>& stmts, llvm::IRBuilder<>& irb);

    void genStmt(ASTNode& stmtNode, llvm::IRBuilder<>& irb){
        switch(stmtNode.type){
            case ASTNode::Type::NStmtDecl:
                {
                    // TODO NStmtDecl only has one child (initializer) anymore, so rework this
                    auto initializer = genExpr(stmtNode.children[0], irb);

                    // i hope setting names doesn't hurt performance, but it wouldn't make sense if it did
                    if(stmtNode.ident.type == IdentifierInfo::AUTO){
                        auto entryBB = &currentFunction->getEntryBlock();
                        auto nonphi = entryBB->getFirstNonPHI();
                        llvm::IRBuilder<> entryIRB(entryBB, entryBB->getFirstInsertionPt()); // i hope this is correct
                        if(nonphi != nullptr){
                            entryIRB.SetInsertPoint(nonphi);
                        }

                        auto alloca = entryIRB.CreateAlloca(i64, nullptr, stmtNode.ident.name); // this returns the ptr to the alloca'd memory
                        irb.CreateStore(initializer, alloca);

                        autoVarmap[stmtNode.ident.uID] = alloca;
                    }else if(stmtNode.ident.type == IdentifierInfo::REGISTER){
                        updateVarmap(irb, stmtNode, initializer);
                    }else{
                        assert(false && "Invalid identifier type");
                    }
                }
                break;
            case ASTNode::Type::NStmtReturn:
                if(stmtNode.children.size() == 0){
                    auto retrn = irb.CreateRet(llvm::PoisonValue::get(i64));
                    warn("Returning nothing is undefined behavior", retrn);
                }else{
                    irb.CreateRet(genExpr(stmtNode.children[0], irb));
                }
                break;
            case ASTNode::Type::NStmtBlock:
                genStmts(stmtNode.children, irb);
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

                    auto condition = genExpr(stmtNode.children[0], irb); // as everything in our beautiful C like language, this is an i64, so "cast" it to an i1
                    condition = irb.CreateICmp(llvm::CmpInst::ICMP_NE, condition, irb.getInt64(0));
                    irb.CreateCondBr(condition, thenBB, elseBB);
                    // block is now finished
                    
                    auto& [thenSealed, thenVarmap] = blockInfo[thenBB];
                    thenSealed = true;
                    // var map is queried recursively anyway, would be a waste to copy it here

                    irb.SetInsertPoint(thenBB);
                    genStmt(stmtNode.children[1], irb);
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
                        genStmt(stmtNode.children[2], irb);
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

                    auto conditionExpr = genExpr(stmtNode.children[0], irb); // as everything in our beautiful C like language, this is an i64, so "cast" it to an i1
                    conditionExpr = irb.CreateICmp(llvm::CmpInst::ICMP_NE, conditionExpr, irb.getInt64(0));
                    irb.CreateCondBr(conditionExpr, bodyBB, contBB);

                    blockInfo[bodyBB].sealed = true; // can only get into body block from condition block -> all predecessors are known

                    irb.SetInsertPoint(bodyBB);
                    genStmt(stmtNode.children[1], irb);
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

    void genStmts(std::vector<ASTNode>& stmts, llvm::IRBuilder<>& irb){
        for(auto& stmt : stmts){
            genStmt(stmt, irb);
            if(stmt.type == ASTNode::Type::NStmtReturn){
                // stop the generation for this block
                break;
            }
        }
    }

    void genFunction(ASTNode& fnNode){
        auto paramNum = fnNode.children[0].children.size();
        auto typelist = llvm::SmallVector<llvm::Type*, 8>(paramNum, i64);
        llvm::Function* fn = findFunction(fnNode.ident.name);
        currentFunction = fn;
        llvm::BasicBlock* entryBB = llvm::BasicBlock::Create(ctx, "entry", fn);
        // simply use getEntryBlock() on the fn when declarations need to be added
        blockInfo[entryBB].sealed = true;
        llvm::IRBuilder<> irb(entryBB);

        for(unsigned int i = 0; i < paramNum; i++){
            llvm::Argument* arg = fn->getArg(i);
            auto& name = fnNode.children[0].children[i].ident.name;
            arg->setName(name);
            updateVarmap(irb, fnNode.children[0].children[i], arg);
        }

        auto& blockNode = fnNode.children[1];
        genStmt(blockNode, irb); // calls gen stmts on the blocks children, but does additional stuff

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
            warn("Function " + fn->getName().str() + " does not seem to be guaranteed to return a value, this is undefined behvaior.");
        }
    }

    bool generate(AST& ast, llvm::raw_ostream* out){
        ASTNode& root = ast.root;

        // declare implicitly declared functions
        for(auto& entry: SemanticAnalysis::externalFunctionsToNumParams){
            auto fnParamCount = entry.second;
            llvm::SmallVector<llvm::Type*, 8> params;
            if(fnParamCount == EXTERNAL_FUNCTION_VARARGS){
                params = {};
            }else{
                params = llvm::SmallVector<llvm::Type*, 8>(fnParamCount, i64);

            }
            llvm::FunctionType* fnTy = llvm::FunctionType::get(i64, params, fnParamCount == EXTERNAL_FUNCTION_VARARGS);
            llvm::Function::Create(fnTy, llvm::GlobalValue::ExternalLinkage, entry.first(), moduleUP.get()); 
        }

        // declare all functions in the file, to easily allow forward declarations
        auto& children = root.children;
        for(auto& fnNode : children){
            auto paramNum = fnNode.children[0].children.size();
            auto typelist = llvm::SmallVector<llvm::Type*, 8>(paramNum, i64);
            llvm::FunctionType* fnTy = llvm::FunctionType::get(i64, typelist, false);
            if(findFunction(fnNode.ident.name)){
                std::cerr << "fatal error: redefinition of function '" << fnNode.ident.name << "'\n";
                return false;
            }
            llvm::Function* fn = llvm::Function::Create(fnTy, llvm::GlobalValue::ExternalLinkage, fnNode.ident.name, moduleUP.get());
            for(unsigned int i = 0; i < paramNum; i++){
                llvm::Argument* arg = fn->getArg(i);
                auto& name = fnNode.children[0].children[i].ident.name;
                arg->setName(name);
            }
        }

        for(auto& child:children){
            genFunction(child);
        }


        bool moduleIsBroken = llvm::verifyModule(*moduleUP, &llvm::errs());
        if(out){
            moduleUP->print(*out, nullptr);
        }

        if(warningsGenerated)
            llvm::errs() << "Warnings were generated during code generation, please check the output for more information\n";
        return !moduleIsBroken;
    }

} // end namespace CodeGen


/*
 Subset of LLVM IR used (checks/not checks: done/not done):
 - Format: <instr>  [<operand type(s)>]
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
 - [x] Poison Values    [i64/ptr] (don't need special handling)

 Not relevant for this task, but used:
 - Call             [return: i64, args: i64...]
 - PHIs             [i64, ptr]
 - Ret              [i64]
 - Unreachable

 ----------------------------------------------------
 ARM (v8-A) subset used: See instructionFunctions

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
        static std::unordered_map<Pattern, llvm::Value* (*)(llvm::IRBuilder<>&), PatternHash> replacementCalls;

        void replaceWithARM(llvm::Instruction* instr) const{

            // TODO with lambda
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
            auto replacement = Pattern::replacementCalls[*this](irb); 

            // remove all operands of the instruction from the program
            std::queue<llvm::Instruction*> toRemove{};
            std::queue<const Pattern*> patternQueue{};
            PUSH_OPS(instr, this);
            while(!toRemove.empty()){
                instr = toRemove.front();
                toRemove.pop();

                PUSH_OPS(instr, patternQueue.front());
                patternQueue.pop();

                instr->eraseFromParent();
            }

            if(!noDelete && root != replacement){
                root->replaceAllUsesWith(replacement);
                root->eraseFromParent();
            }

#undef PUSH_OPS
        }

    private:
        // REFACTOR it might be worth having an alternative std::function for matching, which simply gets the llvm value as an argument and returns true if it matches,
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

        static Pattern make_root(llvm::Value* (*replacementCall)(llvm::IRBuilder<>&), unsigned isMatching = 0, std::vector<Pattern> children = {}, bool noDelete = false){
        //static Pattern make_root(std::function<llvm::Value*(llvm::IRBuilder<>&)> replacementCall, unsigned isMatching = 0, std::vector<Pattern> children = {}, std::initializer_list<Pattern> alternatives = {}){
            Pattern rootPattern{isMatching, children, false, true, noDelete};
            Pattern::replacementCalls[rootPattern] = replacementCall;
            return rootPattern;
        }

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
            if(type!=0 && !root && val->hasNUsesOrMore(2)) return false; // this requirement is important for all 'real' inner nodes, i.e. all except the root and leaves
                                                           // leaves should have type 0

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
    }; // struct Pattern

    const Pattern Pattern::emptyPattern = Pattern{0,{}, false, false, true};;
    std::unordered_map<Pattern, llvm::Value* (*)(llvm::IRBuilder<>&), Pattern::PatternHash> Pattern::replacementCalls{};
    const Pattern Pattern::constantPattern = Pattern(0, {}, true, false, true);

    std::unordered_set<unsigned> skippableTypes{
        llvm::Instruction::Ret,
        llvm::Instruction::Unreachable,
        llvm::Instruction::PHI,
        llvm::Instruction::Alloca,
    };

    /// for useful matching the patterns need to be sorted by totalSize (descending) here. For this simple isel, this is just done by hand
    std::unordered_map<llvm::Instruction*, const Pattern&> matchPatterns(llvm::Function* func, const std::vector<Pattern>& patterns){
        std::unordered_map<llvm::Instruction*, const Pattern&> matches{};
        std::unordered_set<llvm::Instruction*> covered{};
        for(auto& block:*func){
            // iterate over the instructions in a bb in reverse order, to iterate over the dataflow top down
            for(auto& instr:llvm::reverse(block)){

                // skip returns, phis, calls, etc.
                if(skippableTypes.contains(instr.getOpcode())) continue;

                // skip instructions that are already matched
                if(covered.contains(&instr)) continue;
                
                // find largest pattern that matches
                for(auto& pattern:patterns){
                    if(pattern.match(&instr)){
                        matches.emplace(&instr, pattern);
                        /*
                          the following code boils down to:

                          addCovered(&covered, &pattern, &instr){
                              for(auto& child:instr->operands()){
                                  covered[child] = pattern;
                                  addCovered(covered, pattern, child);
                              }
                          }
                        */

                        std::queue<const Pattern*> patternQueue{};
                        patternQueue.push(&pattern);

                        std::queue<llvm::Instruction*> coveredInsertionQueue{}; // lets not do this recursively...
                        coveredInsertionQueue.push(&instr);
                        while(!coveredInsertionQueue.empty()){
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
                EXIT_TODO;

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

    } // namespace Codegen::ISel

namespace Codegen{
    // not an enum class for readability, all instructions are prefixed with ARM_ anyway
    enum ARMInstruction{
        ARM_add,
        ARM_add_SP,
        ARM_add_SHIFT,
        ARM_sub,
        ARM_sub_SP,
        ARM_sub_SHIFT,
        ARM_madd,
        ARM_msub,
        ARM_sdiv,

        ARM_cmp,
        ARM_csel,
        ARM_csel_i1,

        ARM_lsl_imm,
        ARM_lsl_var,
        ARM_asr_imm,
        ARM_asr_var,

        ARM_and,
        ARM_orr,
        ARM_eor,

        ARM_mov,

        ARM_ldr,
        ARM_ldr_sb,
        ARM_ldr_sh,
        ARM_ldr_sw,
        ARM_str,
        ARM_str32,
        ARM_str32_b,
        ARM_str32_h,

        ARM_PSEUDO_str, // for stores which don't know what register they're storing from yet
        ARM_PSEUDO_addr_computation, // specifically to handle ptrtoint scenarios, where we want to use a pointer like an int, this needs a register etc. 

        ARM_b_cond,
        ARM_b,
        ARM_b_cbnz,
        ARM_b_cbz,

        ZExt_handled_in_Reg_Alloc,
    };

    /// functions to serve as substitute for actual ARM instructions
#define CREATE_INST_FN(name, ret, ...)                          \
    {                                                           \
        name,                                                   \
        llvm::Function::Create(                                 \
            llvm::FunctionType::get(ret, {__VA_ARGS__}, false), \
            llvm::GlobalValue::ExternalLinkage,                 \
            #name,                                              \
            *moduleUP                                           \
        )                                                       \
    }

#define CREATE_INST_FN_VARARGS(name, ret)           \
    {                                               \
        name,                                       \
        llvm::Function::Create(                     \
            llvm::FunctionType::get(ret, {}, true), \
            llvm::GlobalValue::ExternalLinkage,     \
            #name,                                  \
            *moduleUP                               \
        )                                           \
    }

    // REFACTOR: make an array, if i find a way to nicely count the number of enum values
    static std::unordered_map<ARMInstruction, llvm::Function*> instructionFunctions;
			// REFACTOR: If I initialize this directly, llvm decides not to use opaque pointers for allocas for some reason. Don't ask me why
    static void initInstructionFunctions(){
		instructionFunctions = {
        CREATE_INST_FN(ARM_add,       i64,                            i64,  i64),
        CREATE_INST_FN(ARM_add_SP,    voidTy,                         i64), // simulate add to stack pointer
        CREATE_INST_FN(ARM_add_SHIFT, i64,                            i64,  i64,                               i64),
        CREATE_INST_FN(ARM_sub,       i64,                            i64,  i64),
        CREATE_INST_FN(ARM_sub_SP,    llvm::PointerType::get(ctx,0), i64), // simulate sub from stack pointer
        CREATE_INST_FN(ARM_sub_SHIFT, i64,                            i64,  i64,                               i64),
        CREATE_INST_FN(ARM_madd,      i64,                            i64,  i64,                               i64),
        CREATE_INST_FN(ARM_msub,      i64,                            i64,  i64,                               i64),
        CREATE_INST_FN(ARM_sdiv,      i64,                            i64,  i64),


        // cmp
        CREATE_INST_FN(ARM_cmp,      i64, i64, i64),
        CREATE_INST_FN_VARARGS(ARM_csel,       i64), // condition is represented as string, so use varargs
        CREATE_INST_FN_VARARGS(ARM_csel_i1,    llvm::Type::getInt1Ty(ctx)), // condition is represented as string, so use varargs

        // shifts by variable and immediate amount
        CREATE_INST_FN(ARM_lsl_imm,  i64,  i64, i64),
        CREATE_INST_FN(ARM_lsl_var,  i64,  i64, i64),
        CREATE_INST_FN(ARM_asr_imm,  i64,  i64, i64),
        CREATE_INST_FN(ARM_asr_var,  i64,  i64, i64),

        // bitwise
        CREATE_INST_FN(ARM_and,      i64, i64, i64),
        CREATE_INST_FN(ARM_orr,      i64, i64, i64),
        CREATE_INST_FN(ARM_eor,      i64, i64, i64), // XOR

        // mov (varargs to accept both i64 and i1)
        CREATE_INST_FN_VARARGS(ARM_mov,       i64),

        // memory access
        // (with varargs to be able to simulate the different addressing modes)
        CREATE_INST_FN_VARARGS(ARM_ldr,      i64),
        CREATE_INST_FN_VARARGS(ARM_ldr_sb,   i64),
        CREATE_INST_FN_VARARGS(ARM_ldr_sh,   i64),
        CREATE_INST_FN_VARARGS(ARM_ldr_sw,   i64),
        CREATE_INST_FN_VARARGS(ARM_str,      voidTy),
        CREATE_INST_FN_VARARGS(ARM_str32,    voidTy),
        CREATE_INST_FN_VARARGS(ARM_str32_b,  voidTy),
        CREATE_INST_FN_VARARGS(ARM_str32_h,  voidTy),

        CREATE_INST_FN_VARARGS(ARM_PSEUDO_str, voidTy),
        CREATE_INST_FN_VARARGS(ARM_PSEUDO_addr_computation, i64),

        // control flow/branches
        CREATE_INST_FN(ARM_b,              voidTy),
        CREATE_INST_FN(ARM_b_cond,         llvm::Type::getInt1Ty(ctx), i64),
        CREATE_INST_FN(ARM_b_cbnz,         llvm::Type::getInt1Ty(ctx), i64),
        CREATE_INST_FN(ARM_b_cbz,          llvm::Type::getInt1Ty(ctx), i64),

        // 'metadata' calls
        CREATE_INST_FN_VARARGS(ZExt_handled_in_Reg_Alloc, i64),
		};
    };




#undef CREATE_INST_FN

    // ARM zero register, technically not necessary, but its nice programatically, in order not to use immediate operands where its not possible
    auto XZR = llvm::ConstantInt::get(i64, 0);

} // namespace Codegen

// TODO for some reason the branch reinsertion seems to turn some conditional branches into unconditional ones (on constexpr conditions). This would fine, except that it breaks the phi nodes.
namespace Codegen::ISel{
    // TODO is there any way to make this nicer? The patterns themselves have so much 'similar' code, but its hard to factor out into something common

/// always inserts a mov to materialize the given constant
#define MAT_CONST(value) irb.CreateCall(instructionFunctions[ARM_mov], {value})

/// check if the given value is a constant and, if so, materialize it
#define MAYBE_MAT_CONST(value)                                                                           \
        (llvm::isa<llvm::ConstantInt>(value) && (!llvm::dyn_cast<llvm::ConstantInt>(value)->isZero())) ? \
            (MAT_CONST(value)) :                                                                         \
            (value)

/// get the nth operand of the instruction
#define OP_N(instr, N) \
        instr->getOperand(N)
            
/// gets the nth operand of the instruction and materializes it if necessary
#define OP_N_MAT(instr, N) \
        MAYBE_MAT_CONST(OP_N(instr, N))

    /// all ARM patterns
    /// these are matched in order, so they are roughly sorted by size
    const std::vector<Pattern> patterns{
        // madd
        Pattern::make_root(
            [](llvm::IRBuilder<>& irb) -> llvm::Value* {
                auto instr = &*irb.GetInsertPoint();
                // first 2 args is mul, last one is add
                auto* mul = llvm::dyn_cast<llvm::Instruction>(OP_N(instr,0));
                auto mulOp1 = OP_N_MAT(mul, 0);
                auto mulOp2 = OP_N_MAT(mul, 1);

                auto addOp2 = OP_N_MAT(instr, 1);

                auto fn = instructionFunctions[ARM_madd];
                return irb.CreateCall(fn, {mulOp1, mulOp2, addOp2});
            },
            llvm::Instruction::Add,
            {
                {llvm::Instruction::Mul, {}, },
                {},
            }
        ),
        Pattern::make_root(
            [](llvm::IRBuilder<>& irb) -> llvm::Value* {
                auto instr = &*irb.GetInsertPoint();

                // first 2 args is mul, last one is add
                auto* mul = llvm::dyn_cast<llvm::Instruction>(OP_N(instr,1));
                auto mulOp1 = OP_N_MAT(mul,0);
                auto mulOp2 = OP_N_MAT(mul,1);

                auto addOp1 = OP_N_MAT(instr,0);

                auto fn = instructionFunctions[ARM_madd];
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
            [](llvm::IRBuilder<>& irb) -> llvm::Value* {
                auto instr = &*irb.GetInsertPoint();

                // first 2 args is mul, last one is sub
                auto* mul = llvm::dyn_cast<llvm::Instruction>(OP_N(instr,1));
                auto mulOp1 = OP_N_MAT(mul,0);
                auto mulOp2 = OP_N_MAT(mul,1);

                auto subOp1 = OP_N_MAT(instr,0);

                auto fn = instructionFunctions[ARM_msub];
                return irb.CreateCall(fn, {mulOp1, mulOp2, subOp1});
            },
            llvm::Instruction::Sub,
            {
                {},
                {llvm::Instruction::Mul, {}},
            }
        ),

#define TWO_OPERAND_INSTR_PATTERN(llvmInstr, armInstr)                                                           \
        Pattern::make_root(                                                                                      \
            [](llvm::IRBuilder<>& irb) -> llvm::Value* {                                                                          \
                auto instr = &*irb.GetInsertPoint();                                                             \
                return irb.CreateCall(instructionFunctions[armInstr], {OP_N_MAT(instr,0), OP_N_MAT(instr,1)}); \
            },                                                                                                   \
            llvm::Instruction::llvmInstr,                                                                        \
            {}                                                                                                   \
        ),

        // shifted add/sub
        // both add variants:
        Pattern::make_root(
            [](llvm::IRBuilder<>& irb) -> llvm::Value* {
                auto instr = &*irb.GetInsertPoint();
                auto addOp1 = OP_N_MAT(instr,0);
                auto* shift = llvm::dyn_cast<llvm::Instruction>(OP_N(instr,1));
                auto shiftOp1 = OP_N_MAT(shift,0);
                auto shiftOp2 = OP_N(shift,1); // this has to be an immediate
                auto fn = instructionFunctions[ARM_add_SHIFT];
                return irb.CreateCall(fn, {addOp1, shiftOp1, shiftOp2});
            },
            llvm::Instruction::Add,
            {
                {},
                {
                    llvm::Instruction::Shl,
                    {
                        {},
                        {Pattern::make_constant()}
                    }
                },
            }
        ),
        Pattern::make_root(
            [](llvm::IRBuilder<>& irb) -> llvm::Value* {
                auto instr = &*irb.GetInsertPoint();
                auto addOp2 = OP_N_MAT(instr,1);
                auto* shift = llvm::dyn_cast<llvm::Instruction>(OP_N(instr,0));
                auto shiftOp1 = OP_N_MAT(shift,0);
                auto shiftOp2 = OP_N(shift,1); // this has to be an immediate
                auto fn = instructionFunctions[ARM_add_SHIFT];
                return irb.CreateCall(fn, {addOp2, shiftOp1, shiftOp2});
            },
            llvm::Instruction::Add,
            {
                {
                    llvm::Instruction::Shl,
                    {
                        {},
                        {Pattern::make_constant()}
                    }
                },
                {},
            }
        ),
        // sub:
        Pattern::make_root(
            [](llvm::IRBuilder<>& irb) -> llvm::Value* {
                auto instr = &*irb.GetInsertPoint();
                auto subOp1 = OP_N_MAT(instr,0);
                auto* shift = llvm::dyn_cast<llvm::Instruction>(OP_N(instr,1));
                auto shiftOp1 = OP_N_MAT(shift,0);
                auto shiftOp2 = OP_N(shift,1); // this has to be an immediate
                auto fn = instructionFunctions[ARM_sub_SHIFT];
                return irb.CreateCall(fn, {subOp1, shiftOp1, shiftOp2});
            },
            llvm::Instruction::Sub,
            {
                {},
                {
                    llvm::Instruction::Shl,
                    {
                        {},
                        {Pattern::make_constant()}
                    }
                },
            }
        ),

        // add, sub (don't need to materialize immediate operands for the second one, they accept them)
        Pattern::make_root(
            [](llvm::IRBuilder<>& irb) -> llvm::Value* {
                auto instr = &*irb.GetInsertPoint();
                auto op1 = OP_N(instr,0);
                auto op2 = OP_N(instr,1);
                if(llvm::isa<llvm::ConstantInt>(op1)){
                    std::swap(op1, op2);
                }

                return irb.CreateCall(instructionFunctions[ARM_add], {MAYBE_MAT_CONST(op1), op2});
            },
            llvm::Instruction::Add,
            {}
        ),
        Pattern::make_root(
            [](llvm::IRBuilder<>& irb) -> llvm::Value* {
                auto instr = &*irb.GetInsertPoint();
                return irb.CreateCall(instructionFunctions[ARM_sub], {OP_N_MAT(instr,0), OP_N(instr,1)});
            },
            llvm::Instruction::Sub,
            {}
        ),

        // mul, div
        Pattern::make_root(
            [](llvm::IRBuilder<>& irb) -> llvm::Value* {
                auto instr = &*irb.GetInsertPoint();
                return irb.CreateCall(instructionFunctions[ARM_madd], {OP_N_MAT(instr,0), OP_N_MAT(instr,1), XZR});
            },
            llvm::Instruction::Mul,
            {}
        ),
        TWO_OPERAND_INSTR_PATTERN(SDiv, ARM_sdiv)
        // remainder
        Pattern::make_root(
            [](llvm::IRBuilder<>& irb) -> llvm::Value* {
            // TODO sure this is right for negative numbers?
                auto instr = &*irb.GetInsertPoint();
                auto quotient = irb.CreateCall(instructionFunctions[ARM_sdiv],  {OP_N_MAT(instr,0), OP_N_MAT(instr,1)});
                return irb.CreateCall(instructionFunctions[ARM_msub], {quotient, OP_N_MAT(instr,1), OP_N_MAT(instr,0)}); // remainder = numerator - (quotient * denominator)
            },
            llvm::Instruction::SRem,
            {}
        ),

        // shifts
        // logical left shift
        Pattern::make_root(
            [](llvm::IRBuilder<>& irb) -> llvm::Value* {
                auto instr = &*irb.GetInsertPoint();
                return irb.CreateCall(instructionFunctions[ARM_lsl_imm], {OP_N_MAT(instr,0), OP_N(instr,1)});
            },
            llvm::Instruction::Shl,
            {
                {},
                {Pattern::make_constant()}
            }
        ),
        TWO_OPERAND_INSTR_PATTERN(Shl, ARM_lsl_var)

        // arithmetic shift right
        Pattern::make_root(
            [](llvm::IRBuilder<>& irb) -> llvm::Value* {
                auto instr = &*irb.GetInsertPoint();
                return irb.CreateCall(instructionFunctions[ARM_asr_imm], {OP_N_MAT(instr,0), OP_N(instr,1)});
            },
            llvm::Instruction::AShr,
            {
                {},
                {Pattern::make_constant()}
            }
        ),
        TWO_OPERAND_INSTR_PATTERN(AShr, ARM_asr_var)

        // bitwise ops
        TWO_OPERAND_INSTR_PATTERN(And, ARM_and)
        TWO_OPERAND_INSTR_PATTERN(Or, ARM_orr)
        TWO_OPERAND_INSTR_PATTERN(Xor, ARM_eor)

        // memory

        // sign extends can only happen after loadsA
        // truncation can only happen before stores
        // load with sign extension
        Pattern::make_root(
            [](llvm::IRBuilder<>& irb) -> llvm::Value* {
                auto* sextInstr     = llvm::dyn_cast<llvm::SExtInst>(&*irb.GetInsertPoint());
                auto* loadInstr     = llvm::dyn_cast<llvm::LoadInst>(OP_N(sextInstr,0));
                auto* gepInstr      = llvm::dyn_cast<llvm::GetElementPtrInst>(loadInstr->getPointerOperand());
                auto* intToPtrInstr = llvm::dyn_cast<llvm::IntToPtrInst>(gepInstr->getPointerOperand());
                auto bitwidthOfLoad = loadInstr->getType()->getIntegerBitWidth();


                switch(bitwidthOfLoad){
                // args: base, offset, offsetshift
                    case 8:
                        return irb.CreateCall(instructionFunctions[ARM_ldr_sb], {OP_N_MAT(intToPtrInstr,0), OP_N_MAT(gepInstr,1), irb.getInt8(0)});
                    case 16:
                        return irb.CreateCall(instructionFunctions[ARM_ldr_sh], {OP_N_MAT(intToPtrInstr,0), OP_N_MAT(gepInstr,1), irb.getInt8(1)});
                    case 32:
                        return irb.CreateCall(instructionFunctions[ARM_ldr_sw], {OP_N_MAT(intToPtrInstr,0), OP_N_MAT(gepInstr,1), irb.getInt8(2)});
                    default: // on 64, the sext is not created by llvm:
                        errx(ExitCode::ERROR_CODEGEN, "Fatal pattern matching error during ISel: sext of load with bitwidth %d not supported", bitwidthOfLoad);
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
            [](llvm::IRBuilder<>& irb) -> llvm::Value* {
                auto* loadInstr     = llvm::dyn_cast<llvm::LoadInst>(&*irb.GetInsertPoint());
                auto* gepInstr      = llvm::dyn_cast<llvm::GetElementPtrInst>(loadInstr->getPointerOperand());
                auto* intToPtrInstr = llvm::dyn_cast<llvm::IntToPtrInst>(gepInstr->getPointerOperand());

                // because it doesn't have a sign extension, it is guaranteed to be a 64 bit load
                return irb.CreateCall(instructionFunctions[ARM_ldr], {OP_N_MAT(intToPtrInstr,0), OP_N_MAT(gepInstr,1), irb.getInt8(3)}); // shift by 3 i.e. times 8
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
            [](llvm::IRBuilder<>& irb) -> llvm::Value* {
                auto* storeInst     = llvm::dyn_cast<llvm::StoreInst>(&*irb.GetInsertPoint());
                auto* gepInstr      = llvm::dyn_cast<llvm::GetElementPtrInst>(storeInst->getPointerOperand());
                auto* intToPtrInstr = llvm::dyn_cast<llvm::IntToPtrInst>(gepInstr->getPointerOperand());
                auto* truncInstr    = llvm::dyn_cast<llvm::TruncInst>(storeInst->getValueOperand());
                auto bitwidthOfStore = truncInstr->getType()->getIntegerBitWidth();
                
                switch(bitwidthOfStore){
                // args: value, base, offset, offsetshift
                    case 8:
                        return irb.CreateCall(instructionFunctions[ARM_str32_b], {OP_N_MAT(truncInstr,0), OP_N_MAT(intToPtrInstr,0), OP_N_MAT(gepInstr,1), irb.getInt8(0)});
                    case 16:
                        return irb.CreateCall(instructionFunctions[ARM_str32_h], {OP_N_MAT(truncInstr,0), OP_N_MAT(intToPtrInstr,0), OP_N_MAT(gepInstr,1), irb.getInt8(1)});
                    case 32:
                        return irb.CreateCall(instructionFunctions[ARM_str32], {OP_N_MAT(truncInstr,0), OP_N_MAT(intToPtrInstr,0), OP_N_MAT(gepInstr,1), irb.getInt8(2)});
                    default: // on 64, the trunc is not created by llvm:
                        errx(ExitCode::ERROR_CODEGEN, "Fatal pattern matching error during ISel: trunc of store with bitwidth %d not supported", bitwidthOfStore);
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
            [](llvm::IRBuilder<>& irb) -> llvm::Value* {
                auto* storeInst     = llvm::dyn_cast<llvm::StoreInst>(&*irb.GetInsertPoint());

                // even though we have no truncation here, if its a constant it might still be < 64 bit, so we need to convert it
                auto* storeValue    = storeInst->getValueOperand();
                if(auto* constInt = llvm::dyn_cast_if_present<llvm::ConstantInt>(storeValue)){
                    storeValue = irb.getInt64(constInt->getZExtValue());
                }

                auto* gepInstr      = llvm::dyn_cast<llvm::GetElementPtrInst>(storeInst->getPointerOperand());
                auto* intToPtrInstr = llvm::dyn_cast<llvm::IntToPtrInst>(gepInstr->getPointerOperand());
                auto bitwidthOfStore = gepInstr->getResultElementType()->getIntegerBitWidth();
                
                switch(bitwidthOfStore){
                // args: value, base, offset, offsetshift
                    case 8:
                        return irb.CreateCall(instructionFunctions[ARM_str32_b], {MAYBE_MAT_CONST(storeValue), OP_N_MAT(intToPtrInstr,0), OP_N_MAT(gepInstr,1), irb.getInt8(0)});
                    case 16:
                        return irb.CreateCall(instructionFunctions[ARM_str32_h], {MAYBE_MAT_CONST(storeValue), OP_N_MAT(intToPtrInstr,0), OP_N_MAT(gepInstr,1), irb.getInt8(1)});
                    case 32:
                        return irb.CreateCall(instructionFunctions[ARM_str32], {MAYBE_MAT_CONST(storeValue), OP_N_MAT(intToPtrInstr,0), OP_N_MAT(gepInstr,1), irb.getInt8(2)});
                    case 64:
                        return irb.CreateCall(instructionFunctions[ARM_str], {MAYBE_MAT_CONST(storeValue), OP_N_MAT(intToPtrInstr, 0), OP_N_MAT(gepInstr,1), irb.getInt8(3)}); // shift by 3 for multiplying by 8
                    default: 
                        errx(ExitCode::ERROR_CODEGEN, "Fatal pattern matching error during ISel: trunc of store with bitwidth %d not supported", bitwidthOfStore);
                }
            },
            llvm::Instruction::Store,
            { 
                // store arg is cast to the target type, so truncated, or target is already i64 or immediate. This case handles non-truncation i.e. arbitrary first (value) arg
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
            [](llvm::IRBuilder<>& irb) -> llvm::Value* {
                auto instr = &*irb.GetInsertPoint();
                return irb.CreateCall(instructionFunctions[ARM_ldr], {OP_N_MAT(instr,0)});
            },
            llvm::Instruction::Load,
            {}
        ),
        Pattern::make_root(
            [](llvm::IRBuilder<>& irb) -> llvm::Value* {
                auto instr = &*irb.GetInsertPoint();
                return irb.CreateCall(instructionFunctions[ARM_str], {OP_N_MAT(instr,0), OP_N_MAT(instr,1)});
            },
            llvm::Instruction::Store,
            {}
        ),

        // subscript with addrof
        Pattern::make_root(
            [](llvm::IRBuilder<>& irb) -> llvm::Value* {
                // add instruction for address calculation
                auto* ptrToInt = llvm::dyn_cast<llvm::PtrToIntInst>(irb.GetInsertPoint());
                auto* gep = llvm::dyn_cast<llvm::GetElementPtrInst>(ptrToInt->getPointerOperand());
                auto* intToPtr = llvm::dyn_cast<llvm::IntToPtrInst>(gep->getOperand(0));

                int bitwidth = gep->getSourceElementType()->getIntegerBitWidth();
                unsigned shiftInt = 0;
                // TODO transform into log2 + sub
                switch(bitwidth){
                    case 8: shiftInt = 0; break;
                    case 16: shiftInt = 1; break;
                    case 32: shiftInt = 2; break;
                    case 64: shiftInt = 3; break;
                    default:
                        errx(ExitCode::ERROR_CODEGEN, "Fatal pattern matching error during ISel: subscript with bitwidth %d not supported", bitwidth);
                }

                auto indexOp = OP_N(gep, 1);
                if(auto indexConst = llvm::dyn_cast_or_null<llvm::ConstantInt>(indexOp)){
                    auto index = indexConst->getSExtValue();
                    return irb.CreateCall(instructionFunctions[ARM_add], {OP_N_MAT(intToPtr,0), irb.getInt64(index << shiftInt)});
                } else {
                    return irb.CreateCall(instructionFunctions[ARM_add_SHIFT], {OP_N_MAT(intToPtr, 0), indexOp, irb.getInt64(shiftInt)});
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

        // ptr to int, can't just remove it, because we need to make the address computation
        Pattern::make_root(
            [](llvm::IRBuilder<>& irb) -> llvm::Value* {
                auto instr = &*irb.GetInsertPoint();
                return irb.CreateCall(instructionFunctions[ARM_PSEUDO_addr_computation], {OP_N_MAT(instr,0)});
            },
            llvm::Instruction::PtrToInt,
            {}
        ),


        // control flow/branches
        // conditional branches always have an icmp NE as their condition, if we match them before the unconditional ones, the plain Br match without children always matches only unconditional ones
        Pattern::make_root(
            [](llvm::IRBuilder<>& irb) -> llvm::Value* {
                auto instr      = &*irb.GetInsertPoint();
                auto cond       = OP_N_MAT(instr,0);
                auto innerCond  =
                    llvm::dyn_cast<llvm::ICmpInst>(
                        llvm::dyn_cast<llvm::ZExtInst>(
                            llvm::dyn_cast<llvm::ICmpInst>(cond)->getOperand(0))->getOperand(0));
                auto pred       = innerCond->getSignedPredicate();
                auto predStr    = llvmPredicateToARM(pred);

                // true and false block are reversed, because we have the negated (ne) condition
                auto falseBlock  = llvm::dyn_cast<llvm::BasicBlock>(OP_N(instr,1));
                auto trueBlock = llvm::dyn_cast<llvm::BasicBlock>(OP_N(instr,2));

                auto cmp = irb.CreateCall(instructionFunctions[ARM_cmp], {OP_N_MAT(innerCond,0), OP_N_MAT(innerCond,1)});

                // fallthrough
                if(irb.GetInsertBlock()->getNextNode() != trueBlock){
                    auto brTrue = irb.CreateCall(instructionFunctions[ARM_b_cond], {cmp});
                    llvmSetStringMetadata(brTrue, "pred", predStr);
                    llvmSetStringMetadata(brTrue, "label", trueBlock->getName());
                }

                // fallthrough
                if(irb.GetInsertBlock()->getNextNode() != falseBlock){
                    auto brFalse = irb.CreateCall(instructionFunctions[ARM_b_cond], {cmp});
                    pred = llvm::ICmpInst::getInversePredicate(pred);
                    predStr = llvmPredicateToARM(pred);
                    llvmSetStringMetadata(brFalse, "pred", predStr);
                    llvmSetStringMetadata(brFalse, "label", falseBlock->getName());
                }

                // reinsert a branch to keep llvm happy and the block well formed, this is ignored later
                return irb.CreateCondBr(irb.getFalse(), trueBlock, falseBlock);
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
            [](llvm::IRBuilder<>& irb) -> llvm::Value* {
                auto instr      = &*irb.GetInsertPoint();
                auto cond       = llvm::dyn_cast<llvm::ICmpInst>(OP_N(instr,0));
                auto condInner  = OP_N_MAT(cond,0);

                // true and false block are reversed, because we have the negated (ne) condition
                auto falseBlock = llvm::dyn_cast<llvm::BasicBlock>(OP_N(instr,1));
                auto trueBlock  = llvm::dyn_cast<llvm::BasicBlock>(OP_N(instr,2));


                // fallthrough
                if(irb.GetInsertBlock()->getNextNode() != trueBlock){
                    auto cbnz = irb.CreateCall(instructionFunctions[ARM_b_cbnz], {condInner});
                    llvmSetStringMetadata(cbnz, "label", trueBlock->getName());
                }

                // fallthrough
                if(irb.GetInsertBlock()->getNextNode() != falseBlock){
                    auto cbz = irb.CreateCall(instructionFunctions[ARM_b_cbz], {condInner});
                    llvmSetStringMetadata(cbz, "label", falseBlock->getName());
                }

                // reinsert a branch to keep llvm happy and the block well formed, this is ignored later
                return irb.CreateCondBr(irb.getFalse(), trueBlock, falseBlock);
            },
            llvm::Instruction::Br,
            {
                {llvm::Instruction::ICmp}, // no requirements, because it can only be NE
                {},
                {},
            }
        ),
        Pattern::make_root(
            [](llvm::IRBuilder<>& irb) -> llvm::Value* {
                auto instr = dyn_cast<llvm::BranchInst>(&*irb.GetInsertPoint());
                // special case: through constant folding, this normally only unconditional branch can be matched by a conditional branch, with an i1 constant (true/false) as its condition
                // -> replace it with an unconditional branch
                if(instr->isConditional()){
                    auto cond = instr->getCondition();

                    assert(llvm::isa<llvm::ConstantInt>(cond) && "conditional branch condition is not a constant int, this means we didn't catch a conditional branch before");

                    auto trueBlock = instr->getSuccessor(0);
                    auto falseBlock = instr->getSuccessor(1);

                    auto branchTo = llvm::dyn_cast<llvm::ConstantInt>(cond)->isZero() ? falseBlock : trueBlock;
                    auto branchNotTo = llvm::dyn_cast<llvm::ConstantInt>(cond)->isZero() ? trueBlock : falseBlock;

                    // remove the incoming edge from the phis of the block which is never branched to
                    for(auto& phi: branchNotTo->phis()){
                        phi.removeIncomingValue(irb.GetInsertBlock());
                    }

                    // make use of fallthrough
                    if(irb.GetInsertBlock()->getNextNode() == branchTo){
                        return irb.CreateBr(branchTo);
                    }else{
                        // no fallthrough possible
                        auto br = irb.CreateCall(instructionFunctions[ARM_b], {});
                        llvmSetStringMetadata(br, "label", branchTo->getName());
                        irb.CreateBr(branchTo);
                        return br;
                    }
                }

                // cannot be conditional branch, because that always has an icmp NE as its condition, thats matched before
                assert(instr->isUnconditional() && "unconditional branch expected");

                auto bb = instr->getSuccessor(0);

                if(irb.GetInsertBlock()->getNextNode() == bb){
                    return irb.CreateBr(bb);
                }else{
                    auto br = irb.CreateCall(instructionFunctions[ARM_b], {});
                    llvmSetStringMetadata(br, "label", bb->getName());

                    irb.CreateBr(bb);
                    return br;
                }
            },
            llvm::Instruction::Br, 
            {}
        ),

        // icmp (also almost all possible ZExts, they're (almost) exclusively used for icmps, only once for a phi in the short circuiting logical ops)
        Pattern::make_root(
            [](llvm::IRBuilder<>& irb) -> llvm::Value* {
                auto* zextInstr = llvm::dyn_cast<llvm::ZExtInst>(&*irb.GetInsertPoint());
                auto* icmpInstr = llvm::dyn_cast<llvm::ICmpInst>(OP_N(zextInstr, 0));

                auto pred       = icmpInstr->getPredicate();
                auto predStr    = llvmPredicateToARM(pred);

                irb.CreateCall(instructionFunctions[ARM_cmp], {OP_N_MAT(icmpInstr, 0), OP_N_MAT(icmpInstr, 1)});

                auto call = irb.CreateCall(instructionFunctions[ARM_csel], {MAT_CONST(irb.getInt64(1)), XZR});
                llvmSetStringMetadata(call, "pred", predStr);
                return call;
            },
            llvm::Instruction::ZExt,
            {
                {llvm::Instruction::ICmp}
            }
        ),
        // raw icmp without ZExt
        Pattern::make_root(
            [](llvm::IRBuilder<>& irb) -> llvm::Value* {
                auto* icmpInstr = llvm::dyn_cast<llvm::ICmpInst>(&*irb.GetInsertPoint());

                auto pred       = icmpInstr->getPredicate();
                auto predStr    = llvmPredicateToARM(pred);

                irb.CreateCall(instructionFunctions[ARM_cmp], {OP_N_MAT(icmpInstr, 0), OP_N_MAT(icmpInstr, 1)});

                auto call = irb.CreateCall(instructionFunctions[ARM_csel_i1], {MAT_CONST(irb.getInt64(1)), XZR});
                llvmSetStringMetadata(call, "pred", predStr);
                return call;
            },
            llvm::Instruction::ICmp,
            {}
        ),

        // ZExt/PHI: only matched in order to match something for this ZExt, it will become a register later anyway
        Pattern::make_root(
            [](llvm::IRBuilder<>& irb) -> llvm::Value* {
                auto* zextInstr = llvm::dyn_cast<llvm::ZExtInst>(&*irb.GetInsertPoint());
                auto* phiInstr = llvm::dyn_cast<llvm::PHINode>(OP_N(zextInstr, 0)); // noDelete = true -> we can use this as is


                return irb.CreateCall(instructionFunctions[ZExt_handled_in_Reg_Alloc], {phiInstr});
            },
            llvm::Instruction::ZExt,
            {
                {llvm::Instruction::PHI, {}, true}
            }
        ),

        Pattern::make_root(
            [](llvm::IRBuilder<>& irb) -> llvm::Value* {
                auto* call = llvm::dyn_cast<llvm::CallInst>(&*irb.GetInsertPoint());
                for(auto& arg: call->args())
                    if(llvm::isa<llvm::ConstantInt>(arg.get()))
                        call->replaceUsesOfWith(arg.get(), MAT_CONST(arg.get()));
                

                return call;
            },
            llvm::Instruction::Call, // need to materialize the operands
            {}
        ),
    };

    void doISel(llvm::raw_ostream* out){
        // add metadata to instr functions
        for(auto& instr : instructionFunctions){
            instr.second->setMetadata("arm_instruction_function", llvm::MDNode::get(ctx, {llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(i64,std::to_underlying(instr.first)))}));
        }

        for(auto& fn : moduleUP->functions()){
            if(fn.isDeclaration()) continue;

            // Because this inserts new blocks and branches, we need to split the critical edges here.
            // so this has to be done before fallthrough and ARM branches are inserted
            llvm::SplitAllCriticalEdges(fn);


            matchPatterns(&fn, patterns);
        }

        if(out){
            moduleUP->print(*out, nullptr);
            bool moduleIsBroken = llvm::verifyModule(*moduleUP, &llvm::errs());
            if(moduleIsBroken) llvm::errs() << "ISel broke module :(\n";
        }
    }
} // namespace Codegen::ISel

// REFACTOR maybe at some point instruction scheduling

// HW 7: register allocation
// Mostly working ARM register allocation (currently only using X0-X7, but could easily be extended to other caller saved registers), tests included in samples/
// The only remaining bug I know of is that sometimes values are spilled in a block that is not dominated by the block in which the value was defined.
//   By definition, this spill is unnecessary, as the fact that it is spilled in this block means, that its lifetime has already ended here, so a dirty fix would be to simply remove all of these uses, which are not dominated by their definition, in another pass.
// 
// I will probably rewrite this if I find the time and enegery, as I have learned a great deal here about how not to do this (for instance: Not doing liveness analysis was a well-considered tradeoff, that turned out to be not worth making, it just made everything harder and more inefficient)

namespace Codegen::RegAlloc{
    // normal enum, because the int values are important and much more convenient
    // TODO add a way to make the number of used registers modular
    enum Register{
        X0 = 0, X1 = 1, X2 = 2, X3 = 3, X4 = 4, X5 = 5, X6 = 6, X7 = 7,

        // special, not used for normal registers (start at X24 because callee saved)
        X24 = 24, // phi-cycle break
        X25 = 25, // mem-mem stores register
    };
    inline Register operator++(Register& reg, int){
        auto old = reg;
        reg = static_cast<Register>((static_cast<int>(reg)+1) % 8);
        // TODO check this again, is this right now?
        return old;
    }

    Register llvmGetRegisterForArgument(llvm::Argument* arg){
        auto argNum = arg->getArgNo();
        assert(argNum < 8 && "more than 8 args not implemented yet");
        return static_cast<Register>(argNum);
    }

    Register phiCycleBreakRegister = X24;
    Register memMemStoresRegister = X25;

    std::unordered_set<Register> initialFreeRegisters{
        Register::X0,
        Register::X1,
        Register::X2,
        Register::X3,
        Register::X4,
        Register::X5,
        Register::X6,
        Register::X7,
    };

    struct AllocatedRegister{
        llvm::Value* value;
        const Register reg;
    };

    struct AllocatedStackslot{
        llvm::Value* value;
        /// offset from base of spill allocation area, but in (ARM) doublewords, i.e. offset 2 means starting at byte 16
        int offset;
    };

    std::unordered_set<unsigned> skippableTypes{
        llvm::Instruction::Ret,
        llvm::Instruction::Br,
        llvm::Instruction::Unreachable,
        llvm::Instruction::Alloca,
    };

    /// as a macro, because as a function it doesn't work, because setMetadata is protected
#define SET_METADATA(val, reg)                                                               \
        (val)->setMetadata("reg", llvm::MDNode::get(ctx,                                     \
            {llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(i64, static_cast<int>(reg)))} \
        ));

    /// the goal here is to write something that actually works, not to write something that is efficient. Because the thing above is a monstrous abomination and disaster
    /// Currently not used anywhere, because that made an even bigger mess. I have to write this again from scratch if I really want to make it better.
    class StackRegisterAllocator{
    public:
        llvm::ValueMap<llvm::Value*, AllocatedStackslot> spillMap{};

        /// where to insert spill instructions
        llvm::IRBuilder<>& irb;
        llvm::AllocaInst* spillsAllocation;

        StackRegisterAllocator(llvm::IRBuilder<>& irb, llvm::AllocaInst* spillsAllocation) : irb(irb), spillsAllocation(spillsAllocation){}

        bool isAllocated(llvm::Value* val){
            return spillMap.find(val) != spillMap.end();
        }

        /// gets (i.e. regallocs) a particular value (presumed to be an operand of an instruction) to a register, if it is allocated (does nothing otherwise), and the instruction we're working on is not a regalloc generated instruction (that would for example lead to loading the operands of spill stores for function arguments.)
        /// replaces all uses of the value with the register in the given instruction.
        void tryGetAndRewriteInstruction(llvm::Instruction* inst, llvm::Value* val){
            if(!isAllocated(val) || inst->hasMetadata("regAllocIgnore")) return;
            auto [reg, load] = get(val);
            inst->replaceUsesOfWith(val, load);
        }

        /// gets register for value known to be on the stack.
        /// returns register and load instruction.
        /// sets register metadata on the load instruction.
        std::pair<AllocatedRegister, llvm::CallInst*> get(llvm::Value* val){
            assert(isAllocated(val) && "value not on stack");
            auto offset = spillMap[val].offset;
            // offset has to be in doublewords, so multiply by 8 by shifting left by 3
            auto load = irb.CreateCall(instructionFunctions[ARM_ldr], {spillsAllocation, irb.getInt64(offset << 3)}, "l");
            llvmSetEmptyMetadata(load, "regAllocIgnore");
            auto reg = nextRegister();
            SET_METADATA(load, reg); // set the register metadata on the load instruction
            return {AllocatedRegister{val, reg}, load};
        }

        /// allocates a new stackslot for a new value not already on the stack.
        /// annotates value with stackslot metadata.
        /// also immediately spills the value into the stackslot, if spill == true.
        AllocatedStackslot allocate(llvm::Value* val, bool spill = true, llvm::Instruction* insertBefore = nullptr){
            assert(!isAllocated(val) && "value already on stack");


            int offset = spillMap.size();
            spillMap[val] = AllocatedStackslot{val,offset};
            if(spill){
                if(!insertBefore) {
                    assert(llvm::isa<llvm::Instruction>(val) && "if no insertBefore is given, value to spill must be an instruction");
                    insertBefore = llvm::dyn_cast<llvm::Instruction>(val)->getNextNode();
                }
                assert(insertBefore && "cannot insert spill instruction anywhere");

                irb.SetInsertPoint(insertBefore); // insert spill after value

                // if we store it, then it needs a register too
                Register reg;
                if(auto inst = llvm::dyn_cast<llvm::Instruction>(val)){
                    reg = nextRegister();
                    SET_METADATA(inst, reg);
                }else{
                    assert(llvm::isa<llvm::Argument>(val) && "value to spill must be an instruction or an argument");
                    reg = llvmGetRegisterForArgument(llvm::dyn_cast<llvm::Argument>(val));
                }
                auto inst = irb.CreateCall(instructionFunctions[ARM_str], {val, spillsAllocation, irb.getInt64(offset << 3)});
                llvmSetEmptyMetadata(inst, "regAllocIgnore"); // TODO set this everywhere where it's appropriate and actually use it in regalloc
            }
            if(auto inst = llvm::dyn_cast<llvm::Instruction>(val)) {
                inst->setMetadata("stackslot_offset", llvm::MDNode::get(ctx, {llvm::ConstantAsMetadata::get(irb.getInt64(offset))}));
            }
            enlargeSpillsAllocation(1);
            return spillMap[val];
        }

        /// takes care of getting all parameters of a function call into the right register
        AllocatedRegister functionCall(llvm::CallInst* call){
            assert(call->getNumOperands() <= 8 && "too many arguments for function call");
            assert(!call->getCalledFunction()->hasMetadata("arm_instruction_function") && "this is not a real function call!");

            nextReg = X0;
            for(auto& arg: call->args()){
                tryGetAndRewriteInstruction(call, arg);
            }
            nextReg = X0;
            allocate(call);
            return get(call).first;
        }

        void enlargeSpillsAllocation(unsigned by){
            spillsAllocation->setOperand(0,
                llvm::ConstantExpr::getAdd(
                    llvm::dyn_cast<llvm::Constant>(spillsAllocation->getOperand(0)),
                    llvm::ConstantInt::get(
                        spillsAllocation->getOperand(0)->getType(),
                        by // add 1 i64 -> 8 bytes
                    )
                )
            );
        }

        /// none of our instructions have more than 8 operands, so we can simply do this
        Register nextRegister(){
            return nextReg++;
        }

    private:
        Register nextReg = X0;
    };

    void handlePhiChainsCycles(StackRegisterAllocator& allocator, llvm::iterator_range<llvm::BasicBlock::phi_iterator> phiNodes, llvm::DenseSet<llvm::PHINode*>& toEraseLater){
        if(phiNodes.begin() == phiNodes.end()){
            return;
        }

        // allocate a stack slot for each phi
        unsigned phiCounter{0};
        unsigned originalSize = allocator.spillMap.size();
        for(auto& phi : phiNodes){
            allocator.spillMap[&phi] = AllocatedStackslot{&phi, static_cast<int>(originalSize+phiCounter)};
            phiCounter++;
        }
        allocator.enlargeSpillsAllocation(phiCounter);

        llvm::IRBuilder<>& irb  = allocator.irb;

        // per edge:
        unsigned edges = phiNodes.begin()->getNumIncomingValues();
        for(unsigned int edgeNum = 0; edgeNum<edges; edgeNum++){
            llvm::ValueMap<llvm::PHINode*, int> numReaders;
            llvm::ValueMap<llvm::PHINode*, llvm::SmallPtrSet<llvm::PHINode*, 8>> readBy;

            llvm::SmallPtrSet<llvm::PHINode*, 8> toHandle;

            for(auto& phi : phiNodes){
                auto val = phi.getIncomingValue(edgeNum);
                auto otherPhi = llvm::dyn_cast_if_present<llvm::PHINode>(val);

                if(otherPhi && otherPhi==&phi){
                    // self-reference/edge, ignore
                    continue;
                }
                toHandle.insert(&phi);

                if(otherPhi && otherPhi->getParent() == phi.getParent()){
                    readBy[otherPhi].insert(&phi);
                    numReaders[otherPhi]++;
                }
            }

            auto insertBefore = phiNodes.begin()->getIncomingBlock(edgeNum)->getTerminator(); // insert stores to phi nodes at the end of the predecessor block (has been broken to take care of crit. edges)
            // because isel the terminators are of course not the last instructions anymore, so correct this so it inserts before the actual branches
            // REFACTOR: this looks terrible

            // We go to the previous one if the previous one is an ARM_cmp, or an ARM_b_xx

            while(insertBefore->getPrevNode() != nullptr){
                auto prevMaybeCall = llvm::dyn_cast_if_present<llvm::CallInst>(insertBefore->getPrevNode());

                if(prevMaybeCall && (
                    prevMaybeCall->getCalledFunction() == instructionFunctions[ARM_b] ||
                    prevMaybeCall->getCalledFunction() == instructionFunctions[ARM_b_cond] ||
                    prevMaybeCall->getCalledFunction() == instructionFunctions[ARM_b_cbnz] ||
                    prevMaybeCall->getCalledFunction() == instructionFunctions[ARM_b_cbz] ||
                    prevMaybeCall->getCalledFunction() == instructionFunctions[ARM_cmp]
                )){
                    insertBefore = insertBefore->getPrevNode();
                }else{
                    break;
                }
            }

            allocator.irb.SetInsertPoint(insertBefore);

            // start at phis with 0 readers

            auto handleChainElement = [&](llvm::PHINode* phi){
                // (pseudo-)store to stack, needs to be looked at by regalloc again later on, possibly to insert load for incoming value, if its not in a register
                irb.CreateCall(
                    instructionFunctions[ARM_PSEUDO_str],
                    // TODO have to copy the value here, as it is deleted later, and if its a constant (-> owned by the phi), it will be deleted later when we delete the phis -> not stonks
                    {phi->getIncomingValue(edgeNum), allocator.spillsAllocation, allocator.irb.getInt64(allocator.spillMap[phi].offset << 3)}
                );
                toHandle.erase(phi);
                for(auto reader: readBy[phi]){
                    numReaders[reader]--;
                }
            };

            unsigned notChangedCounter = 0;
            while(toHandle.size()>0 && notChangedCounter < toHandle.size()){
                auto phi = *toHandle.begin();
                notChangedCounter++;

                if(numReaders[phi] == 0){
                    handleChainElement(phi);

                    notChangedCounter = 0;
                }
            }

            // prevent cycles of length 1 (self edges) by handling them before by not adding readBy/readers entries for them
            if(toHandle.size() > 0){

                // cycle
                // temporarily save one of the phi nodes in the special register
                auto phi = *toHandle.begin();
                auto load = allocator.irb.CreateCall(instructionFunctions[ARM_ldr], {allocator.spillsAllocation, allocator.irb.getInt64(allocator.spillMap[phi].offset << 3)}, "phiCycleBreak");
                auto firstStore = load;
                numReaders[phi]--;
                SET_METADATA(load, phiCycleBreakRegister);

                // handle chain until last element
                while(toHandle.size()>1){
                    phi = *toHandle.begin();

                    if(numReaders[phi] == 0){
                        handleChainElement(phi);
                    }
                }

                // last element gets assigned the special register
                phi = *toHandle.begin();
                allocator.irb.CreateCall(instructionFunctions[ARM_str], {firstStore, allocator.spillsAllocation, allocator.irb.getInt64(allocator.spillMap[phi].offset << 3)});
            }
        }

        // replace all phis with loads from the stack
        // NEW: remove phis, they will be replaced by loads later on
        for(auto& phi : llvm::make_early_inc_range(phiNodes)){
            //allocator.irb.SetInsertPoint(phi.getParent()->getFirstNonPHI());
            //auto load = allocator.irb.CreateCall(instructionFunctions[ARM_ldr], {allocator.spillsAllocation, allocator.irb.getInt64(allocator.spillMap[&phi].offset << 3)}, "phi_" + phi.getName());
            //load->setMetadata("phi", llvm::MDNode::get(ctx, {}));
            //load->setMetadata("noSpill", llvm::MDNode::get(ctx, {}));
            //phi.replaceAllUsesWith(load);
            //phi.eraseFromParent();
            phi.removeFromParent();
            // erase later on using this set:
            toEraseLater.insert(&phi);
        }
    }

    void regallocFn(llvm::Function& f){

        llvm::IRBuilder<> irb(f.getEntryBlock().getFirstNonPHI());
        auto spillsAllocation = irb.CreateAlloca(irb.getInt64Ty(), irb.getInt64(0), "spills");
        StackRegisterAllocator allocator{irb, spillsAllocation};

        // first handle phi node problems, by allocating stack slots for them, and inserting pseudo instructions to load/store them, which will get filled in by the register allocator
        // REFACTOR: To improve the performance of the allocator as whole, we could try to distinguish between register and normal phis, because phis are usually so important, that giving them their own register is worth it
        // the vague strategy would then be: replace the phis with registers, with some heuristic of what percentage of registers they can take up, make a map of bb to phi-occupied-registers, so they are not reallocated (this would be somewhat cumbersome, because it would need to include every BB on the CFG in between the definition of the value used by the phi and the phi itself), and then run the register allocator for the remaining registers on the BBs, taking into account which can still be used

        llvm::DenseSet<llvm::PHINode*> phisToEraseLater{};
        for(auto& bb: f){
            handlePhiChainsCycles(allocator, bb.phis(), phisToEraseLater);
        }

        if(f.arg_size() > 8){
            EXIT_TODO_X("more than 8 arguments not supported yet");
        }
        
        // assign registers from X0 up to X7 to the arguments
        auto insertBefore = f.getEntryBlock().getFirstNonPHI(); // no PHIs in entry, so this is just get first
                                                                // this is wrong, this inserts before allocas -> move to first non-alloca
        while(auto alloca = llvm::dyn_cast_if_present<llvm::AllocaInst>(insertBefore)){
            insertBefore = alloca->getNextNode();
        }
        for(auto& param: f.args()){
            allocator.allocate(&param, true, insertBefore);
            // TODO this doesn't work because the store that is generated to allocate this is analyzed by the reg allocator later on again. We don't want that. Easiest way to solve it is probably to set metadata on instructions generated by the register allocator, so they are ignored?
            // hmmm is there some way to make this more general? feel like the regallocer is so difficult because it doesn't really know what to touch and what not to touch... maybe actually regalloc by placing all instructions that should be looked at in a queue during isel and ignoring all others?
            // cannot set metadata on function parameters, so this has to be implicit
        }

        // i hope this encounters stuff in the correct order
        // I think it does, *except for the arguments of phi nodes*! -> either handle phis before everything else and insert a pseudo reference, to a value that is not yet defined (but will be defined in the same block by the register allocator later)
        for(auto& bb: f){
            for(auto& inst: llvm::make_early_inc_range(bb)){
                if(auto call = llvm::dyn_cast_if_present<llvm::CallInst>(&inst)){
                    // pseudo stores for phis
                    if (call->getCalledFunction() == instructionFunctions[ARM_PSEUDO_str]) {
                        // pseudo store, we need to possibly insert load for mem-mem mov
                        allocator.irb.SetInsertPoint(call);

                        auto val = call->getArgOperand(0);
                        
                        // materialize constant
                        llvm::Value* newVal = val;
                        if(auto constInt = llvm::dyn_cast<llvm::ConstantInt>(val) ) {
                            if(!constInt->isZero()){
                                auto call = irb.CreateCall(instructionFunctions[ARM_mov], {irb.getInt64(constInt->getZExtValue())});
                                newVal = call;

                                allocator.allocate(call);
                                //auto reg = allocator.get(call).first.reg;
                                //SET_METADATA(call, reg);
                            }else {
                                newVal = XZR;
                            }
                            bool isntStillConst = !llvm::isa<llvm::ConstantInt>(newVal) || newVal == XZR;
                            (void) isntStillConst;
                            assert(isntStillConst && "constant should have been materialized");

                            call->replaceUsesOfWith(val, newVal);
                        }else{
                            allocator.tryGetAndRewriteInstruction(call, val);
                        }

                        call->setCalledFunction(instructionFunctions[ARM_str]);
                    }else if(call->hasMetadata("reg")){
                        continue;
                    }else if(call->getCalledFunction() == instructionFunctions[ZExt_handled_in_Reg_Alloc]){
                        inst.replaceAllUsesWith(inst.getOperand(0));
                        inst.eraseFromParent();
                    }else if(!call->getCalledFunction()->hasMetadata("arm_instruction_function")){
                        allocator.functionCall(call);
                    }else{
                        allocator.irb.SetInsertPoint(call);


                        // loads for phis dont need seperate loads for their arguments, those are already correct
                        if(!call->hasMetadata("phi")){
                            for(auto& arg: call->args()){
                                // only handle arguments which are themselves reg-allocated at all, and are not phi nodes (these are only ever directly written to the stack, never spill stored)
                                // the problem here is, this first condition excludes function parameters, thats what the last and is for
                                // TODO wait with the new allocation scheme this doesn't make any sense, we should just load it, right?
                                //if((!llvm::isa<llvm::CallInst>(arg) || !llvm::dyn_cast<llvm::CallInst>(arg)->hasMetadata("reg")) && !llvm::isa<llvm::Argument>(arg)){
                                //    continue;
                                //}

                                allocator.tryGetAndRewriteInstruction(call, arg);
                            }
                        }

                        // TODO think about this again
                        if(allocator.isAllocated(call)){
                            if (!call->hasMetadata("reg"))
                                allocator.get(call);
                        }else
                            // cmp, str, etc. doesnt need an output register (cmp is not void, because the branches use it)
                            if(call->getCalledFunction()->getReturnType()!=voidTy && call->getCalledFunction() != instructionFunctions[ARM_cmp] ){
                                allocator.allocate(call);
                        }
                    }
                } else if(auto alloca = llvm::dyn_cast_if_present<llvm::AllocaInst>(&inst)){
                    // annotate it with noSpill (meaning if it is used as an arugment somewhere, it does not get spilled)
                    // TODO with the new allocation scheme, this doesn't make sense anymore, right?
                    alloca->setMetadata("noSpill", llvm::MDNode::get(ctx, {}));
                } else if(auto returnInst = llvm::dyn_cast_if_present<llvm::ReturnInst>(&inst)){
                    if(auto arg = returnInst->getReturnValue()){
                        irb.SetInsertPoint(returnInst);
                        allocator.tryGetAndRewriteInstruction(returnInst, arg);
                    }
                } else {
                    bool isIgnored = (llvm::isa<llvm::BranchInst>(inst) || llvm::isa<llvm::UnreachableInst>(inst));
                    (void)isIgnored;
                    assert(isIgnored && "unhandled instruction type");
                }
            }
        }

        // delete all the phis, they've been handled now (they are already removed from their parent)
        for(auto phi: phisToEraseLater){

#ifndef NDEBUG
            if(!phi->use_empty())
                for(auto& use: phi->uses())
                    assert(llvm::isa<llvm::PHINode>(use.get()) && "phi which is about to be deleted should only be used by other phis (so effectively no uses anymore)");
#endif
            phi->deleteValue();
        }
    }

    void doRegAlloc(llvm::raw_ostream* out){
        for(auto& f : *moduleUP){
            if(f.isDeclaration()){
                continue;
            }

            regallocFn(f);
        }

        if(out){
            moduleUP->print(*out, nullptr);
            bool moduleIsBroken = llvm::verifyModule(*moduleUP, &llvm::errs());
            if(moduleIsBroken) llvm::errs() << "RegAlloc broke module :(\n";
        }
    }

    
} // namespace Codegen::RegAlloc

namespace Codegen{

    /*
       HW 9 START:
       - the provided programs samples/addressCalculations.b (echoes argv[1] while demonstrating nested pointer arithmetic) and samples/simplemain.b (prints 0-9) should both work, and hopefully demonstrate that some basic stuff does work
       - most of the problems I've encountered were with my terrible register allocation :(. Some of it is fixed, but I imagine a lot isn't.
       - use `./main -a samples/addressCalculations.b 2>/dev/null | aarch64-linux-gnu-gcc -g -x assembler -o test - && qemu-aarch64 -L /usr/aarch64-linux-gnu test hi\ there` to test it out using qemu user emulation. Beware of differences in the elf interpreter path (qemu-aarch64 -L ...) on your system.
    */

    class AssemblyGen{
        llvm::raw_ostream& out;

        /// stores the offset from the framepointer for any alloca (resets with every call to emitFunction)
        /// example: first alloca, only allocates 8 bytes, so it's at fp - 8, so we store -8 here
        llvm::ValueMap<llvm::AllocaInst*, int> framePointerRelativeAddresses{};

        llvm::Function* currentFunction = nullptr;

    public:
        AssemblyGen(llvm::raw_ostream* out): out(*out) {}



        static string llvmNameToBlockLabel(llvm::Function* fn, llvm::StringRef name){
            return ".L" + fn->getName().str() + "_" + name.str();
        }

        static string llvmNameToBlockLabel(llvm::Function* fn, llvm::CallInst* call){
            return llvmNameToBlockLabel(fn, llvmGetStringMetadata(call, "label"));
        }

        /**
          OLD Stack layout (not used anymore):
          |______...______|<- sp at start/CFA
          |_link register_|
          |_old frame ptr_|<- fp
          |__local  vars__|
          |__local  vars__|
          |__local  vars__|
          |______...______|

          -> lr (x29) is at CFA - 8
          -> fp (x30) is at CFA - 16
          -> localvars[0] is at fp - 8 or CFA - 24
          -> localvars[1] is at fp - 16 or CFA - 32
          ...

          NEW Stack layout:

          |______...______|<- sp at start/CFA
          |_link register_|
          |_old frame ptr_|
          |__local  vars__|
          |__local  vars__|
          |__local  vars__|<- fp
          |______...______|
        */
        void emitPrologue(llvm::Function* f){
            out << "\t.global " << f->getName() << "\n";
            out << "\t.type " << f->getName() << ", %function\n"; // arm asm seems to be %function, not @function
            out << f->getName() << ":\n";
            out << "\t.cfi_startproc\n";

            out << "\tstp fp, lr, [sp, -16]!\n"; // still 16 byte aligned
            // now sp is where we want fp to be
            out << "\tmov fp, sp\n";
            out << "\t.cfi_def_cfa fp, 16\n"; // x29 = fp, offset +16 to get to CFA
            out << "\t.cfi_offset 29, -16\n"; // x29 = fp, offset -16 to get to fp
            out << "\t.cfi_offset 30, -8\n";  // x30 = lr, offset -8  to get to lr

            // fp currently points to the old frame ptr, we haven't allocated any local vars yet
        }

        void emitEpilogue(llvm::Function* f){
            out << "\t.size " << f->getName()  << ", .-" << f->getName() << "\n";
            out << "\t.cfi_endproc\n\n";
        }

        template<int bitwidth>
        static const char regPrefix; // 'X' or 'W'

        /// emits a 32/64 bit register (Wn/Xn) using the metadata "reg n" on the instruction
        template<int bitwidth = 64>
        void emitReg(llvm::CallInst* call){
            static_assert(bitwidth == 32 || bitwidth == 64, "bitwidth must be 32 or 64");

            assert(call->hasMetadata("reg") && "instruction has no reg metadata");
            int regNum = dyn_cast<llvm::ConstantAsMetadata>(call->getMetadata("reg")->getOperand(0))->getValue()->getUniqueInteger().getZExtValue();
            // i hope the compiler optimizes this in the 2 generated cases, but i don't see why it wouldn't
            out << regPrefix<bitwidth> << regNum;
        }

        /// emits XZR/WZR, an immediate, or a 32/64 bit register (Wn/Xn) using the metadata "reg n" on the instruction
        /// -> emits the destination register if called on a value
        /// -> emits the source register if called on an operand
        template<int bitwidth = 64>
        void emitReg(llvm::Value* v){
            static_assert(bitwidth == 32 || bitwidth == 64, "bitwidth must be 32 or 64");

            // args can be:
            // - constants (XZR)
            // - CallInsts -> have registers
            // - parameters -> have registers
            // - poison -> warn about it

            if(v == XZR){
                out << regPrefix<bitwidth> << "ZR";
            }else if(auto constInt = llvm::dyn_cast<llvm::ConstantInt>(v)){
                out << constInt->getZExtValue();
            }else if(auto param = llvm::dyn_cast<llvm::Argument>(v)){
                out << regPrefix<bitwidth> << RegAlloc::llvmGetRegisterForArgument(param);
            }else if(auto call = llvm::dyn_cast<llvm::CallInst>(v)){
                // TODO problem: this is called with allocas as args, we have to translate those to fp-relative addresses, and materialize them
                // all of the others are handled in ptrtoint, but this cannot be done for phi nodes, as they use the ptr directly and store it somewhere (in the ARM_PSEUDO_str)
                // the alternative would be to not do phis for ptrs (which should hopefully have happend by now, but check this again)
                // TODO factor out this code to a helper "assert is type" or smth
                emitReg<bitwidth>(call);
            }else if(auto poison = llvm::dyn_cast<llvm::PoisonValue>(v)){
                // this means there is undefined behavior somewhere in the codegen, for example a division by 0, shift by >=64 etc.
                warn("poison value encountered, substituting zero. This means there is undefined behavior in the source, please check for any divisions by zero, shifts by >=64, etc.", poison);
                out << regPrefix<bitwidth> << "ZR";
            }else{
                DEBUGLOG("v: " << *v);
                assert(false && "unhandled value type");
            }
        }

        /// emits the operands of an instruction (including the register assigned to the call as destination register), separated by commas, howMany indicates how many operands to emit at max, 0 means all
        void emitDestAndOps(llvm::CallInst* call, unsigned howMany = 0){
            emitReg(call);

            if(howMany == 0 || howMany > call->arg_size()) howMany = call->arg_size();
            for(unsigned i = 0; i < howMany; i++){
                auto arg = call->getArgOperand(i);
                out << ", ";
                emitReg(arg);
            }
        }

        void emitInstruction(llvm::CallInst* call){
            auto calledFn = call->getCalledFunction();
            if(calledFn->hasMetadata("arm_instruction_function")){
                // 'ARMInstruction's which need special handling:
                // - add_SP
                // - add_SHIFT
                // - sub_SP
                // - sub_SHIFT
                // - csel_i1
                // - csel
                // - lsl_imm
                // - asr_imm
                // - lsl_var
                // - asr_var
                // - ldr
                // - ldr_sb
                // - ldr_sh
                // - ldr_sw
                // - str
                // - str32_b
                // - str32_h
                // - str32
                // - b
                // - b_cond
                // - b_cbnz
                // - b_cbz
                // - PSEUDO_addr_computation
                // - cmp (no destination)
#define CASE(ARMInstruction) (calledFn==instructionFunctions[ARMInstruction])

                out << "\t";

                // handle all immediate operands in these special cases
                if(CASE(ARM_add_SP)){
                    out << "add sp, sp, " << call->getArgOperand(0)->getName();
                }else if (CASE(ARM_add_SHIFT)){
                    out << "add ";
                    emitDestAndOps(call, 2);
                    out << ", ";
                    // immediate
                    // TODO is this even right
                    // (if I correct it here, also correct it for sub)
                    out << "lsl " << llvm::dyn_cast<llvm::ConstantInt>(call->getArgOperand(2))->getZExtValue();
                }else if (CASE(ARM_sub_SP)){
                    out << "sub sp, sp, " << call->getArgOperand(0)->getName();
                }else if (CASE(ARM_sub_SHIFT)){
                    out << "sub ";
                    emitDestAndOps(call, 2);
                    out << ", ";
                    // immediate
                    out << "lsl " << llvm::dyn_cast<llvm::ConstantInt>(call->getArgOperand(2))->getZExtValue();
                }else if (CASE(ARM_csel_i1) || CASE(ARM_csel)){
                    out << "csel ";
                    emitDestAndOps(call, 2);
                    out << ", ";
                    out << llvmGetStringMetadata(call, "pred");
                }else if(CASE(ARM_lsl_imm)){
                    out << "lsl ";
                    emitDestAndOps(call, 1);
                    out << ", ";
                    out << llvm::dyn_cast<llvm::ConstantInt>(call->getArgOperand(1))->getZExtValue();
                }else if(CASE(ARM_asr_imm)){
                    out << "asr ";
                    emitDestAndOps(call, 1);
                    out << ", ";
                    out << llvm::dyn_cast<llvm::ConstantInt>(call->getArgOperand(1))->getZExtValue();
                }else if(CASE(ARM_lsl_var)){
                    out << "lsl ";
                    emitDestAndOps(call, 2);
                }else if(CASE(ARM_asr_var)){
                    out << "asr ";
                    emitDestAndOps(call, 2);
                }else if(CASE(ARM_ldr)){
                    // TODO this load and store thing is *hella* stupid. It does work, but just add some kind of address struct to the instruction which handles this nicely, and get rid of the cases (i.e. transfer them into regalloc or isel) in which we need another register and thus adjust fp

                    // now distinguish between a few cases:
                    // - simple ldr from "normal" alloca (ldr call has only alloca as operand)
                    // - ldr has immediate offset (ldr call has alloca and offset as operands)
                    // - ldr has offset-register and shift (ldr call has alloca, offset, shift as operands)
                    // - ldr comes from inttoptr (ldr call has inttoptr, offset, shift as operands) 

                    // if alloca -> one of the first 3 cases
                    if(auto alloca = llvm::dyn_cast<llvm::AllocaInst>(call->getArgOperand(0))){
                        if(call->arg_size() == 1){
                            out << "ldr ";
                            emitReg(call);
                            out << ", [fp, " << framePointerRelativeAddresses[alloca] << "]";
                        }else if(call->arg_size() == 2){
                            out << "ldr ";
                            emitReg(call);
                            out << ", [fp, " << (framePointerRelativeAddresses[alloca] + llvm::dyn_cast<llvm::ConstantInt>(call->getOperand(1))->getSExtValue()) << "]";
                        }else if(call->arg_size() == 3){
                            // bit of a hack to not use more registers for the address calculation here:
                            // addr = fp - framePointerRelativeAddresses[alloca] + (offset-register << shift)
                            // the problem is that we cant have the fp - framePointerRelativeAddresses[alloca] in one instruction
                            // so calculate that part in fp itself
                            out << "add fp, fp, " << framePointerRelativeAddresses[alloca] << "\n";
                            out << "\t.cfi_adjust_cfa_offset " << framePointerRelativeAddresses[alloca] << "\n";

                            out << "\tldr ";
                            emitReg(call);
                            out << ", [fp, ";
                            emitReg(call->getOperand(1));
                            out << ", lsl ";
                            out << llvm::dyn_cast<llvm::ConstantInt>(call->getOperand(2))->getZExtValue() << "]\n";

                            out << "\tsub fp, fp, " << framePointerRelativeAddresses[alloca];
                            out << "\t.cfi_adjust_cfa_offset " << -framePointerRelativeAddresses[alloca] << "\n";
                        }
                    }else{
                        // in this case its either an inttoptr operand, or a "raw" load from any address

                        if(call->arg_size() == 1){
                            // raw load
                            out << "ldr ";
                            emitReg(call);
                            out << ", [";
                            emitReg(call->getOperand(0));
                            out << "]";
                        }else{
                            assert(call->arg_size() == 3 && "expected inttoptr ldr");
                            // inttoptr operand, i.e. an arbitrary expression in a register
                            // -> address is in register, use that as base, then offset register and shift
                            out << "ldr ";
                            emitReg(call);
                            out << ", [";
                            emitReg(call->getOperand(0));
                            out << ", ";
                            emitReg(call->getOperand(1));
                            out << ", lsl " << llvm::dyn_cast<llvm::ConstantInt>(call->getOperand(2))->getZExtValue() << "]";
                        }
                    }
                }else if(CASE(ARM_ldr_sb) || CASE(ARM_ldr_sh) || CASE(ARM_ldr_sw)){
                    // in this case its an inttoptr operand, i.e. an arbitrary expression in a register
                    // -> address is in register, use that as base, then offset register and shift

                    out << "ldr"; 
                    if(CASE(ARM_ldr_sb)){
                        out << "sb ";
                    }else if(CASE(ARM_ldr_sh)){
                        out << "sh ";
                    }else if(CASE(ARM_ldr_sw)){
                        out << "sw ";
                    }
                    emitReg(call);
                    out << ", [";
                    emitReg(call->getOperand(0));
                    out << ", ";
                    emitReg(call->getOperand(1));
                    out << ", lsl " << llvm::dyn_cast<llvm::ConstantInt>(call->getOperand(2))->getZExtValue() << "]";
                }else if(CASE(ARM_str)){
                    // now distinguish between a few cases:
                    // - simple str to "normal" alloca (str call has what to store, and alloca as operand)
                    // - str has immediate offset (str call has what to store, alloca and offset as operands)
                    // - str has offset-register and shift (str call has what to store, alloca, offset, shift as operands)
                    // - str comes from inttoptr (str call has what to store, inttoptr, offset, shift as operands) 

                    // if alloca -> one of the first 3 cases
                    if(auto alloca = llvm::dyn_cast<llvm::AllocaInst>(call->getArgOperand(1))){
                        if(call->arg_size() == 2){
                            out << "str ";
                            emitReg(call->getOperand(0));
                            out << ", [fp, " << framePointerRelativeAddresses[alloca] << "]";
                        }else if(call->arg_size() == 3){
                            out << "str ";
                            emitReg(call->getOperand(0));
                            out << ", [fp, " << (framePointerRelativeAddresses[alloca] + llvm::dyn_cast<llvm::ConstantInt>(call->getOperand(2))->getSExtValue()) << "]";
                        }else if(call->arg_size() == 4){
                            // TODO does this still work for the new stack layout? (also for loads)
                            // bit of a hack to not use more registers for the address calculation here:
                            // addr = fp - framePointerRelativeAddresses[alloca] + (offset-register << shift)
                            // the problem is that we cant have the fp - framePointerRelativeAddresses[alloca] in one instruction
                            // so calculate that part in fp itself
                            out << "add fp, fp, " << framePointerRelativeAddresses[alloca] << "\n";
                            out << "\t.cfi_adjust_cfa_offset " << framePointerRelativeAddresses[alloca] << "\n";

                            out << "\tstr ";
                            emitReg(call->getOperand(0));
                            out << ", [fp, ";
                            emitReg(call->getOperand(2));
                            out << ", lsl ";
                            out << llvm::dyn_cast<llvm::ConstantInt>(call->getOperand(3))->getZExtValue() << "]\n";

                            out << "\tsub fp, fp, " << framePointerRelativeAddresses[alloca];
                            out << "\t.cfi_adjust_cfa_offset " << -framePointerRelativeAddresses[alloca] << "\n";
                        }
                    }else{
                        // in this case its either an inttoptr operand, or a "raw" store from any address
                        if(call->arg_size() == 2){
                            // raw str
                            out << "str ";
                            emitReg(call->getOperand(0));
                            out << ", [";
                            emitReg(call->getOperand(1));
                            out << "]";
                        } else{
                            assert(call->arg_size() == 4 && "expected inttoptr str");

                            // inttoptr operand, i.e. an arbitrary expression in a register
                            // -> address is in register, use that as base, then offset register and shift
                            out << "str ";
                            emitReg(call->getOperand(0));
                            out << ", [";
                            emitReg(call->getOperand(1));
                            out << ", ";
                            emitReg(call->getOperand(2));
                            out << ", lsl " << llvm::dyn_cast<llvm::ConstantInt>(call->getOperand(3))->getZExtValue() << "]";
                        }
                    }
                }else if(CASE(ARM_str32) || CASE(ARM_str32_h) || CASE(ARM_str32_b)){
                    // in this case its an inttoptr operand, i.e. an arbitrary expression in a register
                    // -> address is in register, use that as base, then offset register and shift

                    out << "str"; 
                    if(CASE(ARM_str32_h)){
                        out << "h ";
                    }else if(CASE(ARM_str32_b)){
                        out << "b ";
                    }else if(CASE(ARM_str32)){
                        out << " ";
                    }
                    emitReg<32>(call->getOperand(0));
                    out << ", [";
                    emitReg<64>(call->getOperand(1));
                    out << ", ";
                    emitReg<64>(call->getOperand(2));
                    out << ", lsl " << llvm::dyn_cast<llvm::ConstantInt>(call->getOperand(3))->getZExtValue() << "]";
                }else if(CASE(ARM_b)){
                    // fallthrough has been handled in ISel by not creating branch calls in that case
                    out << "b " << llvmNameToBlockLabel(currentFunction, call);
                }else if(CASE(ARM_b_cond)){
                    out << "b." << llvmGetStringMetadata(call, "pred") << " " << llvmNameToBlockLabel(currentFunction, call);
                }else if(CASE(ARM_b_cbnz)){
                    out << "cbnz ";
                    emitReg(call->getOperand(0));
                    out << ", " << llvmNameToBlockLabel(currentFunction, call);
                }else if(CASE(ARM_b_cbz)){
                    out << "cbz ";
                    emitReg(call->getOperand(0));
                    out << ", " << llvmNameToBlockLabel(currentFunction, call);
                }else if(CASE(ARM_PSEUDO_addr_computation)){
                    // arg is always an alloca
                    assert(llvm::isa<llvm::AllocaInst>(call->getArgOperand(0)) && "expected alloca as arg for addr computation");
                    out << "add ";
                    emitReg(call);
                    out << ", fp, " << framePointerRelativeAddresses[llvm::dyn_cast<llvm::AllocaInst>(call->getArgOperand(0))];
                }else if(CASE(ARM_cmp)){
                    out << "cmp ";
                    emitReg(call->getOperand(0));
                    out << ", ";
                    emitReg(call->getOperand(1));
                }else{
                    // standard case for instructions which don't need modifying
                    out << call->getCalledFunction()->getName().substr(4) << " ";

                    emitReg(call);
                    for(auto& arg: call->args()){
                        out << ", ";
                        auto argVal = arg.get();
                        emitReg(argVal);
                    }
                }
                out << "\n";
            }else{
                // call arguments have already been moved to the correct registers, so discard them
                out << "\tbl " << call->getCalledFunction()->getName() << "\n";
            }
        }

        void emitFunction(llvm::Function* f){
            // TODO frame pointer chaining 
            currentFunction = f;

            emitPrologue(f);
            // fp points to old fp currently

            framePointerRelativeAddresses.clear();
            unsigned stackAllocation{0};

            // handle allocas
            for(auto& inst: f->getEntryBlock()){
                if(auto alloca = llvm::dyn_cast_if_present<llvm::AllocaInst>(&inst)){
                    // add bitwidth in bytes (* allocation size if applicable) to stack allocation
                    int typeBitWidth = alloca->getAllocatedType()->getIntegerBitWidth();
                    int allocationSize = 1*typeBitWidth/8;
                    if (alloca->getNumOperands() > 0) allocationSize *= dyn_cast<llvm::ConstantInt>(alloca->getOperand(0))->getZExtValue();

                    stackAllocation += allocationSize;
                    framePointerRelativeAddresses[alloca] = - static_cast<int>(stackAllocation); // we have subtracted the previous stack allocation as well as the new allocation, bringing us to the point from which the current allocation can be addressed. THIS IS ALSO NOT CORRECT YET! this currently assumes the fp pointing to the old fp. We add the stackframe size later, so the addressing is from the bottom of the fixedsize stack frame
                }
            }
            if(stackAllocation>0){
                if(stackAllocation%16!=0) stackAllocation += 16 - stackAllocation%16;
                out << "\tsub sp, sp, " << stackAllocation << "\n";
                out << "\tmov fp, sp " << "\n"; // fp now points to bottom of fixed size stack frame

                // for all frame pointer relative addresses, add the size of the allocation, so that they point to the correct (starting from the bottom of the fixed size stack frame)
                // this is done to allow a bigger addressing range
                for(auto [alloca, offset]: framePointerRelativeAddresses){
                    framePointerRelativeAddresses[alloca] += stackAllocation;
                }

                // now correct cfi directives
                out << "\t.cfi_def_cfa fp, " << stackAllocation+16 << "\n"; // +stack frame size to get to old fp, +16 to get from old fp to CFA
            }

            for(auto& bb: *f){
                if(bb.hasNPredecessors(0) && !bb.isEntryBlock()) continue; // skip unreachable blocks (they haven't been reg allocated anyway)

                out << llvmNameToBlockLabel(currentFunction, bb.getName()) << ":\n";
                for(auto& inst: bb){
                    if(auto call = llvm::dyn_cast_if_present<llvm::CallInst>(&inst)){
                        emitInstruction(call);
                    }else if(auto ret = llvm::dyn_cast_if_present<llvm::ReturnInst>(&inst)){
                        if(ret->getNumOperands()>0){
                            out << "\tmov X0, ";
                            emitReg(ret->getOperand(0));
                            out << "\n";
                        }
                        out << "\tadd sp, fp, " << stackAllocation <<"\n"; // deallocate stackframe (fp+stackAllocation gets back to old fp)
                        out << "\tldp fp, lr, [sp], 16\n";
                        // TODO save restore cfi here
                        out << "\t.cfi_restore 30\n";
                        out << "\t.cfi_restore 29\n";
                        out << "\t.cfi_def_cfa sp, 0\n";
                        out << "\tret\n";

                    }else if(llvm::isa<llvm::BranchInst>(&inst) || llvm::isa<llvm::AllocaInst>(&inst) || llvm::isa<llvm::UnreachableInst>(&inst)){
                        // branches are simply ignored, they are explicitly converted to branch arm instructions in ISel
                        // allocas are ignored, they are handled in the prologue
                        // unreachables are ignored, because what else should we do?
                    }else{
                        llvm::errs() << "Unhandled instruction in asm writing: " << inst << "\n";
                        exit(ExitCode::ERROR_CODEGEN);
                    }
                }
            }

            emitEpilogue(f);
        }

        void doAsmOutput(){
            for(auto& f: *moduleUP){
                if(f.isDeclaration()) continue;
                emitFunction(&f);
            }
        }

    };

    template<>
    char AssemblyGen::regPrefix<64> = 'X';

    template<>
    char AssemblyGen::regPrefix<32> = 'W';

} // namespace Codegen::Asm

template<typename T>
concept char_ptr = requires(T t){
    {t[0]} -> std::convertible_to<const char>;
};

int execute(const std::string& command, char_ptr auto&&... commandArgs){
    pid_t pid;

    auto convToCharPtr = [](auto& arg) -> const char *{
        if constexpr(std::is_same_v<std::decay_t<decltype(arg)>, std::string>)
            return arg.c_str();
        else 
            return arg;
    };

    if((pid=fork())==0){
        // child
        const char *const args[] = {command.c_str(), convToCharPtr(commandArgs)..., nullptr};
        execvp(command.c_str(), const_cast<char *const *>(args));
        llvm::errs() << "Failed to execvp " << command.c_str() << ": " << strerror(errno) << "\n";
        return ExitCode::ERROR_IO | ExitCode::ERROR_LINK;
    }

    // parent
    int wstatus;
    wait(&wstatus);
    if(!WIFEXITED(wstatus)){
        llvm::errs() << command << " did not exit normally\n";
        return ExitCode::ERROR_LINK;
    }
    return ExitCode::SUCCESS;
}

int main(int argc, char *argv[]) {
    // TODO check https://www.llvm.org/docs/Frontend/PerformanceTips.html at some point
#define MEASURE_TIME_START(point) auto point ## _start = std::chrono::high_resolution_clock::now()

#define MEASURE_TIME_END(point) auto point ## _end = std::chrono::high_resolution_clock::now()

#define MEASURED_TIME_AS_SECONDS(point, iterations) std::chrono::duration_cast<std::chrono::duration<double>>(point ## _end - point ## _start).count()/(static_cast<double>(iterations))

    Codegen::initInstructionFunctions();
    auto parsedArgs = ArgParse::parse(argc, argv);

    if(ArgParse::args.help()){
        ArgParse::printHelp(argv[0]);
        return ExitCode::SUCCESS;
    }

    std::string inputFilename = *ArgParse::args.input;
    int iterations = 1;
    if(ArgParse::args.iterations()){
        iterations = std::stoi(*ArgParse::args.iterations);
    }

    if(access(inputFilename.c_str(),R_OK) != 0){
        err(ExitCode::ERROR_IO, "Could not open input file %s", inputFilename.c_str());
    }

    std::ifstream inputFile{inputFilename};
    std::string preprocessedFilePath;

    auto epochsecs = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count(); //cpp moment
    if(ArgParse::args.preprocess()){
        preprocessedFilePath="/tmp/" + std::to_string(epochsecs) + ".bpreprocessed";
        execute("cpp", "-E", "-P", *ArgParse::args.input, "-o", preprocessedFilePath);

        inputFile = std::ifstream{preprocessedFilePath};
    }

    Parser parser{inputFile};

    MEASURE_TIME_START(parse);
    unique_ptr<AST> ast;
    for(int i = 0; i<iterations; i++){
        ast = parser.parse();
        assert(ast->validate() && "the parser should always produce a valid AST");
        parser.resetTokenizer();
    }
    MEASURE_TIME_END(parse);

    if(ArgParse::args.preprocess())
        std::filesystem::remove(preprocessedFilePath);

    MEASURE_TIME_START(semanalyze);
    for(int i = 0; i<iterations; i++){
        SemanticAnalysis::reset();
        SemanticAnalysis::analyze(*ast);
    }
    MEASURE_TIME_END(semanalyze);

    if(parser.failed || SemanticAnalysis::failed){
        return ExitCode::ERROR_SYNTAX;
    }

    bool genSuccess = false;

    double codegenSeconds{0.0};
    double iselSeconds{0.0};
    double regallocSeconds{0.0};
    double asmSeconds{0.0};

    if(parsedArgs.contains(ArgParse::args.dot) || parsedArgs.contains(ArgParse::args.url)){
        if(ArgParse::args.url()){
            std::stringstream ss;
            ast->printDOT(ss);
            auto compactSpacesRegex = std::regex("\\s+");
            auto str = std::regex_replace(ss.str(), compactSpacesRegex, " ");
            std::cout << "https://dreampuf.github.io/GraphvizOnline/#" << url_encode(str) << std::endl;
        } else {
            assert(ArgParse::args.dot());
            if(ArgParse::args.output()){
                std::ofstream outputFile{*ArgParse::args.output};
                ast->printDOT(outputFile);
                outputFile.close();
            }else{
                ast->printDOT(std::cout);
            }
        }
    }else {
        std::error_code errorCode;
        std::string outfile = ArgParse::args.output() ? *ArgParse::args.output : "";
        llvm::raw_fd_ostream llvmOut = llvm::raw_fd_ostream(outfile == ""? "-" : outfile, errorCode);
        llvm::raw_fd_ostream devNull = llvm::raw_fd_ostream("/dev/null", errorCode);

        MEASURE_TIME_START(codegen);
        if(!(genSuccess = Codegen::generate(*ast, ArgParse::args.llvm() ? &llvmOut : nullptr))){
            llvm::errs() << "Generating LLVM-IR failed :(\nIndividual errors displayed above\n";
            goto continu;
        }
        MEASURE_TIME_END(codegen);
        codegenSeconds = MEASURED_TIME_AS_SECONDS(codegen, 1);

        if(ArgParse::args.output()){
            // adapted from https://www.llvm.org/docs/tutorial/MyFirstLanguageFrontend/LangImpl08.html
            llvm::InitializeAllTargetInfos();
            llvm::InitializeAllTargets();
            llvm::InitializeAllTargetMCs();
            llvm::InitializeAllAsmParsers();
            llvm::InitializeAllAsmPrinters();

            auto targetTriple = llvm::sys::getDefaultTargetTriple();
            Codegen::moduleUP->setTargetTriple(targetTriple);

            std::string error;
            auto target = llvm::TargetRegistry::lookupTarget(targetTriple, error);

            // Print an error and exit if we couldn't find the requested target.
            // This generally occurs if we've forgotten to initialise the
            // TargetRegistry or we have a bogus target triple.
            if (!target) {
                llvm::errs() << error;
                goto continu;
            }

            auto CPU = "generic";
            auto features = "";
            auto tempObjFileName = *ArgParse::args.output + ".o-XXXXXX";
            auto tempObjFileFD = mkstemp(tempObjFileName.data());

            llvm::TargetOptions opt;
            auto RM = llvm::Optional<llvm::Reloc::Model>();

            // For some reason, the targetMachine needs to be deleted manually, so encapsulate it in a unique_ptr
            auto targetMachineUP = std::unique_ptr<llvm::TargetMachine>(target->createTargetMachine(targetTriple, CPU, features, opt, RM));

            auto DL = targetMachineUP->createDataLayout();
            Codegen::moduleUP->setDataLayout(DL);

            {
                std::error_code ec;
                llvm::raw_fd_ostream dest(tempObjFileFD, true);

                if(ec){
                    llvm::errs() << "Could not open file: " << ec.message() << "\n";
                    goto continu;
                }

                // old pass manager for backend
                llvm::legacy::PassManager pass;
                auto fileType = llvm::CGFT_ObjectFile;

                if(targetMachineUP->addPassesToEmitFile(pass, dest, nullptr, fileType)){
                    llvm::errs() << "TargetMachine can't emit a file of this type" << "\n";
                    goto continu;
                }


                pass.run(*Codegen::moduleUP);
                dest.flush();
            }

            auto ex = [](auto p) {return std::filesystem::exists(p);};

            // link
            if(ex("/lib/ld-linux-x86-64.so.2") && ex("/lib/crt1.o") && ex("/lib/crti.o") && ex("/lib/crtn.o")){
                if(auto ret = execute(
                            "ld",
                            "-o", (*ArgParse::args.output).c_str(),
                            tempObjFileName.c_str(),
                            "--dynamic-linker", "/lib/ld-linux-x86-64.so.2", "-lc", "/lib/crt1.o", "/lib/crti.o", "/lib/crtn.o");
                        ret != ExitCode::SUCCESS)
                    return ret;
            }else{
                // just use gcc in that case
                if(auto ret = execute(
                            "gcc",
                            "-lc",
                            "-o", (*ArgParse::args.output).c_str(),
                            tempObjFileName.c_str());
                        ret != ExitCode::SUCCESS)
                    return ret;
            }

            std::filesystem::remove(tempObjFileName);

        }else if(!ArgParse::args.llvm()){
            // isel
            MEASURE_TIME_START(isel);
            Codegen::ISel::doISel(ArgParse::args.isel() ? &llvmOut : nullptr);
            MEASURE_TIME_END(isel);
            iselSeconds = MEASURED_TIME_AS_SECONDS(isel, 1);

            // regalloc
            MEASURE_TIME_START(regalloc);
            Codegen::RegAlloc::doRegAlloc(ArgParse::args.regalloc() ? &llvmOut : nullptr);
            MEASURE_TIME_END(regalloc);
            regallocSeconds = MEASURED_TIME_AS_SECONDS(regalloc, 1);

            // asm
            MEASURE_TIME_START(asm);
            (Codegen::AssemblyGen(ArgParse::args.asmout() ? &llvmOut : &devNull)).doAsmOutput();
            MEASURE_TIME_END(asm);
            asmSeconds = MEASURED_TIME_AS_SECONDS(asm, 1);
        }
    }
continu:

    //print execution times
    if(ArgParse::args.benchmark()){
        std::cout
        << "Average parse time (over "                  << iterations << " iterations): " << MEASURED_TIME_AS_SECONDS(parse, iterations)      << "s\n"
        << "Average semantic analysis time: (over "     << iterations << " iterations): " << MEASURED_TIME_AS_SECONDS(semanalyze, iterations) << "s\n"
        << "Average codegeneration time: (over "        << 1          << " iterations): " << codegenSeconds                                   << "s\n"
        << "Average instruction selection time: (over " << 1          << " iterations): " << iselSeconds                                      << "s\n"
        << "Average register allocation time: (over "   << 1          << " iterations): " << regallocSeconds                                  << "s\n"
        << "Average assembly generation time: (over "   << 1          << " iterations): " << asmSeconds                                       << "s\n"
        << "AST Memory usage: "                         << 1e-6*(ast->getRoughMemoryFootprint())                                              << "MB\n";
    }

    if(!genSuccess) return ExitCode::ERROR_CODEGEN;

    return ExitCode::SUCCESS;
}
