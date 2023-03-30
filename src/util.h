#pragma once

#include <concepts>
#include <iostream>
#include <iomanip>
#include <string>
#include <regex>
#include <sys/wait.h>
#include <unistd.h>

#pragma GCC diagnostic push 
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wcomment"
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>
#pragma GCC diagnostic pop

using std::string;
using std::string_view;
using std::unique_ptr;
using namespace std::literals::string_literals;

#ifndef NDEBUG
#define DEBUGLOG(x) llvm::errs() << x << "\n"; fflush(stderr);
#define IFDEBUG(x) x
#define IFDEBUGELSE(x, y) x

#else

#define DEBUGLOG(x, ...)
#define IFDEBUG(x)
#define IFDEBUGELSE(x, y) y

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

    extern std::map<Arg, std::string> parsedArgs;
    
    // struct for all possible arguments
    const struct {
        const Arg help{      "h", "help"      , 0, "Show this help message and exit"                                                                     , false, true};
        const Arg input{     "i", "input"     , 1, "Input file"                                                                                          ,  true, false};
        const Arg dot{       "d", "dot"       , 0, "Output AST in GraphViz DOT format (to stdout by default, or file using -o) (overrides -p)"           , false, true};
        const Arg output{    "o", "output"    , 2, "Output file"                                                                                         , false, false};
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

    inline bool Arg::operator()() const{
        return parsedArgs.contains(*this);
    }

    inline std::string& Arg::operator*() const{
        return parsedArgs[*this];
    }

    void printHelp(const char *argv0);

    //unordered_map doesnt work because of hash reasons (i think), so just define <, use ordered
    std::map<Arg, std::string>& parse(int argc, char *argv[]);

} // end namespace ArgParse

// like https://www.llvm.org/docs/ProgrammersManual.html#dss-sortedvectormap recommends, use a sorted vector for strict insert then query map (this is even a subset of that, it doesn't support inserting after building at all)
template<std::totally_ordered K, typename V>
struct InsertOnceQueryAfterwardsMap{
	using ElemPairType = typename std::pair<K,V>;

	static int compare(const ElemPairType& elem1, const ElemPairType& elem2){
		return elem1.first < elem2.first;
	}

	llvm::SmallVector<ElemPairType> vec;

	InsertOnceQueryAfterwardsMap() = default;

	InsertOnceQueryAfterwardsMap(const llvm::ArrayRef<ElemPairType> &arr) : vec(arr){
		std::sort(vec.begin(), vec.end(), compare);
	}

    /// only supports lookup of actually inserted items, will segfault otherwise
	const V& operator[](const K &key) const{
        assert(std::is_sorted(vec.begin(), vec.end(), compare) && "InsertOnceQueryAfterwardsMap not sorted");

		auto it = std::lower_bound(vec.begin(), vec.end(), ElemPairType{key,V{}}, compare); // the V{} is just a dummy value, it will be ignored
		assert((it != vec.end() && it->first == key) && "Item from InsertOnceQueryAfterwardsMap not found");
		return it->second;
	}

    /// at
    /*
    std::optional<const V&> at(const K &key) const{
        assert(std::is_sorted(vec.begin(), vec.end(), compare) && "InsertOnceQueryAfterwardsMap not sorted");

        auto it = std::lower_bound(vec.begin(), vec.end(), ElemPairType{key,V{}}, compare); // the V{} is just a dummy value, it will be ignored
        if(it != vec.end() && it->first == key)
            return {it->second};
        else
            return {};
    }
    */

    bool contains(const K &key) const{
        assert(std::is_sorted(vec.begin(), vec.end(), compare) && "InsertOnceQueryAfterwardsMap not sorted");

        auto it = std::lower_bound(vec.begin(), vec.end(), ElemPairType{key,V{}}, compare); // the V{} is just a dummy value, it will be ignored
        return it != vec.end() && it->first == key;
    }

	// expose iterators

	typename llvm::SmallVector<ElemPairType>::iterator begin(){
		return &*vec.begin();
	}

	typename llvm::SmallVector<ElemPairType>::iterator end(){
		return &*vec.end();
	}
};

// explicit instantiation to catch errors
template struct InsertOnceQueryAfterwardsMap<int, llvm::SmallString<32>>;

string url_encode(const string& value);

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

int llvmCompileAndLinkMod(llvm::Module& mod);
