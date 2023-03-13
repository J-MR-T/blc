#include <filesystem>

#pragma GCC diagnostic push 
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wcomment"
#include <llvm/Config/llvm-config.h>

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

#include "util.h"

namespace ArgParse{

std::map<Arg, string> parsedArgs{};

void printHelp(const char *argv0) {
    std::cerr << "A Compiler for a B like language" << std::endl;
    std::cerr << "Usage: " << std::endl;
    for (auto &arg : args) {
        std::cerr << "  ";
        if (arg.shortOpt != "")
            std::cerr << "-" << arg.shortOpt;

        if (arg.longOpt != "") {
            if (arg.shortOpt != "")
                std::cerr << ", ";

            std::cerr << "--" << arg.longOpt;
        }

        if (arg.pos != 0)
            std::cerr << " (or positional, at position " << arg.pos << ")";
        else if (arg.flag)
            std::cerr << " (flag)";

        std::cerr << "\n    "
            // string replace all \n with \n \t here
            << std::regex_replace(arg.description, std::regex("\n"), "\n    ")
            << std::endl;
    }

    std::cerr << "\nExamples: \n"
        << "  " << argv0 << " -i input.b -d -o output.dot\n"
        << "  " << argv0 << " input.b -d output.dot\n"
        << "  " << argv0 << " input.b -du\n"
        << "  " << argv0 << " -lE input.b\n"
        << "  " << argv0 << " -l main.b main\n"
        << "  " << argv0 << " -ls input.b\n"
        << "  " << argv0 << " -sr input.b\n"
        << "  " << argv0
        << " -a bSamples/asm/addressCalculations.b | aarch64-linux-gnu-gcc "
        "-g -x assembler -o test - && qemu-aarch64 -L "
        "/usr/aarch64-linux-gnu test hi\\ there\n";
}

std::map<Arg, std::string>& parse(int argc, char *argv[]) {
    std::stringstream ss;
    ss << " ";
    for (int i = 1; i < argc; ++i) {
        ss << argv[i] << " ";
    }

    string argString = ss.str();

    // handle positional args first, they have lower precedence
    // find them all, put them into a vector, then match them to the possible args
    std::vector<string> positionalArgs{};
    for (int i = 1; i < argc; ++i) {
        for (const auto &arg : args) {
            if (!arg.flag && (("-" + arg.shortOpt) == string{argv[i - 1]} ||
                        ("--" + arg.longOpt) == string{argv[i - 1]})) {
                // the current arg is the value to another argument, so we dont count it
                goto cont;
            }
        }

        if (argv[i][0] != '-') {
            // now we know its a positional arg
            positionalArgs.emplace_back(argv[i]);
        }
cont:
        continue;
    }

    for (const auto &arg : args) {
        if (arg.pos != 0) {
            // this is a positional arg
            if (positionalArgs.size() > arg.pos - 1) {
                parsedArgs[arg] = positionalArgs[arg.pos - 1];
            }
        }
    }

    bool missingRequired = false;

    // long/short/flags
    for (const auto &arg : args) {
        if (!arg.flag) {
            std::regex matchShort{" -" + arg.shortOpt + "\\s*([^\\s]+)"};
            std::regex matchLong{" --" + arg.longOpt + "(\\s*|=)([^\\s=]+)"};
            std::smatch match;
            if (arg.shortOpt != "" &&
                    std::regex_search(argString, match, matchShort)) {
                parsedArgs[arg] = match[1];
            } else if (arg.longOpt != "" &&
                    std::regex_search(argString, match, matchLong)) {
                parsedArgs[arg] = match[2];
            } else if (arg.required && !parsedArgs.contains(arg)) {
                std::cerr << "Missing required argument: -" << arg.shortOpt << "/--"
                    << arg.longOpt << std::endl;
                missingRequired = true;
            }
        } else {
            std::regex matchFlagShort{" -[a-zA-z]*" + arg.shortOpt};
            std::regex matchFlagLong{" --" + arg.longOpt};
            if (std::regex_search(argString, matchFlagShort) ||
                    std::regex_search(argString, matchFlagLong)) {
                parsedArgs[arg] =
                    ""; // empty string for flags, will just be checked using .contains
            }
        };
    }

    if (missingRequired) {
        printHelp(argv[0]);
        exit(ExitCode::ERROR);
    }
    return parsedArgs;
}

} // end namespace ArgParse

// taken from https://stackoverflow.com/a/17708801
string url_encode(const string &value) {
    std::ostringstream escaped;
    escaped.fill('0');
    escaped << std::hex;

    for (string::const_iterator i = value.begin(), n = value.end(); i != n;
         ++i) {
        string::value_type c = (*i);

        // Keep alphanumeric and other accepted characters intact
        if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
            escaped << c;
            continue;
        }

        // Any other characters are percent-encoded
        escaped << std::uppercase;
        escaped << '%' << std::setw(2) << int((unsigned char)c);
        escaped << std::nouppercase;
    }

    return escaped.str();
}

/// returns -1 or return val of subprocesses on error, 0 on success
int llvmCompileAndLinkMod(llvm::Module& mod){
    using namespace ArgParse;

    // adapted from https://www.llvm.org/docs/tutorial/MyFirstLanguageFrontend/LangImpl08.html
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllAsmPrinters();

    auto targetTriple = llvm::sys::getDefaultTargetTriple();
    mod.setTargetTriple(targetTriple);

    std::string error;
    auto target = llvm::TargetRegistry::lookupTarget(targetTriple, error);

    // Print an error and exit if we couldn't find the requested target.
    // This generally occurs if we've forgotten to initialise the
    // TargetRegistry or we have a bogus target triple.
    if (!target) {
        llvm::errs() << error;
        return -1;
    }

    auto CPU = "generic";
    auto features = "";
    auto tempObjFileName = *args.output + ".o-XXXXXX";
    auto tempObjFileFD = mkstemp(tempObjFileName.data());

    llvm::TargetOptions opt;
    auto RM = llvm::Optional<llvm::Reloc::Model>();

    // For some reason, the targetMachine needs to be deleted manually, so encapsulate it in a unique_ptr
    auto targetMachineUP = std::unique_ptr<llvm::TargetMachine>(target->createTargetMachine(targetTriple, CPU, features, opt, RM));

    auto DL = targetMachineUP->createDataLayout();
    mod.setDataLayout(DL);

    {
        std::error_code ec;
        llvm::raw_fd_ostream dest(tempObjFileFD, true);

        if(ec){
            llvm::errs() << "Could not open file: " << ec.message() << "\n";
            return -1;
        }

        // old pass manager for backend
        llvm::legacy::PassManager pass;
        auto fileType = llvm::CGFT_ObjectFile;

        if(targetMachineUP->addPassesToEmitFile(pass, dest, nullptr, fileType)){
            llvm::errs() << "TargetMachine can't emit a file of this type" << "\n";
            return -1;
        }


        pass.run(mod);
        dest.flush();
    }

    auto ex = [](auto p) {return std::filesystem::exists(p);};

    // link
    if(ex("/lib/ld-linux-x86-64.so.2") && ex("/lib/crt1.o") && ex("/lib/crti.o") && ex("/lib/crtn.o")){
        if(auto ret = execute(
                    "ld",
                    "-o", (*args.output).c_str(),
                    tempObjFileName.c_str(),
                    "--dynamic-linker", "/lib/ld-linux-x86-64.so.2", "-lc", "/lib/crt1.o", "/lib/crti.o", "/lib/crtn.o");
                ret != ExitCode::SUCCESS)
            return ret;
    }else{
        // just use gcc in that case
        if(auto ret = execute(
                    "gcc",
                    "-lc",
                    "-o", (*args.output).c_str(),
                    tempObjFileName.c_str());
                ret != ExitCode::SUCCESS)
            return ret;
    }

    std::filesystem::remove(tempObjFileName);


    return 0;
}
