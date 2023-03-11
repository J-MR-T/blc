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
