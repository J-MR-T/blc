import lit.formats
import os

# TODO find an option to automatically set -j1 per default for lit

config.name = "B-like compiler"
config.test_format = lit.formats.ShTest(True)

config.suffixes = ['.b']

config.excludes = ['bSamples']

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.test_source_root, 'tests')

config.substitutions.append(("%blc", os.path.join(config.test_source_root, "blc")))
config.substitutions.append(("%FileCheckWithLLVMBackend" , "%blc -l %s %t && %t | FileCheck"))
config.substitutions.append(("%FileCheckWithMLIRBackend" , "%blc -m %s %t && %t | FileCheck"))
config.substitutions.append(("%FileCheckWithARMBackend", r"%blc -a %s | aarch64-linux-gnu-gcc -g -x assembler -o %t - && qemu-aarch64 -L /usr/aarch64-linux-gnu %t | FileCheck"))
config.substitutions.append(("%compareBothBackends", r"%blc -l %s %t-llvm; %blc -m %s %t-mlir; %blc -a %s | aarch64-linux-gnu-gcc -g -x assembler -o %t-arm - && diff <(%t-llvm) <(qemu-aarch64 -L /usr/aarch64-linux-gnu %t-arm)"))

config.recursiveExpansionLimit = 3
