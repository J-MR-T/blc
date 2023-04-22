// RUN: %FileCheckWithLLVMBackend %s
// RUN: %FileCheckWithARMBackend %s

#include "lib.b"

baar(){
    return (1 && 2 && 3 && 4) + (5 && 6 && 7 || 8) + (9 || 0) + (1 && 0) - (2 || 1) * (1 && 10) + ((1 || 0) && (0 || 0)); // should yield 2
}

main(){
    printnum(baar());
    // CHECK: 2
    return 0;
}
