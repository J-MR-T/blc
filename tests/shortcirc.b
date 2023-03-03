// RUN: %blc -lE %s %t && %t | FileCheck %s
// RUN: %blc -aE %s | aarch64-linux-gnu-gcc -g -x assembler -o %t1 - && qemu-aarch64 -L /usr/aarch64-linux-gnu %t1 | FileCheck %s

#include "lib.b"

foo(){
    return (5 && 6 && 7 || 8) + (0 && 0); // should yield 1
}

bar(){
    return (1 && 2 && 3 && 4) + (5 && 6 && 7 || 8) + (9 || 0) + (0 && 0); // should yield 3
}

baz(){
    return (0 && 0); // should yield 0
}

main(){
    printnum(foo());
    // CHECK: 1
    printnum(bar());
    // CHECK-NEXT: 3
    printnum(baz());
    // CHECK-NEXT: 0
    return 0;
}
