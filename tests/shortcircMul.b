// RUN: %blc -a %s | aarch64-linux-gnu-gcc -g -x assembler -o %t1 - && qemu-aarch64 -L /usr/aarch64-linux-gnu %t1 | FileCheck %s

baar(){
    return (1 && 2 && 3 && 4) + (5 && 6 && 7 || 8) + (9 || 0) + (1 && 0) - (2 || 1) * (1 && 10) + ((1 || 0) && (0 || 0)); // should yield 2
}

main(){
    printnum(baar());
    // CHECK-NEXT: 2
    return 0;
}
