// RUN: %bc -l %s %t && %t | FileCheck %s
// RUN: %bc -a %s | aarch64-linux-gnu-gcc -g -x assembler -o %t - && qemu-aarch64 -L /usr/aarch64-linux-gnu %t | FileCheck %s

println(){
    register fmt = calloc(2,1);
    fmt[0@1] = 10;
    printf(fmt);
    free(fmt);
    return 0;
}

main(){
    register n = 0;
    while(n < 10){
        putchar(n + 48);
        n = n + 1;
    }
    println();
    // CHECK: 0123456789
    return 0;
}
