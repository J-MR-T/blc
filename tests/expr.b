// RUN: %blc -lE %s %t-1; %blc -aE %s | aarch64-linux-gnu-gcc -g -x assembler -o %t - && qemu-aarch64 -L /usr/aarch64-linux-gnu %t | diff -y <(%t-1) -

#include "lib.b"

foo(a,b,c,d){
    return a - (1 << a) >> b + b* c % d-1;
}

main(){
    register i =0;
    register fmt = calloc(100,1);
    fmt[0@1] = 102;
    fmt[1@1] = 111;
    fmt[2@1] = 114;
    fmt[3@1] = 32;
    fmt[4@1] = 105;
    fmt[5@1] = 61;
    fmt[6@1] = 37;
    fmt[7@1] = 100;
    // 'for i=%d'

    while(i < 500){
        //srand(i);
        printf(fmt, i);
        println();
        
        printnum(foo(i*5, i%3, i%4, i%5+1));
        printnum(foo(i+20, i/3, i*5, i-4 + 5));
        printnum(foo(i*i, i/5, i + 5, i*2 +2));
        printnum(foo(i << 2, i, i-1, i*50 +2));

        i = i+1;
    }

    return 0;
}
