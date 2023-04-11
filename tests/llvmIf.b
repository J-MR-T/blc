// RUN: %blc -lE %s 2>&1 | FileCheck %s
// RUN: %blc -lE %s %t-1; %blc -aE %s | aarch64-linux-gnu-gcc -g -x assembler -o %t - && qemu-aarch64 -L /usr/aarch64-linux-gnu %t | diff -y <(%t-1) -

#include "lib.b"

// CHECK-NOT: Warning
fn(x, y){
    register z = x;
    register a = 0;
    if((a = z*x) > y){
        return a;
    }else{
        return fn2(a, y);
    }
}


fn2(x, y){
    register z = x;
    register a = 0;
    register retval = 0;
    if((a = z*x) > y){
        retval = a;
    }else{
        retval = fn3(a, y);
    }
    return retval;
}

fn3(x, y){
    auto z = x;
    auto a = 0;
    if((a = z*x) > y){
        z = 55;
        return a;
    }else{
        auto a = a*z*x*y;
        auto b = y;
        auto c = x;
        auto d = z;
        auto e = 20;

        PRINT_CUR_LINE()
        printnum(a);
        printnum(c);
        printnum(e);

        println();

        return fn4(a, b, c, d, e);
    }
}

fn4(x, y, hihi, hoho, haha){
    auto z = x;
    auto a = 0;

    if(haha == 0) return 0;

    if((((a = z*x || a)*a + (z && (a && a*a))*z) > y)){
        PRINT_CUR_LINE()
        if(a+hihi) a = hihi; else a = hoho;
        if(y+hoho){
            if(y+haha){

            }
            {
                if(x) return a;
            }
        }else{
            if(1) 0; else if(2) 1; else if(3){
                42;
            }else{
                9;
            }
        }
        if(a) return a; else return x;
        return a;
    }else if(5) {
        PRINT_CUR_LINE()
        return x  >> 13 % haha-2;
    }else
    return 50;

    return 0;
}

main(){
    // try all kinds of different combinations of these functions

    register i =0;
    while(i < 500){
        srand(i);
        
        printnum(fn(i*5, i%3));
        printnum(fn2(i+20, i/3));
        printnum(fn3(i*i, i/5));
        printnum(fn4(i << 2, i, i-1, i*50, i/2));

        printnum(fn(rand(), rand()));
        printnum(fn2(rand(), rand()));
        printnum(fn3(rand(), rand()));
        printnum(fn4(rand(), rand(), rand(), rand(), rand()));

        i = i+1;
    }

    return 0;
}
