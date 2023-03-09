// RUN: %blc -lE %s %t-1; %blc -aE %s | aarch64-linux-gnu-gcc -g -x assembler -o %t - && qemu-aarch64 -L /usr/aarch64-linux-gnu %t | diff -y <(%t-1) -

#include "lib.b"

fn4(x, hihi, haha){
    printnum(x);
    printnum(hihi);
    printnum(haha);
    return 0;
}

main(){
    fn4(0,0,20);

    return 0;
}
