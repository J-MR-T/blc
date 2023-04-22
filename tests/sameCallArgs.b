// RUN: %compareBothBackends

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
