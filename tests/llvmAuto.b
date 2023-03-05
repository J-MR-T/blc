// RUN: %blc -l %s 2>&1 | FileCheck %s
// RUN: %blc %s 2>&1 | not grep Warn
// RUN: %blc -l %s %t-1; %blc -a %s | aarch64-linux-gnu-gcc -g -x assembler -o %t - && qemu-aarch64 -L /usr/aarch64-linux-gnu %t | diff <(%t-1) -

// CHECK-NOT: Warning
fn(a,b,c){
    register fmt = calloc(8,1);
    fmt[0@1] = 37;
    fmt[1@1] = 100;
    fmt[2@1] = 10;

    auto x = a;
    auto y = b;
    auto z = c;

    x=y+z;
    y=x+z;
    z=x+y;

    printf(fmt, x);
    printf(fmt, y);
    printf(fmt, z);

    return 0;
}

main(){
    fn(1,2,3);
    fn(-3,2,4);
    fn(1,-1,3);
    fn(0,3,2);
    fn(1,5,3);
    fn(1,4,2);
    fn(8,4,1);
    fn(5,2,3);
    return 0;
}
