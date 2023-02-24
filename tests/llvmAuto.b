// RUN: %bc -l %s 2>&1 | FileCheck %s

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
