// RUN: %FileCheckWithLLVMBackend %s
// RUN: %FileCheckWithARMBackend %s
// RUN: %FileCheckWithMLIRBackend %s

/// prints a number and a newline
printnum(n){
    register fmt = calloc(4,1);
    fmt[0@1] = 37; // %
    fmt[1@1] = 100; // d
    fmt[2@1] = 10;  // \n
    printf(fmt, n);
    free(fmt);
    return 0;
}

foo(){
    return (5 && 6 && 7 || 8) + (0 && 0); // should yield 1
}

bar(){
    return (1 && 2 && 3 && 4) + (5 && 6 && 7 || 8) + (9 || 0) + (0 && 0); // should yield 3
}

baar(){
    return (1 && 2 && 3 && 4) + (5 && 6 && 7 || 8) + (9 || 0) + (1 && 0) - (2 || 1) - (0 && 10) + ((1 || 0) && (0 || 0)); // should yield 2
}

baz(){
    return (0 && 0); // should yield 0
}

main(){
    printnum(foo()); // CHECK: 1
    printnum(bar()); // CHECK-NEXT: 3
    printnum(baar()); // CHECK-NEXT: 2
    printnum(baz()); // CHECK-NEXT: 0
    return 0;
}
