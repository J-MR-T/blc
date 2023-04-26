// RUN: %FileCheckWithLLVMBackend %s
// RUN: %FileCheckWithARMBackend %s
// RUN: %FileCheckWithMLIRBackend %s

println(){
    register fmt = calloc(2,1);
    fmt[0@1] = 10;
    printf(fmt);
    // TODO this free call trips up the llvm ir conversion for some reason
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
