// RUN: %blc -l %s 2>&1 | not grep Warn

/// only prints a newline
println(){
    register fmt = calloc(2,1);
    fmt[0@1] = 10;
    printf(fmt);
    free(fmt);
    return 0;
}

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

randUpTo(n){
    return rand() % n;
}
