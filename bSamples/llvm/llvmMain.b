fib(n){
    if(n <= 2) return 1;
    return fib(n-1) + fib(n-2);
}

bar(){
    return (1 && 2 && 3 && 4) + (5 && 6 && 7 || 8) + (9 || 0) + (0 && 0); // should yield 3
}

main(argc, argv){
    register fmt = calloc(16, 1);
    fmt[0@1] = 65;
    fmt[1@1] = 66;
    fmt[2@1] = 67;
    fmt[3@1] = 10; // newline

    fmt[4@1] = 37; // %
    fmt[5@1] = 100; // d
    fmt[6@1] = 10;
    printf(fmt, fib(20));

    // remove first 4 chars (ABC\n)
    fmt[0@1] = 
    fmt[1@1] = 
    fmt[2@1] = 32;

    fmt[7@1] = 37;
    fmt[8@1] = 100;
    fmt[9@1] = 10;
    printf(fmt, bar(), fib(22));

    printf(fmt, (!0), (!1));

    return 0;
}
