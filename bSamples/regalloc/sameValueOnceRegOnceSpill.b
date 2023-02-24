foo(x, n){
    while(1){
        register y = x + 1;

        // spill everything on purpose here
        func();

        register z = x + 2;
    }
}
