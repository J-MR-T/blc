

fnA(){ 
    register fmt = calloc(4, 1);
    fmt[0@1] = 65;
    fmt[1@1] = 10;
    printf(fmt);

    return 0;
}
fnB(){
    register fmt = calloc(4, 1);
    fmt[0@1] = 66;
    fmt[1@1] = 10;
    printf(fmt);

    return 0;
}
fnC(){
    register fmt = calloc(4, 1);
    fmt[0@1] = 67;
    fmt[1@1] = 10;
    printf(fmt);

    return 0;
}

main(x, z){
    auto a = 0;
    (a = ((z*x) || a));


    //register a = 0;

    //return fnA() && fnB();
    //return fnA() && fnB() || fnC();
    //return (a = fnA()) && (a && fnB()) || (fnC() || a || fnB());
}
