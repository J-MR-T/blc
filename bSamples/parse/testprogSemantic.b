declBeforeUse(x){
    if(y>x){
        return 1;
    }

    teeeest+1;

    register teeeest = 1;
    teeeest = x+x;


}

assignIdentSubscript(x){
    x = x+x;
    x[2] = 1;
    1+2=x;
}

addrIdentSubscript(x){
    x = &x[x@4]; // okay
    x = &(1+1); //not okay
}

addrRegister(x){
    auto y = &x;
}
