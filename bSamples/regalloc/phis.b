criticalEdge(x){
    register a = 0;
    if(x > 0){
        a = 1;
    }
    return a;
}

cycle(x){
    register a = 0;
    register b = 1;
    while(1){
        register c = b;
        b = a;
        a = c;
    }
    return a;
}
