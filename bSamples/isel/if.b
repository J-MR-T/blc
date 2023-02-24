fnH(x){
    auto y = x;
    x = y[0@4];
    if(x == 0) return x;
    x = y[0@2];
    if(x == 0) return x;
    x = y[0@1];
    if(x == 0) return x;
    return y;
}
