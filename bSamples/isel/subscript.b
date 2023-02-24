foo(x){
    auto a = x;
    register aPtr = &a;
    register b = aPtr[3@8];
    aPtr[2@1] = x;
    aPtr[1@8] = 0;
    return aPtr[0@2];
}
