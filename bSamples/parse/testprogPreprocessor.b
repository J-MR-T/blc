#define HELLO(a,b,c)\
    if(a) { \
        b; \
    } else { \
        c; \
    } 

myfunc(x){
    auto a = 1;
    HELLO(a,return 5, return 6)
}
