test(){
    return;
}

fnA(){ return 0; }
fnB(){ return 0; }
fnC(){ return 0; }

foo(){
    test();
    return !fnA() && fnB() || fnC();
}

bar(){
    return (1 && 2 && 3 && 4) + (5 && 6 && 7 || 8) + (9 || 0) + (0 && 0); // should yield 3
}


fnD(x, y){
    register z = x;
    register a = 0;
    if((a = z*x) > y){
        return a;
    }else{
        return fn(a, y);
    }
}


fnE(x, y){
    register z = x;
    register a = 0;
    register retval = 0;
    if((a = z*x) > y){
        retval = a;
    }else{
        retval = fnnnnnnn(a, y);
    }
    return retval;
}

fnF(x, y){
    auto z = x;
    auto a = 0;
    if((a = z*x) > y){
        z = 55;
        return a;
    }else{
        return fn(a, y);
    }
}

fnG(a, b, c){
    if (a)
        return a;
    else
        return b;
    return c;
}

fnNoArgMatch(a){
    register b = a;
    return fnNoArgMatch(a,b);
}

fnTestForwardDecl(){
    return forwardDecl();
}

forwardDecl(){
    return 0;
}

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

#include "llvmIfInsane.b"
