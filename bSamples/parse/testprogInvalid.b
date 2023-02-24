    //hello
    gauss(x) {
        registr res = -0; //wrong
        while (x > 0) {
            res = res + x //wrong
            x = y - 1 -; //wrong
        }
        return res;
    }

reallyComplexExpr(x,y,z){
    return (1=(y = x>y ><> z + -x - (y*x/(z+=x+<x*2) == 5&2)^2 || 0)); //wrong
}

//me is comment
    ifTest(x) {
        if (x < -5)
            return;//  me evil
        x = x + 3;
    }//hihi
    //muhahah

    isBool(x) { return !!x == x; }

    callTest(a b) { //wrong
        register c = foo(a, b);
        auto c = b; //wrong
        a = &a[a@4]; // okay
        return bar(c, a) + baf(a) + baz(c);
    }

    baz(a, b) { return } //wrong

    unreachableCode(a) {
        else (a > 0) return a; //wrong
        else return -a;
        return a + 1;
    }

    foo(a, b) {
        a[b] = b;
        1+2=x; //wrong
        x = &(1+1); //wrong
        return a[b] + a[b@1];
    }

    addrof(ptr) {
        auto var = 1;
        ptr[1] = &var;
        register ptr2 = &ptr[1];
        ptr2[0] = 2;
        auto c = &ptr; //wrong
        return var;
    }
