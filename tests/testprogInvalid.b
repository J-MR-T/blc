// RUN: not %blc -n %s 2>&1 | FileCheck %s --check-prefix=CHECK-PARSE 
// RUN: not %blc -n %s 2>&1 | FileCheck %s --check-prefix=CHECK-SEMA
    //hello
    gauss(x) {
        registr res = -0; //wrong
        // CHECK-PARSE: Line 5: Unexpected token: res
        while (x > 0) {
            res = res + x //wrong
            // CHECK-PARSE: Line 10: Unexpected token: x
            x = y - 1 -; //wrong
        }
        return res;
    }

reallyComplexExpr(x,y,z){
    return (1=(y = x>y ><> z + -x - (y*x/(z+=x+<x*2) == 5&2)^2 || 0)); //wrong
    // CHECK-PARSE: Line 16: Unexpected token: <
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
        // CHECK-PARSE: Line 30: Unexpected token: b
        register c = foo(a, b);
        // CHECK-SEMA: Variable 'b' used but not declared
        auto c = b; //wrong
        // CHECK-SEMA: Variable 'b' used but not declared
        a = &a[a@4]; // okay
        return bar(c, a) + baf(a) + baz(c);
    }

    baz(a, b) { return } //wrong
    // CHECK-PARSE: Line 40: Unexpected token: }

    unreachableCode(a) {
        else (a > 0) return a; //wrong
        // CHECK-PARSE: Line 46: Unexpected token: else
        else if (a < 0) return -a; //wrong
        else return -a;
        return a + 1;
    }

    foo(a, b) {
        a[b] = b;
        1+2=x; //wrong
        // CHECK-SEMA: LHS of assignment
        register x = &(1+1); //wrong
        // CHECK-SEMA: LHS of assignment
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
