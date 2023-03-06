// RUN: %blc -l %s %t && %t | FileCheck %s
// RUN: %blc %s 2>&1 | not grep Warn

println(){
    register fmt = calloc(2,1);
    fmt[0@1] = 10;
    printf(fmt);
    free(fmt);
    return 0;
}

getIntFmtString(){
    register fmt = calloc(4,1);
    fmt[0@1] = 37; // %
    fmt[1@1] = 100; // i
    fmt[2@1] = 10; // \n
    return fmt;
}

fnA(fmt){
    auto a = 2;
    printf(fmt, a);
    {
        a = 3;
        printf(fmt, a);
    }
    printf(fmt, a);

    println(); // expected: 2,3,3

    return 0;
}

fnB(fmt){
    register b = 2;
    printf(fmt, b);
    {
        b = 3;
        printf(fmt, b);
    }
    printf(fmt, b); // expected: 2,3,3

    println();
    return 0;
}

fnC(fmt){
    auto c = 2;
    printf(fmt, c);
    {
        auto c = 3;
        printf(fmt, c);
    }
    printf(fmt, c); // expected: 2,3,2
    println();

    return 0;
}

fnD(fmt){
    auto i = 2;
    printf(fmt, i);
    {
        auto i = 3;
        printf(fmt, i);
        {
            auto i = 4;
            printf(fmt, i);
            {
                i = 5;
                printf(fmt, i);
            }
            printf(fmt, i);
        }
        printf(fmt, i);
    }
    printf(fmt, i); // expected: 2, 3, 4, 5, 5, 3, 2
    println();

    return 0;
}

fnE(fmt){
    auto e = 2;
    printf(fmt, e);
    {
        register e = e+1;
        printf(fmt, e);
        {
            auto e = e+1;
            printf(fmt, e);
            {
                e = e+1;
                printf(fmt, e);
            }
            printf(fmt, e);
        }
        printf(fmt, e);
    }
    printf(fmt, e); // expected: 2, 3, 4, 5, 5, 3, 2
    println();

    return 0;
}

fnF(fmt){
    auto f = 2;
    printf(fmt, f);
    {
        register f = 3;
        printf(fmt, f);
        {
            auto f = 4;
            printf(fmt, f);
            {
                f = 5;
                printf(fmt, f);
            }
            printf(fmt, f);
        }
        printf(fmt, f);
    }
    printf(fmt, f); // expected: 2, 3, 4, 5, 5, 3, 2

    println();
    return 0;
}

fnG(fmt,g){
    printf(fmt, g);
    {
        auto g = 3;
        printf(fmt, g);
        {
            register g = 4;
            printf(fmt, g);
            {
                g = 5;
                printf(fmt, g);
            }
            printf(fmt, g);
        }
        printf(fmt, g);
    }
    printf(fmt, g); // expected: 2,3,4,5,5,3,2

    println();
    return 0;
}

fnH(fmt,x,y){
    auto a = 2;
    if(x==y){
        register a = 3;
    }else{
        auto a = 4;
    }
    printf(fmt, a);
    return 0;
}

// TODO a similar thing to fnD, but then save the address of one of the inner things into an outer one, dereference it and print that

fnI(fmt){
    auto i = 2;
    auto iAddr = &i;
    printf(fmt, i);
    {
        register i = 3;
        printf(fmt, i);
        {
            auto i = 4;
            printf(fmt, i);
            {
                i=5;
                printf(fmt, iAddr[0]+3);
            }
            printf(fmt, i);
        }
        printf(fmt, i);
    }
    printf(fmt, i); // expected: 2, 3, 4, 5, 5, 3, 2
    println();

    return 0;
}

fnJ(x){
    {
        register a =0;
        a = a+x;
    }

    {
        register a =0;
        x = a+x;
    }
    return 0;
}

main(){
    register fmt = getIntFmtString();

    fnA(fmt);
    // CHECK: 2
    // CHECK-NEXT: 3
    // CHECK-NEXT: 3
    // CHECK-EMPTY:
    fnB(fmt);
    // CHECK-NEXT: 2
    // CHECK-NEXT: 3
    // CHECK-NEXT: 3
    // CHECK-EMPTY:
    fnC(fmt);
    // CHECK-NEXT: 2
    // CHECK-NEXT: 3
    // CHECK-NEXT: 2
    // CHECK-EMPTY:
    fnD(fmt);
    // CHECK-NEXT: 2
    // CHECK-NEXT: 3
    // CHECK-NEXT: 4
    // CHECK-NEXT: 5
    // CHECK-NEXT: 5
    // CHECK-NEXT: 3
    // CHECK-NEXT: 2
    // CHECK-EMPTY:
    fnE(fmt);
    // CHECK-NEXT: 2
    // CHECK-NEXT: 3
    // CHECK-NEXT: 4
    // CHECK-NEXT: 5
    // CHECK-NEXT: 5
    // CHECK-NEXT: 3
    // CHECK-NEXT: 2
    // CHECK-EMPTY:
    fnF(fmt);
    // CHECK-NEXT: 2
    // CHECK-NEXT: 3
    // CHECK-NEXT: 4
    // CHECK-NEXT: 5
    // CHECK-NEXT: 5
    // CHECK-NEXT: 3
    // CHECK-NEXT: 2
    // CHECK-EMPTY:
    fnG(fmt,2);
    // CHECK-NEXT: 2
    // CHECK-NEXT: 3
    // CHECK-NEXT: 4
    // CHECK-NEXT: 5
    // CHECK-NEXT: 5
    // CHECK-NEXT: 3
    // CHECK-NEXT: 2
    // CHECK-EMPTY:
    fnH(fmt, 1, 1);
    // CHECK-NEXT: 2
    fnH(fmt, 2, 1);
    // CHECK-NEXT: 2
    println();
    // CHECK-EMPTY:
    fnI(fmt);
    // CHECK-NEXT: 2
    // CHECK-NEXT: 3
    // CHECK-NEXT: 4
    // CHECK-NEXT: 5
    // CHECK-NEXT: 5
    // CHECK-NEXT: 3
    // CHECK-NEXT: 2

    return 0;
}
